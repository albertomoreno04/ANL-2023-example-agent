import logging
import math
import random
from random import randint
import time
from typing import cast, List

from geniusweb.actions.Accept import Accept # Accept Offer (Action with Bid)
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer # Throws Bid
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid # Bid with Issue Values
from geniusweb.issuevalue.Domain import Domain # Name + Issue values
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.group32_rl_opponent import OpponentModel

class Group32Agent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """
    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_made_offer : Bid = None
        self.last_received_bid: Bid = None
        self.received_bids: List[Bid] = []
        self.opponent_model: OpponentModel = None

        # Keep track of all bids from opponent (and yours?)
        self.all_bids : AllBidsList = None
        self.bids_with_utilities : list[tuple[Bid, float]] = None
        self.num_of_top_bids : int = 1
        self.previous_opp_bid = None # Initialised later
        self.best_opponent_bid_offered = False

        self.last_time = None
        self.top_bids_percentage: float = 0.01
        self.opponent_best_bid: Bid = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")


            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )

            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()
            # Initialize all bids
            self.all_bids = AllBidsList(self.domain)

            if self.opponent_model is None:
                # Initialize model
                self.opponent_model = OpponentModel(self.domain)

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]), # Stacked Alternating Offers Protocol
            set(["geniusweb.profile.utilityspace.LinearAdditive"]), # Defines utility space as a Linear Additive function
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.
        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            bid = cast(Offer, action).getBid()
            # Initialize opponent model if it is not yet initialized.
            if self.opponent_model is None:
                # Initialize model
                self.opponent_model = OpponentModel(self.domain)

            reward = self.score_bid(bid)
            self.opponent_model.update(bid, reward) # Update Q-values based on our utility of the current bid
            self.last_received_bid = bid # Update last received bid
            self.received_bids.append(bid)

    def my_turn(self):
        """

        Logic to play in each turn

        :return: action to play next
        """
        if self.last_received_bid is None:
            # If no offer received, return offer with next_bid
            new_bid = self.find_bid()
            action = Offer(self.me, new_bid)
            self.last_made_offer = new_bid
            self.send_action(action)
            return

        progress = self.progress.get(time.time() * 1000)

        counter_bid = self.find_bid()

        last_offer_util = float(self.score_bid(self.last_received_bid))
        new_bid_util = float(self.score_bid(counter_bid))

        opp_util_received = float(self.get_opponent_utility(self.last_received_bid, self.last_time))
        opp_util_counter = float(self.get_opponent_utility(counter_bid, self.last_time))

        nash_received = last_offer_util * opp_util_received
        nash_counter = new_bid_util * opp_util_counter

        # Check acceptance condition
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
        else:
            if progress < 0.3:
                # Remain relatively  self.last_received_bid conservative if still in the exploration phase
                accept_probability = 0.0
            elif progress <= 0.8:
                # Seek for more exploitation than in previous phase
                nash_improvement = (nash_received - nash_counter) / nash_counter if nash_counter > 0 else 0 # Check for improvement in Nash product
                time_pressure = ((progress - 0.8) / 0.2) ** 3.0 if progress >= 0.8 else 0.0 # Increasing time pressure
                base_probability = 0.1 + time_pressure * 0.3 # Base prob
                accept_probability = base_probability * (1 + nash_improvement) # Increase it proportional to the improvement in nash product

                # Add Tit-for-Tat behaviour - Concede if opponent concedes
                if hasattr(self, 'previous_opp_bid') and self.previous_opp_bid is not None:
                    previous_opp_util = float(self.score_bid(self.previous_opp_bid))
                    if last_offer_util > previous_opp_util:
                        accept_probability *= 1.2 # Concede if opponent has conceded
            else:
                gamma = 3.0  # gamma > 1 makes the curve non-linear and steeper near the deadline
                time_pressure = ((progress - 0.8) / 0.2) ** gamma if progress >= 0.8 else 0.0
                accept_probability = 0.5 + time_pressure * 0.3

            # Accept based on the accept probability and whether offers are not dissimilar in utilitity
            if random.random() < accept_probability and abs(last_offer_util - new_bid_util) < 0.05:
                action = Accept(self.me, self.last_received_bid)
            else:
                # If not accepted, make counter offer
                action = Offer(self.me, counter_bid)
                self.last_made_offer = counter_bid

        if hasattr(self, 'last_received_bid') and self.last_received_bid is not None:
            self.previous_opp_bid = self.last_received_bid

        # Send action
        self.send_action(action)

    def accept_condition(self, bid: Bid) -> bool:
        """

        Implements the accept condition

        :param: bid to accept or reject

        :return: bool to accept or reject
        """
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time.time() * 1000)

        bid_utility = self.score_bid(bid)

        next_bid = self.find_bid()
        next_bid_utility = self.score_bid(next_bid)

        AC_next : bool = bid_utility >= next_bid_utility # accepts if the received bid is better for us than our next offer

        T = 0.96
        alpha_threshold = 0.70
        AC_time : bool = (progress >= T) and (bid_utility >= alpha_threshold) # accepts if approaching deadline and bid is good enough

        return AC_next or AC_time # accept if either evaluates to true

    def find_bid(self, eps : float = 0.1) -> Bid:
        """

        Finds next bid to offer

        eps : time pressure decay parameter
        """
        num_of_bids = min(self.all_bids.size(), 30000) # Cap the number of bids to 30,000
        progress = self.progress.get(time.time() * 1000)

        if self.bids_with_utilities is None:
            self.bids_with_utilities = []

            for index in range(num_of_bids):
                bid = self.all_bids.get(index)
                bid_utility = float(self.score_bid(bid))
                self.bids_with_utilities.append((bid, bid_utility))

            self.bids_with_utilities.sort(key=lambda tup: tup[1], reverse=True) # Sort bids based on utility

            # Increase the range of bids as deadline approaches:
            # At first, we want to ensure only our best of the best offers are selected
            # Then, we become more lenient and concede by widening the range
            self.num_of_top_bids = max(5, int(num_of_bids * (self.top_bids_percentage + 0.2 * progress ** (1 / eps))))

        if self.last_received_bid is None:
            # If no bid has been made by opponent, return greedily
            return self.bids_with_utilities[0][0]

        top_bids = self.bids_with_utilities[:self.num_of_top_bids]
        T = 0.95

        # If we need to make agreement, first send their previous best offer
        if (progress > T) and not self.best_opponent_bid_offered:
            self.best_opponent_bid_offered = True
            return self.opponent_best_bid
        # If we are back in this situation, return the best bid for opponent from the top bids for us
        elif progress > T:
            top_bids = [(bid[0], self.get_opponent_utility(bid[0])) for (bid, bid_utility) in top_bids]
            top_bids.sort(key = lambda t : t[1], reverse = True)
            return top_bids[0][0] # Returns best offer for opponent (based on our beliefs) within our top bids

        # Return random offer from our top bids
        picked_ranking = randint(0, math.floor(self.num_of_top_bids) - 1)

        return top_bids[picked_ranking][0]

    def score_bid(self, bid: Bid, alpha : float = 0.95) -> float:
        """

        Score bid

        :param: bid
        :param: alpha: selfishness parameter
        """

        u_self = float(self.get_utility(bid))

        if self.opponent_model is not None:
            u_opp = float(self.opponent_model.get_predicted_utility(bid))
        else:
            u_opp = 0.0

        # Take into account 'selfishness'
        score1 = alpha * u_self + (1.0 - alpha) * u_opp

        gamma = 0.6 # How much to value nash product vs social welfare
        nash_product = u_self * u_opp
        social_welfare = (u_self + u_opp) / 2.0
        score2 = gamma * nash_product + (1.0 - gamma) * social_welfare

        lam = 0.95
        composite_score = lam * score1 + (1 - lam) * score2

        return composite_score

    def get_utility(self, bid : Bid) -> float:
        """
        Returns pure utility based on domain

        :param: bid

        :return: pure utility
        """
        pure_util = float(self.profile.getUtility(bid))
        return float(pure_util)

    def get_opponent_utility(self, bid : Bid, time : float) -> float:
        """
        Returns predicted opponent utility

        :param: bid
        :time: current time

        :return: predicted opponent utility
        """
        if self.opponent_model is not None:
            return self.opponent_model.get_predicted_utility(bid)
        else:
            return 0.0



