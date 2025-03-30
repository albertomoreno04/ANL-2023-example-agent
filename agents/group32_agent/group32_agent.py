import logging
import math
import random
from decimal import Decimal
from random import randint
import time
from typing import cast, TypedDict, List

import numpy as np
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

#from .utils.group32_bayesian_opponent import OpponentModel
from .utils.group32_rl_opponent import OpponentModel

# Loggin Utils
class SessionData(TypedDict):
    # TODO: Additionally keep track of other metrics
    progressAtFinish: float
    utilityAtFinish: float
    didAccept: bool
    isGood: bool
    topBidsPercentage: float
    forceAcceptAtRemainingTurns: float

class DataDict(TypedDict):
    sessions: list[SessionData]


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

        self.data_dict : DataDict = None # For learning adaptations


        self.last_received_bid: Bid = None
        self.received_bids: List[Bid] = []
        self.opponent_model: OpponentModel = None

        # Keep track of all bids from opponent (and yours?)
        self.all_bids : AllBidsList = None
        self.bids_with_utilities : list[tuple[Bid, float]] = None
        self.num_of_top_bids : int = 1
        self.min_util : float = 0.8 # TODO: Adjust value
        self.previous_opp_bid = None # Initialised later

        # Logging helpers
        self.round_times : List[Decimal] = []
        self.last_time = None
        self.avg_time = None
        self.utility_at_finish : float = 0
        self.did_accept : bool = False
        self.top_bids_percentage: float = 0.01
        self.force_accept_at_remaining_turns: float = 1
        self.force_accept_at_remaining_turns_light: float = 1
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

            if hasattr(self, "data_dict") and isinstance(self.data_dict, dict):
                self.learn_from_past_sessions()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # Update session data
            self.learn_from_past_sessions()

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
            self.save_data()
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
            # Update Q-values
            reward = (self.opponent_model.get_predicted_utility(bid) + self.get_utility(bid)) / 2.0
            self.opponent_model.update(bid, reward)
            self.last_received_bid = bid
            self.received_bids.append(bid)

    def my_turn(self):
        if self.last_received_bid is None:
            action = Offer(self.me, self.find_bid())
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

        # Check acceptance
        if self.accept_condition(self.last_received_bid):
            action = Accept(self.me, self.last_received_bid)
        else:
            if progress < 0.3:
                # Remain relatively conservative if still in the exploration phase
                accept_probability = 0.05 if last_offer_util > new_bid_util else 0.0
            elif progress < 0.8:
                nash_improvement = (nash_received - nash_counter) / nash_counter if nash_counter > 0 else 0
                base_probability = 0.2 + progress * 0.3 # Increases proportional to time
                accept_probability = base_probability * (1 + nash_improvement)

                # Add Tit-for-Tat behaviour - Concede if opponent concedes
                if hasattr(self, 'previous_opp_bid') and self.previous_opp_bid is not None:
                    previous_opp_util = float(self.score_bid(self.previous_opp_bid))
                    if last_offer_util > previous_opp_util:
                        accept_probability *= 1.2 # Concede

            else:
                time_pressure = (progress - 0.8) / 0.2 # 0 to 1 in final phase
                accept_probability = 0.5 + time_pressure * 0.3

                if last_offer_util >= new_bid_util:
                    accept_probability += 0.2

            if random.random() < accept_probability:
                action = Accept(self.me, self.last_received_bid)
            else:
                action = Offer(self.me, counter_bid)

        if hasattr(self, 'last_received_bid') and self.last_received_bid is not None:
            self.previous_opp_bid = self.last_received_bid

        self.send_action(action)

    def save_data(self):
        """ Stores session data to learn from experience"""

        session_data: SessionData = {
            "progressAtFinish": self.progress.get(time.time() * 1000),
            "utilityAtFinish": self.utility_at_finish,
            "didAccept": self.did_accept,
            "isGood": self.utility_at_finish >= 0.7,  # Example threshold for 'good' outcomes
            "topBidsPercentage": self.top_bids_percentage,
            "forceAcceptAtRemainingTurns": self.force_accept_at_remaining_turns,
        }

        if not self.data_dict:
            self.data_dict = {"sessions": []}

        # Append session data to memory structure
        self.data_dict["sessions"].append(session_data)


    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time.time() * 1000)

        bid_utility = self.score_bid(bid)

        reservation_value = 0.3 # TODO: Consider other values

        next_bid = self.find_bid()
        next_bid_utility = self.score_bid(next_bid)

        if progress < 0.3:
            # Exploration phase
            return bid_utility > next_bid_utility + 0.1 and bid_utility > 0.7

        elif progress < 0.8:
            alpha = 1.1
            beta = 0.02
            if bid_utility >= alpha * next_bid_utility + beta:
                return True

            # Estimate Nash Products
            estimated_opponent_utility = self.get_opponent_utility(bid, self.last_time)
            nash_product = bid_utility * estimated_opponent_utility

            next_bid_nash = next_bid_utility * self.get_opponent_utility(next_bid, self.last_time)

            if nash_product > next_bid_nash * 1.05:
                return True

            # Else reject
            return False

        else:
            # End phase, should be more willing to accept
            if bid_utility >= reservation_value and bid_utility >= next_bid_utility:
                return True

            if progress > 0.95 and bid_utility >= reservation_value:
                return True

            return False

    def find_bid(self) -> Bid:
        num_of_bids = self.all_bids.size()

        if self.bids_with_utilities is None:
            self.bids_with_utilities = []

            for index in range(num_of_bids):
                bid = self.all_bids.get(index)
                bid_utility = float(self.score_bid(bid))
                self.bids_with_utilities.append((bid, bid_utility))

            self.bids_with_utilities.sort(key=lambda tup: tup[1], reverse=True)

            self.num_of_top_bids = max(5, num_of_bids * self.top_bids_percentage)

        if (self.last_received_bid is None):
            return self.bids_with_utilities[0][0]

        progress = self.progress.get(time.time() * 1000)
        light_threshold = 0.95

        if (progress > light_threshold):
            return self.opponent_best_bid

        if (num_of_bids < self.num_of_top_bids):
            self.num_of_top_bids = num_of_bids / 2

        self.min_util = self.bids_with_utilities[math.floor(self.num_of_top_bids) - 1][1]

        picked_ranking = randint(0, math.floor(self.num_of_top_bids) - 1)

        return self.bids_with_utilities[picked_ranking][0]

    # def find_bid(self) -> Bid:
    #     num_of_bids = self.all_bids.size()
    #
    #     if self.bids_with_utilities is None:
    #         self.bids_with_utilities = []
    #
    #         for index in range(num_of_bids):
    #             bid = self.all_bids.get(index)
    #             own_utility = float(self.score_bid(bid))
    #             opponent_utility = self.opponent_model.get_predicted_utility(bid)
    #             self.bids_with_utilities.append((bid, own_utility, opponent_utility))
    #
    #         # Perform non-dominated sorting (Pareto optimal)
    #         self.pareto_front = self.get_pareto_front(self.bids_with_utilities)
    #
    #     if self.last_received_bid is None:
    #         # Initially return the best bid from the Pareto front
    #         return self.pareto_front[0][0]
    #
    #     progress = self.progress.get(time.time() * 1000)
    #     light_threshold = 0.95
    #
    #     if progress > light_threshold:
    #         return self.opponent_best_bid
    #
    #     # Randomly choose a bid from the Pareto front to promote diversity
    #     picked_bid = random.choice(self.pareto_front)[0]
    #
    #     return picked_bid

    def get_pareto_front(self, bids_with_utilities):
        """Perform non-dominated sorting to find the Pareto front."""
        pareto_front = []

        for candidate in bids_with_utilities:
            dominated = False
            for other in bids_with_utilities:
                if other == candidate:
                    continue
                # Check if candidate is dominated by other
                if other[1] >= candidate[1] and other[2] >= candidate[2] and (
                        other[1] > candidate[1] or other[2] > candidate[2]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(candidate)

        # Sort Pareto front by own utility, descending
        pareto_front.sort(key=lambda x: x[1], reverse=True)

        return pareto_front

    def score_bid(self, bid: Bid) -> float:

        u_self = float(self.get_utility(bid))
        u_opp = float(self.get_opponent_utility(bid, self.last_time))

        nash_product = u_self * u_opp
        social_welfare =  (u_self + u_opp) / 2.0

        gamma = 0.6

        score = gamma * nash_product + (1 - gamma) * social_welfare

        return score

    def get_utility(self, bid : Bid) -> float:
        pure_util = float(self.profile.getUtility(bid))

        return float(pure_util)

    def get_opponent_utility(self, bid : Bid, time : float) -> float:
        if self.opponent_model is not None:
            return self.opponent_model.get_predicted_utility(bid)
        else:
            return 0.0

    def learn_from_past_sessions(self):
        """Adapts strategy parameters based on historical performance data stored in self.data_dict."""
        if not self.data_dict or not self.data_dict.get("sessions"):
            self.logger.log(logging.INFO, "No past session data available for learning.")
            return

        sessions = self.data_dict["sessions"]

        # Analyze session outcomes
        failed_sessions = [s for s in sessions if not s["didAccept"]]
        successful_sessions = [s for s in sessions if s["didAccept"]]
        low_utility_sessions = [s for s in successful_sessions if s["utilityAtFinish"] < 0.5]

        # Calculate metrics
        failure_rate = len(failed_sessions) / len(sessions)
        avg_utility = np.mean([s["utilityAtFinish"] for s in successful_sessions]) if successful_sessions else 0
        low_utility_rate = len(low_utility_sessions) / len(sessions)

        self.force_accept_at_remaining_turns = max(0.8, min(1.2, 1.0 + failure_rate * 0.3))
        self.force_accept_at_remaining_turns_light = max(0.9, min(1.1, 1.0 + low_utility_rate * 0.2))

        self.top_bids_percentage = max(0.005, min(0.05, 0.01 + low_utility_rate * 0.02))

        self.min_util = max(0.3, min(0.5, 0.4 - (avg_utility - 0.5) * 0.1))

