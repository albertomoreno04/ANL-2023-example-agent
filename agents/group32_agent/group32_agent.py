import logging
import math
import random
from decimal import Decimal
from random import randint
import time
from typing import cast, TypedDict, List

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

from .utils.group32_opponent import OpponentProfile
from .utils.group32_bayesian_opponent import OpponentModel

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
                # possible_types = list(range(1, len(self.domain.getIssues()) + 1))
                # self.opponent_model = OpponentProfile(self.domain, possible_types)
                self.opponent_model = OpponentModel(self.domain)

            # Bayesian update based on each bid
            self.opponent_model.update(bid)
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
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)


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
            # TODO: Parametrize as epsilon
            return bid_utility > next_bid_utility + 0.1 and bid_utility > 0.7

        elif progress < 0.8:
            alpha = 1.0
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


    def get_rank(self, bid: Bid) -> float:
        """Calculate the rank number of a given offer.

            The rank is determined by ordering all received bids in descending order of utility,
            then assigning a rank based on the position of the bid in this list, normalized to [0,1].

            Args:
                bid (Bid): The bid whose rank number needs to be calculated.

            Returns:
                float: The rank of the bid, a value between 0 and 1.
            """
        if not self.received_bids:
            return 0.0  # If no bids are received, default to rank 0.

        sorted_bids = sorted(self.received_bids, key=lambda b: self.score_bid(b), reverse=True)

        # Find the index of the bid
        try:
            index = sorted_bids.index(bid)
        except ValueError:
            return 0.0

        rank = (index + 1) / len(sorted_bids)

        return rank

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

    def score_bid(self, bid: Bid) -> float:
        # alpha: your own utility for the bid at the current time.
        alpha = float(self.get_utility(bid))

        # Current belief BT(T)
        opp_utility = self.get_opponent_utility(bid, self.last_time)

        lu_self = alpha # Lu(alpha) ~ alpha
        lu_opp = opp_utility

        beta = (lu_opp + lu_self) * opp_utility

        return float(min(alpha, beta))

    def get_utility(self, bid : Bid) -> float:
        pure_util = float(self.profile.getUtility(bid))

        return float(pure_util)

    def get_opponent_utility(self, bid : Bid, time : float) -> float:
        if self.opponent_model is not None:
            return self.opponent_model.get_predicted_utility(bid)
        else:
            return 0.0


