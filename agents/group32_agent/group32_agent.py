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

        self.data_dict : DataDict = None


        self.last_received_bid: Bid = None
        self.received_bids: List[Bid] = []
        self.opponent_model: OpponentModel = None

        # Keep track of all bids from opponent (and yours?)
        self.all_bids : AllBidsList = None
        self.bids_with_utilities : list[tuple[Bid, float]] = None
        self.num_of_top_bids : int = 1
        self.min_util : float = 0.8 # TODO: Adjust value

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
        self.sum_of_alphas = 0

        self.r1 = 0.5
        self.r2 = 0.4
        self.r3 = 0.8


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

        # 1) Compute your best counter-offer, QO(t).
        counter_bid = self.find_bid()

        # 2)  Check if the opponent's last offer is significantly better than your counter bid.
        last_offer_util = float(self.score_bid(self.last_received_bid))
        new_bid_util = float(self.score_bid(counter_bid))
        epsilon = 0.05  # margin for immediate acceptance
        #if last_offer_util >= new_bid_util + epsilon:
        if last_offer_util >= 0.80:
            action = Accept(self.me, self.last_received_bid)
            self.send_action(action)
            return

        # 3) Otherwise, compare the opponent's believed utility for each bid.
        opp_util_old = float(self.get_opponent_utility(self.last_received_bid, self.last_time))
        opp_util_new = float(self.get_opponent_utility(counter_bid, self.last_time))

        # 4) Adjust threshold for small utility difference.
        if abs(opp_util_new - opp_util_old) <= 0.015:  # tighter threshold for offering
            action = Offer(self.me, counter_bid)
        else:
            # 5) Lower the probability of acceptance to generate more offers.
            prob_accept = self.get_rank(self.last_received_bid) * 0.15 # reduce acceptance probability
            if random.random() < prob_accept:
                action = Accept(self.me, self.last_received_bid)
            else:
                action = Offer(self.me, counter_bid)

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
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.score_bid(bid) > 0.8, # TODO: Reservation condition???
            progress > 0.95,
        ]
        return all(conditions)

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
        alpha = float(self.get_utility(bid, self.last_time))

        # Current belief BT(T)
        opp_utility = self.get_opponent_utility(bid, self.last_time)

        lu_self = alpha
        lu_opp = opp_utility

        beta = (lu_opp + lu_self) * opp_utility

        return float(min(alpha, beta))

    def get_utility(self, bid : Bid, time : float) -> float:
        pure_util = float(self.profile.getUtility(bid)) # Based on values from domain

        decision_util = self.get_decision_utility(bid, time)

        experienced_util = self.get_experienced_utility(bid, time)  # Based on bid-history

        # r1: weight for decision utility
        # r2: weight for experienced utility
        # r3: overall weight for relationship measurement
        overall_util = pure_util + self.r3 * (self.r1 * decision_util + self.r2 * experienced_util)  # TODO: Change
        # Cast to float
        return float(overall_util)

    def get_opponent_utility(self, bid : Bid, time : float) -> float:
        if self.opponent_model is not None:
            return self.opponent_model.get_predicted_utility(bid)
        else:
            return 0.0

    def get_decision_utility(self, bid : Bid, time : float) -> float:
        """
        Heuristic:  If the current bid shows a concession compared to our last bid,
            then decision utility increases because we are 'sacrificing' some profit to preserve the relationship.
            Otherwise, if our bid is more aggressive (i.e. higher profit than before), we lower decision utility.
        return: utility value
        """

        # If concession present, increase decision utility since we are trying to preserve the relationship
        if self.last_received_bid is not None:
            prev_util = float(self.profile.getUtility(self.last_received_bid)) # Get pure utility
            current_util = float(self.profile.getUtility(bid))

            delta = prev_util - current_util # Normalized delta

            # If we are conceding (delta positive), decision utility increases
            decision_util = max(0, delta)
        else:
            decision_util = 0.5 # Neutral value

        return decision_util


    def get_experienced_utility(self, bid : Bid, time : float) -> float:
        """
        Heuristic: Similar bids have smaller distance and hence higher experience utility.
        """
        if self.last_received_bid is not None:
            distance = self.bid_distance(bid, self.last_received_bid)
            experienced_util = 1 - distance
        else:
            experienced_util = 1.0
        return experienced_util

    def bid_distance(self, bid1: Bid, bid2: Bid) -> float:
        """
        Computes a simple normalized distance between two bids.
        """
        issues = list(bid1.getIssueValues().keys())
        total = len(issues)
        if total == 0:
            return 0.0
        diff_count = sum(1 for issue in issues if bid1.getIssueValues()[issue] != bid2.getIssueValues()[issue])
        return diff_count / total

# TODO: Add time dependence/pressure
