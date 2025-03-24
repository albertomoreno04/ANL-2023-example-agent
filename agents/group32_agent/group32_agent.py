import logging
from decimal import Decimal
from random import randint
from time import time
from typing import cast, TypedDict

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

from .utils.group32_opponent import OpponentModel

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
        self.opponent_model: OpponentModel = None

        # Keep track of all bids from opponent (and yours?)
        self.all_bids : AllBidsList = None
        self.bids_with_utilities : list[tuple[Bid, float]] = None
        self.num_of_top_bids : int = 1
        self.min_util : float = 0.8 # TODO: Adjust value

        # Logging helpers
        self.round_times : list[Decimal] = []
        self.last_time = None
        self.avg_time = None
        self.utility_at_finish : float = 0
        self.did_accept : bool = False
        self.top_bids_percentage: float = 1 / 300
        self.force_accept_at_remaining_turns: float = 1
        self.force_accept_at_remaining_turns_light: float = 1
        self.opponent_best_bid: Bid = None
        self.logger.log(logging.INFO, "party is initialized")

        # Params
        self.r1 = 0.2 # Decision utility weight
        self.r2 = 0.3 # Experienced utility weight
        self.r3 = 0.5 # Overall relationship weight

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
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool: # TODO: Implement
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > 0.8,
            progress > 0.95,
        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        # Retrieve all possible bids from the negotiation domain.
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = -float('inf')
        best_bid = None

        # Try 500 random bids.
        for _ in range(500): # TODO: Probably do intelligent search
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score = bid_score
                best_bid = bid

        return best_bid

    def score_bid(self, bid: Bid) -> float:
        # alpha: your own utility for the bid at the current time.
        alpha = self.get_utility(bid, self.last_time)

        # opp_utility: the opponent's utility for the bid based on your current belief (BT(t)).
        opp_utility = self.get_opponent_utility(bid, self.last_time)

        # For the purpose of ranking offers, we assume that the Luce number is proportional
        # to the raw utility (since the denominator is constant across bids).
        lu_own = alpha
        lu_opp = opp_utility

        # beta: combine the Luce numbers (i.e. likelihood estimates) with the opponent's utility.
        beta = (lu_opp + lu_own) * opp_utility

        # The QO function selects the bid maximizing min{alpha, beta}.
        return min(alpha, beta)

    def get_utility(self, bid : Bid, time : float) -> Decimal:
        pure_util = self.profile.getUtility(bid) # TODO: Check this

        # Compute the decision utility component for relationship measurement.
        decision_util = self.get_decision_utility(bid, time)

        experienced_util = self.get_experienced_utility(bid, time) # Based on bid-history

        # r1: weight for decision utility
        # r2: weight for experienced utility
        # r3: overall weight for relationship measurement
        overall_util = pure_util + self.r3 * (self.r1 * decision_util + self.r2 * experienced_util)

        return overall_util

    def get_decision_utility(self, bid : Bid, time : float) -> float:
        """
        Heuristic:  If the current bid shows a concession compared to our last bid,
            then decision utility increases because we are 'sacrificing' some profit to preserve the relationship.
            Otherwise, if our bid is more aggressive (i.e. higher profit than before), we lower decision utility.
        return: utility value
        """

        # If concession present, increase decision utility since we are trying to preserve the relationship
        if self.last_received_bid is not None:
            prev_util = self.profile.getUtility(self.last_received_bid) # Get pure utility
            current_util = self.profile.getUtility(bid)

            delta = prev_util - current_util # Normalized delta

            # If we are conceding (delta positive), decision utility increases;
            # if not, we set a low decision utility.
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
            # More similarity (i.e., smaller distance) means higher experienced utility.
            experienced_util = 1 - distance
        else:
            experienced_util = 1.0  # Maximum credibility if there is no previous bid.
        return experienced_util

    def bid_distance(self, bid1: Bid, bid2: Bid) -> float:
        """
        Computes a simple normalized distance between two bids.
        """
        issues = list(bid1.keys())
        total = len(issues)
        if total == 0:
            return 0.0
        diff_count = sum(1 for issue in issues if bid1[issue] != bid2.get(issue))
        return diff_count / total

    def get_opponent_utility(self, bid : Bid, time : float) -> float:
        pass


# TODO: Get Opponent Utilities + Map them to classes