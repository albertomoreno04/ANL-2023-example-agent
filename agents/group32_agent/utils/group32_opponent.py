from collections import defaultdict

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace

from decimal import Decimal
from typing import Dict, List
from geniusweb.issuevalue.Domain import Domain
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace, ValueSetUtilities


class OpponentModelEstimator:
    """
    This opponent model uses estimators to calculate the predicted utility
    of a bid. It is built from IssueEstimators (one per issue) that use
    ValueEstimators for each value.
    """

    def __init__(self, domain: Domain):
        self.offers: List[Bid] = []
        self.domain = domain
        # Create an IssueEstimator for each issue using the domain's available values.
        # We assume domain.getIssuesValues() returns a dict mapping issue IDs to DiscreteValueSet.
        self.issue_estimators: Dict[str, IssueEstimator] = {
            issue: IssueEstimator(value_set) for issue, value_set in domain.getIssuesValues().items()
        }


    def update(self, bid: Bid):
        """Updates the model with a new bid."""
        self.offers.append(bid)
        for issue, estimator in self.issue_estimators.items():
            estimator.update(bid.getValue(issue))

    def get_predicted_utility(self, bid: Bid) -> float:
        """Calculates the predicted utility of a bid using the estimators."""
        if not self.offers or bid is None:
            return 0.0

        total_issue_weight = 0.0
        value_utilities = []
        issue_weights = []

        # For each issue, obtain the utility for the bid's value and the estimator's weight.
        for issue, estimator in self.issue_estimators.items():
            value: Value = bid.getValue(issue)
            value_util = estimator.get_value_utility(value)
            value_utilities.append(value_util)
            issue_weights.append(estimator.weight)
            total_issue_weight += estimator.weight

        # Normalize weights to sum to 1 (if total is zero, assume equal weight).
        if total_issue_weight == 0:
            norm_weights = [1.0 / len(issue_weights)] * len(issue_weights)
        else:
            norm_weights = [w / total_issue_weight for w in issue_weights]

        # Calculate the overall predicted utility as the weighted sum.
        predicted_utility = sum(w * vu for w, vu in zip(norm_weights, value_utilities))
        return predicted_utility

class OpponentProfile:
    """
    Maintains a set of candidate opponent models (one per type) and updates
    a probability distribution over types using Bayesian updating.
    """

    def __init__(self, domain: Domain, types: List[int]):
        """
        :param domain: The negotiation domain.
        :param types: A list of candidate opponent type identifiers (e.g., [1, 2, 3]).
        """
        self.domain = domain
        self.types = types
        # For each type, create an OpponentModelEstimator.
        self.models: Dict[int, OpponentModelEstimator] = {
            t: OpponentModelEstimator(domain) for t in types
        }
        # Initialize uniform prior probabilities.
        self.type_probabilities: Dict[int, float] = {t: 1.0 / len(types) for t in types}
        self.believed_type: int = None
        self.believed_model: OpponentModelEstimator = None

    def update(self, bid: Bid):
        """
        Update each candidate model with the new bid and perform Bayesian updating
        of the probabilities. The likelihood of each type is taken to be its model's
        predicted utility for the observed bid.
        """
        # Update all candidate models.
        for t in self.types:
            self.models[t].update(bid)

        total_prob = 0.0
        new_probs = {}
        # For each type, compute the likelihood of the bid under that candidate's model.
        for t in self.types:
            likelihood = self.models[t].get_predicted_utility(bid)
            new_probs[t] = self.type_probabilities[t] * likelihood
            total_prob += new_probs[t]

        # Normalize probabilities.
        if total_prob > 0:
            for t in self.types:
                new_probs[t] /= total_prob
        else:
            new_probs = {t: 1.0 / len(self.types) for t in self.types}

        self.type_probabilities = new_probs
        # Choose the candidate type with the highest probability.
        self.believed_type = max(self.types, key=lambda t: self.type_probabilities[t])
        self.believed_model = self.models[self.believed_type]

    def getUtility(self, bid: Bid) -> float:
        """
        Returns the predicted utility of the bid according to the currently
        believed opponent model.
        """
        if self.believed_model is not None:
            return self.believed_model.get_predicted_utility(bid)
        else:
            return 0.0

class ValueEstimator:
    def __init__(self):
        self.count = 0
        self.utility = 0

    def update(self):
        self.count += 1

    def recalculate_utility(self, max_value_count: int, weight: float):
        if weight < 1:
            mod_value_count = ((self.count + 1) ** (1 - weight)) - 1
            mod_max_value_count = ((max_value_count + 1) ** (1 - weight)) - 1

            self.utility = mod_value_count / mod_max_value_count
        else:
            self.utility = 1

class IssueEstimator:
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError(
                "This issue estimator only supports issues with discrete values"
            )

        self.bids_received = 0
        self.max_value_count = 0
        self.num_values = value_set.size()
        self.value_trackers = defaultdict(ValueEstimator)
        self.weight = 0

    def update(self, value: Value):
        self.bids_received += 1

        # get the value tracker of the value that is offered
        value_tracker = self.value_trackers[value]

        # register that this value was offered
        value_tracker.update()

        # update the count of the most common offered value
        self.max_value_count = max([value_tracker.count, self.max_value_count])

        equal_shares = self.bids_received / self.num_values
        self.weight = (self.max_value_count - equal_shares) / (
            self.bids_received - equal_shares
        )

        # recalculate all value utilities
        for value_tracker in self.value_trackers.values():
            value_tracker.recalculate_utility(self.max_value_count, self.weight)

    def get_value_utility(self, value: Value):
        if value in self.value_trackers:
            return self.value_trackers[value].utility
        # Else return 0
        return 0
