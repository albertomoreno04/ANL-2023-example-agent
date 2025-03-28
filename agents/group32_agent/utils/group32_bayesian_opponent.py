from collections import defaultdict
import numpy as np

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value


class OpponentModel:
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain
        self.time_step = 0
        self.sigma = 0.1  # Standard deviation for conditional probability

        self.issue_estimators = {
            i: IssueEstimator(v) for i, v in domain.getIssuesValues().items()
        }

    def update(self, bid: Bid):
        # Keep track of all bids received
        self.offers.append(bid)
        self.time_step += 1

        # Predict opponent's utility based on concession model
        predicted_utility = self.predict_utility(self.time_step)

        # Calculate partial utilities for each issue
        partial_utilities = {}
        for issue_id in self.issue_estimators:
            partial_utilities[issue_id] = self._calculate_partial_utility(bid, issue_id)

        # Update all issue estimators with the value that is offered for that issue
        for issue_id, issue_estimator in self.issue_estimators.items():
            value = bid.getValue(issue_id)
            # Update the estimator
            issue_estimator.update(value)

            # Also perform Bayesian update
            issue_estimator.bayesian_update(
                value,
                predicted_utility,
                partial_utilities[issue_id],
                self.sigma
            )

        # Normalize weights
        self._normalize_weights()

    def predict_utility(self, time_step: int) -> float:
        """Predict utility based on concession model from paper"""
        return max(0.0, 1.0 - 0.05 * time_step)

    def _calculate_partial_utility(self, bid: Bid, exclude_issue: str) -> float:
        """Calculate utility excluding the specified issue"""
        partial_utility = 0.0

        for issue_id, issue_estimator in self.issue_estimators.items():
            if issue_id == exclude_issue:
                continue

            value = bid.getValue(issue_id)
            value_utility = issue_estimator.get_value_utility(value)
            partial_utility += issue_estimator.weight * value_utility

        return partial_utility

    def _normalize_weights(self):
        """Normalize issue weights to sum to 1.0"""
        total_issue_weight = sum(estimator.weight for estimator in self.issue_estimators.values())

        if total_issue_weight > 0:
            for issue_estimator in self.issue_estimators.values():
                issue_estimator.weight = issue_estimator.weight / total_issue_weight

    def get_predicted_utility(self, bid: Bid):
        if len(self.offers) == 0 or bid is None:
            return 0

        # initiate
        total_issue_weight = 0.0
        value_utilities = []
        issue_weights = []

        for issue_id, issue_estimator in self.issue_estimators.items():
            # get the value that is set for this issue in the bid
            value: Value = bid.getValue(issue_id)

            # collect both the predicted weight for the issue and
            # predicted utility of the value within this issue
            value_utilities.append(issue_estimator.get_value_utility(value))
            issue_weights.append(issue_estimator.weight)

            total_issue_weight += issue_estimator.weight

        # normalise the issue weights such that the sum is 1.0
        if total_issue_weight == 0.0:
            issue_weights = [1 / len(issue_weights) for _ in issue_weights]
        else:
            issue_weights = [iw / total_issue_weight for iw in issue_weights]

        # calculate predicted utility by multiplying all value utilities with their issue weight
        predicted_utility = sum(
            [iw * vu for iw, vu in zip(issue_weights, value_utilities)]
        )

        return predicted_utility


class IssueEstimator:
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError(
                "This issue estimator only supports issues with discrete values"
            )

        self.bids_received = 0
        self.max_value_count = 0
        self.num_values = value_set.size()

        # Create value estimators for each possible value
        self.values = list(value_set.getValues())
        self.value_trackers = defaultdict(ValueEstimator)

        # Initialize value trackers with normalized positions
        for i, value in enumerate(self.values):
            normalized_pos = i / (len(self.values) - 1) if len(self.values) > 1 else 0.5
            self.value_trackers[value] = ValueEstimator()
            self.value_trackers[value].normalized_position = normalized_pos

        # Weight hypotheses (discrete values between 0-1)
        self.n_weight_hypotheses = 10
        self.weight_hypotheses = np.linspace(0, 1, self.n_weight_hypotheses)

        # Probability distribution over weight hypotheses (initially uniform)
        self.weight_probs = np.ones(self.n_weight_hypotheses) / self.n_weight_hypotheses

        # Current expected weight
        self.weight = 0.5

    def update(self, value: Value):
        self.bids_received += 1

        # get the value tracker of the value that is offered
        value_tracker = self.value_trackers[value]

        # register that this value was offered
        value_tracker.update()

        # update the count of the most common offered value
        self.max_value_count = max(value_tracker.count, self.max_value_count)

        # update predicted issue weight using original frequency method
        equal_shares = self.bids_received / self.num_values
        frequency_weight = (self.max_value_count - equal_shares) / (
                self.bids_received - equal_shares
        ) if self.bids_received > equal_shares else 0

        # Blend frequency-based and Bayesian weights (initially rely more on frequency)
        if self.bids_received <= 3:
            self.weight = frequency_weight
        else:
            # Gradually transition to Bayesian weight
            self.weight = self._get_bayesian_weight()

        # recalculate all value utilities
        for value_tracker in self.value_trackers.values():
            value_tracker.recalculate_utility(self.max_value_count, self.weight)

    def _get_bayesian_weight(self) -> float:
        """Calculate expected weight based on current probability distribution"""
        return np.sum(self.weight_probs * self.weight_hypotheses)

    def bayesian_update(self, value: Value, predicted_utility: float,
                        partial_utility: float, sigma: float):
        """Bayesian update for issue weights and value evaluation functions"""
        # Skip weight update for first bid (not enough information)
        if self.bids_received > 1:
            # Update weight hypotheses using Bayesian learning
            new_weight_probs = np.zeros_like(self.weight_probs)

            for i, w in enumerate(self.weight_hypotheses):
                # Get current evaluation for the value
                value_tracker = self.value_trackers[value]
                eval_value = value_tracker.get_bayesian_utility()

                # Calculate total utility with this weight
                utility = partial_utility + w * eval_value

                # Calculate likelihood using Gaussian distribution (from paper)
                likelihood = np.exp(-((utility - predicted_utility) ** 2) /
                                    (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

                # Calculate posterior probability
                new_weight_probs[i] = self.weight_probs[i] * likelihood

            # Normalize probabilities
            total_prob = new_weight_probs.sum()
            if total_prob > 0:
                self.weight_probs = new_weight_probs / total_prob

        # Update value evaluation function using Bayesian learning
        value_tracker = self.value_trackers[value]
        value_tracker.bayesian_update(partial_utility, predicted_utility,
                                      self.weight, sigma)

    def get_value_utility(self, value: Value):
        if value in self.value_trackers:
            return self.value_trackers[value].utility
        return 0


class ValueEstimator:
    def __init__(self):
        self.count = 0
        self.utility = 0

        # Normalized position of this value within its issue range (0-1)
        self.normalized_position = 0.5

        # Hypotheses: 1=downhill, 2=uphill, 3=triangular (as in paper)
        self.eval_functions = [1, 2, 3]

        # Initial uniform probability distribution over hypotheses
        self.eval_probs = {func_type: 1.0 / 3.0 for func_type in self.eval_functions}

    def update(self):
        self.count += 1

    def _apply_eval_function(self, func_type: int) -> float:
        """Apply evaluation function to normalized position"""
        if func_type == 1:  # Downhill
            return 1 - self.normalized_position
        elif func_type == 2:  # Uphill
            return self.normalized_position
        elif func_type == 3:  # Triangular
            return 1 - 2 * abs(self.normalized_position - 0.5)
        else:
            raise ValueError(f"Unknown evaluation function type: {func_type}")

    def get_bayesian_utility(self) -> float:
        """Get utility based on Bayesian evaluation functions"""
        utility = 0.0
        for func_type in self.eval_functions:
            eval_value = self._apply_eval_function(func_type)
            utility += self.eval_probs[func_type] * eval_value
        return utility

    def bayesian_update(self, partial_utility: float, predicted_utility: float,
                        weight: float, sigma: float):
        """Update evaluation function hypotheses using Bayesian learning"""
        new_eval_probs = {}

        for func_type in self.eval_functions:
            # Calculate evaluation according to this function type
            eval_value = self._apply_eval_function(func_type)

            # Calculate total utility with this evaluation
            utility = partial_utility + weight * eval_value

            # Calculate likelihood using Gaussian distribution
            likelihood = np.exp(-((utility - predicted_utility) ** 2) /
                                (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

            # Calculate posterior probability (Bayes' rule)
            new_eval_probs[func_type] = self.eval_probs[func_type] * likelihood

        # Normalize probabilities
        total_prob = sum(new_eval_probs.values())
        if total_prob > 0:
            for func_type in self.eval_functions:
                self.eval_probs[func_type] = new_eval_probs[func_type] / total_prob

        # Update utility using Bayesian approach
        self.utility = self.get_bayesian_utility()

    def recalculate_utility(self, max_value_count: int, weight: float):
        """
        Original method for compatibility with the original structure,
        but enhanced with Bayesian learning
        """
        # For initial bids, blend frequency-based and Bayesian approaches
        if self.count <= 3 and weight < 1:
            # Original frequency-based formula
            mod_value_count = ((self.count + 1) ** (1 - weight)) - 1
            mod_max_value_count = ((max_value_count + 1) ** (1 - weight)) - 1

            freq_utility = mod_value_count / mod_max_value_count if mod_max_value_count > 0 else 0

            # Blend with Bayesian utility
            bayes_utility = self.get_bayesian_utility()
            self.utility = 0.5 * freq_utility + 0.5 * bayes_utility
        else:
            # After sufficient bids, rely on Bayesian approach
            self.utility = self.get_bayesian_utility()


# class ScalableBayesianOpponentModel:
#     """
#     Scalable version of the Bayesian opponent model for multi-issue negotiation.
#     Uses additional independence assumptions to make learning more efficient.
#     """
#
#     def __init__(self, n_issues: int, issue_ranges: List[Tuple[float, float]],
#                  domain_knowledge: Optional[Dict] = None, sigma: float = 0.1,
#                  n_weight_hypotheses: int = 10):
#         """Initialize the scalable model."""
#         self.n_issues = n_issues
#         self.issue_ranges = issue_ranges
#         self.sigma = sigma
#         self.n_weight_hypotheses = n_weight_hypotheses
#         self.eval_functions = [1, 2, 3]  # downhill, uphill, triangular
#
#         # Initialize weights for each issue independently
#         self.weight_hypotheses = {(issue, j): j / (n_weight_hypotheses - 1)
#                                   for issue in range(n_issues)
#                                   for j in range(n_weight_hypotheses)}
#
#         # Initialize uniform probability distributions
#         if not domain_knowledge:
#             self.weight_probs = {(i, j): 1.0 / n_weight_hypotheses
#                                  for i in range(n_issues)
#                                  for j in range(n_weight_hypotheses)}
#
#             self.eval_probs = {(i, j): 1.0 / len(self.eval_functions)
#                                for i in range(n_issues)
#                                for j in range(len(self.eval_functions))}
#
#         self.time_step = 0
#         self.opponent_bids = []
#
#     def _update_weight_probs(self, issue: int, bid: List[float],
#                              predicted_utility: float) -> None:
#         """Update weight hypotheses for a single issue."""
#         # Calculate partial utility without this issue
#         partial_utility = self._calculate_partial_utility(bid, issue)
#
#         new_probs = {}
#         total_prob = 0.0
#
#         for j in range(self.n_weight_hypotheses):
#             # Get weight for this hypothesis
#             weight = self.weight_hypotheses.get((issue, j), 0.0)
#
#             # Calculate expected evaluation for this issue
#             expected_eval = self._get_expected_evaluation(issue, bid[issue])
#
#             # Calculate total utility with this weight
#             utility = partial_utility + weight * expected_eval
#
#             # Calculate likelihood using Gaussian distribution
#             likelihood = np.exp(-((utility - predicted_utility) ** 2) /
#                                 (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
#
#             # Calculate posterior probability
#             prior = self.weight_probs.get((issue, j), 0.0)
#             posterior = prior * likelihood
#
#             new_probs[(issue, j)] = posterior
#             total_prob += posterior
#
#         # Normalize probabilities
#         if total_prob > 0:
#             for key in new_probs:
#                 self.weight_probs[key] = new_probs[key] / total_prob
#
#     def update_model(self, bid: List[float]) -> None:
#         """Update the model with a new opponent bid."""
#         self.time_step += 1
#         self.opponent_bids.append(bid)
#
#         # Predict utility based on time step
#         predicted_utility = self.predict_utility(self.time_step)
#
#         # Skip updating weights for the first bid
#         if self.time_step > 1:
#             # Update each issue's weights independently
#             for issue in range(self.n_issues):
#                 self._update_weight_probs(issue, bid, predicted_utility)
#
#         # Update evaluation functions for each issue
#         for issue in range(self.n_issues):
#             self._update_eval_probs(issue, bid, predicted_utility)
#
#         # Normalize weights to sum to 1
#         self._normalize_weights()
#
#     def _calculate_partial_utility(self, bid: List[float], exclude_issue: int,
#                                    weights: List[float]) -> float:
#         """Calculate utility excluding a specific issue."""
#         partial_utility = 0.0
#
#         for issue in range(self.n_issues):
#             if issue == exclude_issue:
#                 continue
#
#             # Find most likely evaluation function for this issue
#             max_prob = 0.0
#             best_func = 1
#
#             for func_type in self.eval_functions:
#                 prob = self.eval_probs.get((issue, func_type), 0.0)
#                 if prob > max_prob:
#                     max_prob = prob
#                     best_func = func_type
#
#             # Calculate evaluation
#             issue_value = bid[issue]
#             eval_value = self._apply_eval_function(issue_value, best_func,
#                                                    self.issue_ranges[issue])
#
#             # Add to partial utility
#             partial_utility += weights[issue] * eval_value
#
#         return partial_utility
#
#
#     def _update_eval_probs(self, bid: List[float], predicted_utility: float) -> None:
#         """Update probabilities of evaluation function hypotheses using Bayes' rule."""
#         expected_weights = self._get_expected_weights()
#
#         # Update each issue independently
#         for issue in range(self.n_issues):
#             # Calculate partial utility without this issue
#             partial_utility = self._calculate_partial_utility(bid, issue, expected_weights)
#
#             new_probs = {}
#             total_prob = 0.0
#
#             for func_type in self.eval_functions:
#                 # Calculate utility with this evaluation function
#                 issue_value = bid[issue]
#                 eval_value = self._apply_eval_function(issue_value, func_type,
#                                                        self.issue_ranges[issue])
#                 utility = partial_utility + expected_weights[issue] * eval_value
#
#                 # Calculate likelihood
#                 likelihood = np.exp(-((utility - predicted_utility) ** 2) /
#                                     (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
#
#                 # Calculate posterior
#                 prior = self.eval_probs.get((issue, func_type), 0.0)
#                 posterior = prior * likelihood
#
#                 new_probs[(issue, func_type)] = posterior
#                 total_prob += posterior
#
#             # Normalize probabilities for this issue
#             if total_prob > 0:
#                 for key in new_probs:
#                     if key[0] == issue:
#                         self.eval_probs[key] = new_probs[key] / total_prob
#
#     def _evaluate_bid(self, bid: List[float], weights: List[float],
#                       eval_funcs: Dict) -> float:
#         """Evaluate the utility of a bid according to a hypothesis."""
#         utility = 0.0
#
#         for i in range(self.n_issues):
#             func_type = eval_funcs.get((i, 0), 1)
#             eval_value = self._apply_eval_function(bid[i], func_type, self.issue_ranges[i])
#             utility += weights[i] * eval_value
#
#         return utility
#
#     def _get_expected_weights(self) -> List[float]:
#         """Calculate expected weights based on current probability distribution."""
#         expected_weights = [0.0] * self.n_issues
#
#         for i, weights in enumerate(self.weight_hypotheses):
#             prob = self.weight_probs.get(i, 0.0)
#             for issue in range(self.n_issues):
#                 expected_weights[issue] += prob * weights[issue]
#
#         # Normalize weights to sum to 1
#         total = sum(expected_weights)
#         if total > 0:
#             expected_weights = [w / total for w in expected_weights]
#
#         return expected_weights
#
#
#     def get_expected_utility(self, bid: List[float]) -> float:
#         """Calculate the expected utility of a bid for the opponent."""
#         expected_utility = 0.0
#
#         # Get expected weights
#         expected_weights = self._get_expected_weights()
#
#         # For each issue, calculate expected evaluation
#         for issue in range(self.n_issues):
#             issue_value = bid[issue]
#
#             # Calculate expected evaluation for this issue
#             expected_eval = 0.0
#             for func_type in self.eval_functions:
#                 prob = self.eval_probs.get((issue, func_type), 0.0)
#                 eval_value = self._apply_eval_function(issue_value, func_type,
#                                                        self.issue_ranges[issue])
#                 expected_eval += prob * eval_value
#
#             # Add weighted contribution to expected utility
#             expected_utility += expected_weights[issue] * expected_eval
#
#         return expected_utility
#
#
#     def _apply_eval_function(self, value: float, func_type: int,
#                              issue_range: Tuple[float, float]) -> float:
#         """Apply an evaluation function to an issue value."""
#         min_val, max_val = issue_range
#
#         # Normalize value to [0, 1] range
#         if max_val - min_val == 0:
#             normalized = 0.5
#         else:
#             normalized = (value - min_val) / (max_val - min_val)
#
#         if func_type == 1:  # Downhill
#             return 1 - normalized
#         elif func_type == 2:  # Uphill
#             return normalized
#         elif func_type == 3:  # Triangular
#             return 1 - 2 * abs(normalized - 0.5)
#         else:
#             raise ValueError(f"Unknown evaluation function type: {func_type}")
#
#
#
#     def predict_utility(self, time_step: int) -> float:
#         """
#         Predict opponent's expected utility at a given time step based on
#         rationality assumption (linear concession model).
#         """
#         return max(0.0, 1.0 - 0.05 * time_step)
#
#     def update_model(self, bid: List[float]) -> None:
#         """Update the opponent model using Bayesian learning based on a new bid."""
#         self.time_step += 1
#         self.opponent_bids.append(bid)
#
#         # Predict opponent's utility for this time step
#         predicted_utility = self.predict_utility(self.time_step)
#
#         # Update weight hypotheses probabilities (skip first bid)
#         if self.time_step > 1:
#             self._update_weight_probs(bid, predicted_utility)
#
#         # Update evaluation function hypotheses probabilities
#         self._update_eval_probs(bid, predicted_utility)
#
#     def _update_weight_probs(self, bid: List[float], predicted_utility: float) -> None:
#         """Update probabilities of weight hypotheses using Bayes' rule."""
#         new_probs = {}
#         total_prob = 0.0
#
#         for i, weights in enumerate(self.weight_hypotheses):
#             # Construct evaluation functions based on current probabilities
#             eval_funcs = {}
#             for issue in range(self.n_issues):
#                 max_prob = 0.0
#                 best_func = 1
#
#                 for func_type in self.eval_functions:
#                     prob = self.eval_probs.get((issue, func_type), 0.0)
#                     if prob > max_prob:
#                         max_prob = prob
#                         best_func = func_type
#
#                 eval_funcs[(issue, 0)] = best_func
#
#             # Calculate utility for this hypothesis
#             utility = self._evaluate_bid(bid, weights, eval_funcs)
#
#             # Calculate likelihood using Gaussian distribution
#             likelihood = np.exp(-((utility - predicted_utility) ** 2) /
#                                 (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
#
#             # Calculate posterior probability
#             prior = self.weight_probs.get(i, 0.0)
#             posterior = prior * likelihood
#
#             new_probs[i] = posterior
#             total_prob += posterior
#
#         # Normalize probabilities
#         if total_prob > 0:
#             for i in new_probs:
#                 new_probs[i] /= total_prob
#
#         self.weight_probs = new_probs
#
#     def find_best_bid(self, own_utility_func, target_utility: float,
#                       delta: float = 0.02, n_samples: int = 1000) -> List[float]:
#         """
#         Find the bid that maximizes opponent utility among bids with a
#         certain utility for the agent itself.
#         """
#         best_bid = None
#         best_opp_utility = -1.0
#
#         # Sample random bids
#         for _ in range(n_samples):
#             # Generate random bid
#             bid = [min_val + np.random.random() * (max_val - min_val)
#                    for min_val, max_val in self.issue_ranges]
#
#             # Calculate own utility
#             own_utility = own_utility_func(bid)
#
#             # Check if it's within acceptable range of target utility
#             if abs(own_utility - target_utility) <= delta:
#                 # Calculate opponent utility
#                 opp_utility = self.get_expected_utility(bid)
#
#                 # Update best bid if this is better
#                 if opp_utility > best_opp_utility:
#                     best_opp_utility = opp_utility
#                     best_bid = bid
#
#         return best_bid
