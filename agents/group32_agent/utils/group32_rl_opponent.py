import time
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain


class OpponentModel:
    def __init__(self, domain: Domain, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 num_time_buckets: int = 10):
        """
        Initialize the opponent model.

        Args:
            domain (Domain): The negotiation domain.
            learning_rate (float): Learning rate for Q-learning.
            discount_factor (float): Discount factor for future rewards.
            num_time_buckets (int): Number of discretized time buckets.
        """
        self.domain = domain
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_time_buckets = num_time_buckets

        # Initialize Q-values for each issue-value pair for each time bucket.
        # The state is defined as (issue, time_bucket) and the action is the chosen value.
        self.q_table = {}
        for issue in domain.getIssues():
            values = domain.getIssuesValues()[issue].getValues()
            for bucket in range(num_time_buckets):
                self.q_table[(issue, bucket)] = {value: 0.0 for value in values}

        self.previous_bid = None

        # It is assumed that self.progress will be assigned externally (e.g., from the negotiation framework)
        self.progress = None

    def _get_time_bucket(self, progress: float) -> int:
        """
        Convert a negotiation progress value (assumed in [0,1]) to a discrete time bucket.
        """
        bucket = int(progress * self.num_time_buckets)
        if bucket >= self.num_time_buckets:
            bucket = self.num_time_buckets - 1
        return bucket

    def update(self, bid: Bid, reward: float):
        """
        Update Q-values based on the received bid and associated reward.
        The state is (issue, time_bucket) and action is the offered value.

        Args:
            bid (Bid): The bid offered by the opponent.
            reward (float): The reward signal (e.g., set to 1 for reinforcing the observed choice).
        """
        # Get current negotiation progress.
        # If self.progress is set, use it; otherwise, assume a mid-negotiation progress.
        try:
            progress = self.progress.get(time.time() * 1000)
        except AttributeError:
            progress = 0.5  # default fallback

        bucket = self._get_time_bucket(progress)

        for issue in self.domain.getIssues():
            value = bid.getValue(issue)
            state = (issue, bucket)
            old_q = self.q_table[state][value]
            max_future_q = max(self.q_table[state].values())

            # Standard Q-learning update
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
            self.q_table[state][value] = new_q

        self.previous_bid = bid

    def get_predicted_utility(self, bid: Bid) -> float:
        """
        Predict the opponent's utility for a given bid.
        For each issue, obtain the Q-value (which reflects the opponent's preference)
        for the offered value at the current time bucket, and average these values.

        Returns:
            float: The estimated utility in [0,1] (if Q-values are normalized appropriately).
        """
        try:
            progress = self.progress.get(time.time() * 1000)
        except AttributeError:
            progress = 0.5

        bucket = self._get_time_bucket(progress)
        total = 0.0
        count = 0
        for issue in self.domain.getIssues():
            value = bid.getValue(issue)
            state = (issue, bucket)
            if value not in self.q_table[state]:
                continue
            total += self.q_table[state][value]
            count += 1

        return total / count if count > 0 else 0.0

    def choose_best_bid(self, possible_bids: list[Bid]) -> Bid:
        """
        Select the bid with the highest predicted opponent utility from a list of possible bids.
        """
        best_bid = max(possible_bids, key=self.get_predicted_utility)
        return best_bid
