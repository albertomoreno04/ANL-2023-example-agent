import time
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.progress.Progress import Progress


class OpponentModel:
    def __init__(self, domain: Domain, progress : Progress, learning_rate: float = 0.1, discount_factor: float = 0.9,
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
        self.progress = progress

        # The state is defined as (issue, time_bucket) and the action is the chosen value.
        self.q_table = {}
        for issue in domain.getIssues():
            values = domain.getIssuesValues()[issue].getValues()
            for bucket in range(num_time_buckets):
                self.q_table[(issue, bucket)] = {value: 0.0 for value in values}

        self.previous_bid = None

    def _get_time_bucket(self, progress: float) -> int:
        """
        Convert a negotiation progress value to a discrete time bucket.

        :return: time bucket index
        """
        bucket = int(progress * self.num_time_buckets)
        if bucket >= self.num_time_buckets:
            bucket = self.num_time_buckets - 1
        return bucket

    def update(self, bid: Bid, reward: float):
        """
        Update Q-values based on the received bid and associated reward (score function from agent)

        Args:
            bid (Bid): The bid offered by the opponent.
            reward (float): The reward signal (e.g., set to 1 for reinforcing the observed choice).
        """
        try:
            progress = self.progress.get(time.time() * 1000)
        except AttributeError:
            progress = 0.5

        bucket = self._get_time_bucket(progress)

        for issue in self.domain.getIssues():
            value = bid.getValue(issue)
            state = (issue, bucket)
            old_q = self.q_table[state][value]
            max_future_q = max(self.q_table[state].values())

            # Q-learning update with discount factor
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
            self.q_table[state][value] = new_q

        self.previous_bid = bid

    def get_predicted_utility(self, bid: Bid) -> float:
        """
        Predict the opponent's utility for a given bid.
        For each issue, obtain the Q-value for the offered value at the current time bucket, and average these values.

        :return: The estimated utility in [0,1].
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
        Selects the bid with the highest utility.

        :return: Selected bid
        """
        best_bid = max(possible_bids, key=self.get_predicted_utility)
        return best_bid
