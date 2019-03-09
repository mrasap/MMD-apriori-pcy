from apriori.lib import *
from apriori.config import *
from apriori.apriori_algorithm import Apriori
import timeit


class PCY(Apriori):
    def __init__(self, bucket_size: int, *args, **kwargs):
        """
        Extend the apriori algorithm with the Park Chen Yu upgrade.

        The concept is that in the first run, a lot of memory is not utilized, while that is the
        bottle neck of the algorithm. Therefore, we (computationally) invest in a hashmap that
        can be used to speed up the process in the second run.

        :param bucket_size: int: the amount of buckets that we are hashing
        :param args: list of arguments of the super class
        :param kwargs: dict of arguments of the super class
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.bucket_size = bucket_size
        self.buckets = {}

    def construct_candidates_pcy(self, filtered: list) -> list:
        """
        Extends the construct_candidates function to also build a hashmap on the first run.
        """
        if self.k != 2:
            return self.construct_candidates(filtered)

        else:
            candidates = list()
            # n_total = 0
            # n_skipped = 0
            for i, s1 in enumerate(filtered):
                for s2 in filtered[i + 1:len(filtered)]:
                    candidate = set(s1).union(set(s2))
                    # n_total += 1
                    if len(candidate) == self.k:
                        if self.buckets[hash(frozenset(candidate)) % self.bucket_size] == 1:
                            if candidate not in candidates:
                                candidates.append(candidate)
                        # else:
                        #     n_skipped += 1

            # print("Sets skipped by PCY =", n_skipped, "/", n_total)
            return candidates

    def count_candidates_pcy(self, candidates: list) -> dict:
        """
        Extends the count_candidates function to also build a hashmap on the first run.
        """
        if self.k != 1:
            return self.count_candidates(candidates)

        self.buckets = {i: 0 for i in range(self.bucket_size)}
        candidates_count = {frozenset(c): 0 for c in candidates}

        for basket in self.baskets:
            subsets = get_subsets(basket, self.k)
            for i, s1 in enumerate(subsets):
                if s1 in candidates_count:
                    candidates_count[s1] += 1
                for s2 in subsets[i + 1:len(subsets)]:
                    hash_result = hash(frozenset(s1.union(s2)))
                    self.buckets[hash_result % self.bucket_size] += 1

        # this step isn't useful in python because the dict value is stored as 32 bits anyway
        self.buckets = {k: 1 if v >= self.support_threshold else 0 for k, v in self.buckets.items()}

        return candidates_count

    def get_frequent_sets(self):
        """
        Overwrites the get_frequent_sets function to incorporate the pcy algorithm.
        """
        filtered_candidates = None

        for self.k in range(1, (self.k_max + 1)):
            candidates = self.construct_candidates_pcy(filtered_candidates)
            counted_candidates = self.count_candidates_pcy(candidates)
            filtered_candidates = self.filter_candidates(counted_candidates)

        return filtered_candidates


if __name__ == "__main__":
    baskets = generate_documents()

    runner = PCY(baskets=baskets, k=K, support_threshold=SUPPORT_THRESHOLD,
        bucket_size=BUCKET_SIZE)

    start_time = timeit.default_timer()
    frequents = runner.get_frequent_sets()
    elapsed = timeit.default_timer() - start_time

    print(elapsed)
    print("\nRESULTS = \n", frequents)
