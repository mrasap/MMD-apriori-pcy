from apriori.lib import *
from apriori.config import *
import timeit

class Apriori:
    def __init__(self, baskets: list, k: int, support_threshold: int):
        """
        Creates a class that can generate the sets of items that are frequently found in all the
        baskets.

        Based on the apriori algorithm in the book mining big datasets.

        :param baskets: list[set[string]]: collection of all sets, each representing a document of
        strings
        :param k: int: the amount of strings in the set
        :param support_threshold: the minimal amount of times a candidate should be present
        """
        self.baskets = baskets
        self.k_max = k
        self.k = 0
        self.support_threshold = support_threshold

    def construct_candidates(self, filtered: list) -> list:
        """
        Construct all possible candidates that you can find in the baskets.

        A candidate is valid if the amount of strings in the set is equal to k. In addition, if a
        previous run has already been performed, each subset of the candidate must be found in the
        filtered collection.

        :param filtered: list[frozenset[set[str]]]: collection of all subsets of strings that
        fulfilled the filter criteria of the previous k-value.
        :return: list[set[str]]: a collection of all possible candidates
        """
        candidates = list()

        if self.k == 1:
            for basket in self.baskets:
                for item in basket:
                    if {item} not in candidates:
                        candidates.append({item})
            return candidates
        else:
            for i, s1 in enumerate(filtered):
                for s2 in filtered[i + 1:len(filtered)]:
                    candidate = set(s1).union(set(s2))
                    if len(candidate) == self.k:
                        if candidate not in candidates:
                            candidates.append(candidate)

        return candidates

    def count_candidates(self, candidates) -> dict:
        """
        This function will count the amount of times each candidate is present in the baskets.

        :param candidates: list[frozenset[set[str]]]: a collection of all possible candidates
        :return: dict[frozenset -> int]: a dict that maps all possible candidates to the amount of
        times it can be found in the baskets.
        """
        candidates_count = {frozenset(c): 0 for c in candidates}
        for basket in self.baskets:
            for subset in getSubsets(basket, self.k):
                if subset in candidates_count:
                    candidates_count[subset] += 1

        return candidates_count

    def filter_candidates(s, candidatesCount: dict) -> list:
        """
        This function will only return the candidates that have been counted at least
        support_threshold times.

        :param candidatesCount: dict[frozenset -> int]: a dict that maps all possible candidates to
        the amount of times it can be found in the baskets.
        :return: list[frozenset[set[str]]]: collection of all subsets of strings that
        fulfilled the filter criteria.
        """
        filteredCandidates = [k for k, v in candidatesCount.items() if v >= s.support_threshold]

        return filteredCandidates

    def get_frequent_sets(self):
        """
        This function will return all sets with k strings that are found supportThreshold times in
        the baskets.

        :return: list[frozenset[set[string]]]: collection of all sets of k strings that were found
        support_threshold times in the baskets.
        """
        filtered_candidates = None

        for self.k in range(1, (self.k_max + 1)):
            candidates = self.construct_candidates(filtered_candidates)
            counted_candidates = self.count_candidates(candidates)
            filtered_candidates = self.filter_candidates(counted_candidates)

        return filtered_candidates


if __name__ == "__main__":
    baskets = generateDocuments()

    print("BASKETS")
    for basket in baskets:
        print(basket)

    # Use the function getFrequentSets for k = 1, 2 and 3
    runner = Apriori(baskets=baskets, k=K, support_threshold=SUPPORT_THRESHOLD)

    start_time = timeit.default_timer()
    frequents = runner.get_frequent_sets()
    elapsed = timeit.default_timer() - start_time

    print(elapsed)
    print("\nRESULTS = \n", frequents)
