from sortedcontainers import SortedSet


def get_subsets(set1: set, k: int) -> SortedSet:
    """
    Get the subsets from your basket.
    """

    def getSubsets_(set1: set, subset: set, subsetSize: int, candidates: SortedSet):
        if subsetSize == len(subset):
            candidates.add(frozenset(x for x in subset))
        else:
            for s in set1:
                subset.add(s)
                clone = set(set1)
                clone.remove(s)
                getSubsets_(clone, subset, subsetSize, candidates)
                subset.remove(s)

    result: SortedSet = SortedSet()
    getSubsets_(set(set1), set(), k, result)
    return result


def split_documents(documents: list) -> list:
    """
    Split the collection of documents into buckets.

    :param documents: list[str]: a collection of all the documents
    :return: list[set[str]]: a collection of baskets
    """

    return [set(document.lower().split(" ")) for document in documents]


def generate_documents() -> list:
    """
    Generate a collection of documents to play around with.

    :return: a collection of documents
    """

    return split_documents([
        "Cat and dog bites",
        "Yahoo news claims a cat mated with a dog and produced viable offspring",
        "Cat killer likely is a big dog",
        "Professional free advice on dog training puppy training",
        "Cat and kitten training and behavior",
        "Dog & Cat provides dog training in Eugene Oregon",
        "Dog and cat is a slang term used by police officers for a male female relationship",
        "Shop for your show dog grooming and pet supplies"
    ])
