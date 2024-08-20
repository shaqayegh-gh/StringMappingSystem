import abc

from Levenshtein import distance
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessors import CleanString, StringPreprocessor


class StringsSimilarityInterface(abc.ABC):

    @abc.abstractmethod
    def find_similarity_percentage(self, text_1: str, text_2: str) -> float:
        pass


class LevenshteinSimilarity(StringsSimilarityInterface):

    def find_similarity_percentage(self, text_1: str, text_2: str) -> float:
        # Calculate the Levenshtein Distance
        levenshtein_dist = distance(text_1, text_2)

        # Determine the maximum length of the strings
        max_length = max(len(text_1), len(text_2))

        # Calculate the similarity percentage
        similarity_percent = ((max_length - levenshtein_dist) / max_length) * 100

        return similarity_percent


class FuzzyLevenshteinSimilarity(StringsSimilarityInterface):

    def find_similarity_percentage(self, text_1: str, text_2: str) -> float:
        return fuzz.ratio(text_1, text_2)


class CosineSimilarity(StringsSimilarityInterface):

    def find_similarity_percentage(self, text_1: str, text_2: str) -> float:
        vector = CountVectorizer().fit([text_1, text_2])
        vectors = vector.transform([text_1, text_2])
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0][1] * 100


class JaccardSimilarity(StringsSimilarityInterface):
    def find_similarity_percentage(self, text_1: str, text_2: str) -> float:
        set1 = set(text_1)
        set2 = set(text_2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # Calculate Jaccard similarity coefficient
        jaccard_similarity = intersection / union if union else 0

        # Convert Jaccard similarity coefficient to percentage (scale 0-100)
        similarity_percentage = jaccard_similarity * 100
        return similarity_percentage


class StringSimilarityFactoryService:

    def __init__(self, text_1: str, text_2: str, similarity_method: StringsSimilarityInterface):
        self.text_1 = text_1
        self.text_2 = text_2
        self.similarity_method = similarity_method

    def __call__(self, *args, **kwargs):
        return self.similarity_method.find_similarity_percentage(text_1=self.text_1, text_2=self.text_2)


if __name__ == '__main__':
    text1 = 'پژو 207I-MT'
    text2 = 'پژو 207'
    result1 = CleanString(text=text1, pre_processor=StringPreprocessor())()
    result2 = CleanString(text=text2, pre_processor=StringPreprocessor())()

    for method in [LevenshteinSimilarity, CosineSimilarity, JaccardSimilarity, FuzzyLevenshteinSimilarity]:
        print(StringSimilarityFactoryService(result1, result2, method())())
