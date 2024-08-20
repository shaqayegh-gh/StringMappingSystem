import abc
import re
import unicodedata


class StringPreprocessorInterface(abc.ABC):

    @abc.abstractmethod
    def clean_sting(self, text: str) -> str:
        pass


class StringPreprocessor(StringPreprocessorInterface):

    @staticmethod
    def lower_casing(text):
        return text.lower()

    @staticmethod
    def removing_punctuation(text):
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def removing_whitespace(text):
        text = ' '.join(text.split())
        return text

    @staticmethod
    def unicode_normalization(text):
        text = unicodedata.normalize('NFC', text)
        return text

    def clean_sting(self, text: str) -> str:
        text = self.lower_casing(text)
        text = self.removing_punctuation(text)
        text = self.removing_whitespace(text)
        text = self.unicode_normalization(text)
        return text


class CleanString:
    def __init__(self, text, pre_processor: StringPreprocessorInterface):
        self.text = text
        self.pre_processor = pre_processor

    def __clean_string(self) -> str:
        text = self.pre_processor.clean_sting(self.text)
        return text

    def __call__(self, *args, **kwargs):
        return self.__clean_string()
