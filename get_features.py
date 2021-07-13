from sklearn.feature_extraction.text import CountVectorizer


# This class is designed to extract features from text
class FeatureExtractor:

    def __init__(self, training_data):
        self.training_data = training_data


class NgramFeatureExtractor(FeatureExtractor):
    def __init__(self, training_data, ngram, min_freq):
        super().__init__(training_data)
        self.ngram = ngram
        self.min_freq = min_freq
        self.vectorizer = None
        self.train_feature_matrix = None

    def fit_vectorizer(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, self.ngram), token_pattern=r'\b\w+\b', min_df=self.min_freq,
                                          binary=True)
        self.train_feature_matrix = self.vectorizer.fit_transform(self.training_data).toarray()

    def transform_vectorizer(self, test_data):
        test_feature_matrix = self.vectorizer.transform(test_data).toarray()
        return test_feature_matrix


