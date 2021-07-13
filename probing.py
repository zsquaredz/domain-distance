import statistics
import utils
from get_features import *


def calc_stdev(score_list):
    variance = statistics.stdev(score_list)
    return variance


def calc_src_stdev(src_category, category_list, relative_loss):
    loss_list = []
    for category in category_list:
        if category != src_category:
            src_trg_relative_loss = relative_loss[(src_category, category)]
            loss_list.append(src_trg_relative_loss)
    std = calc_stdev(loss_list)
    print("with {} as source the std is {}".format(src_category, round(std, 3)))
    return std


def get_top_domain_features(src, trg, top_k, ngram, min_freq):
    features = []
    X_train_src, _, _, _ = utils.load_existing_dataset(src)
    X_train_trg, _, _, _ = utils.load_existing_dataset(trg)
    domain_labels = [0] * len(X_train_src) + [1] * len(X_train_trg)
    fne = NgramFeatureExtractor(X_train_src + X_train_trg, ngram, min_freq)
    fne.fit_vectorizer()
    MIsorted, RMI = utils.get_top_NMI(top_k, fne.train_feature_matrix, domain_labels)
    MIsorted.reverse()
    for i in range(top_k):
        feature_name = fne.vectorizer.get_feature_names()[MIsorted[i]]
        print(feature_name)
        features.append(feature_name)
    return features


def get_top_sentiment_features(category, top_k, ngram, min_freq):
    features = []
    X_train, y_train, _, _ = utils.load_existing_dataset(category)
    fne = NgramFeatureExtractor(X_train, ngram, min_freq)
    fne.fit_vectorizer()
    MIsorted, RMI = utils.get_top_NMI(top_k, fne.train_feature_matrix, y_train)
    MIsorted.reverse()

    for i in range(top_k):
        feature_name = fne.vectorizer.get_feature_names()[MIsorted[i]]
        s_count = utils.get_counts(fne.train_feature_matrix, fne.vectorizer.get_feature_names().index(
            feature_name)) if feature_name in fne.vectorizer.get_feature_names() else 0
        if s_count > min_freq:
            print(feature_name)
            features.append(feature_name)
    return features


def check_feature_overlap(domain_features, sentiment_features):
    feature_overlap = utils.overlap(domain_features, sentiment_features)
    nubmer_of_features = len(domain_features)
    number_of_overlap = len(feature_overlap)
    print("###################3")
    for f in feature_overlap:
        print(f)
    print("out of {} features, {} overlap".format(nubmer_of_features, number_of_overlap))





