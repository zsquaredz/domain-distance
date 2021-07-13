from get_features import *
import utils
from sklearn import svm
import os
import pickle


def train_linear_classifier(feature_extractor, clf, y_train):
    feature_extractor.fit_vectorizer()
    clf.fit(feature_extractor.train_feature_matrix, y_train)
    return clf


def predict_linear_classifier(feature_extractor, clf, X_test, y_test):
    X_test = feature_extractor.transform_vectorizer(X_test)
    score = clf.score(X_test, y_test)
    return score


def fill_category_classifier_dict(category_list, ngram, min_freq, clf_name):
    filename = "flies/clf_for_category"
    if os.path.isfile(filename):
        clf_for_category = utils.load_file(filename)
        return clf_for_category
    clf_for_category = {}
    print("########## Training Classifiers ##########")
    for category in category_list:
        X_train, y_train, X_test, y_test = utils.load_existing_dataset(category)
        fne = NgramFeatureExtractor(X_train, ngram, min_freq)
        if clf_name == "LinearSVC":
            clf = svm.LinearSVC(random_state=42, loss="hinge")
        clf = train_linear_classifier(fne, clf, y_train)
        print("trained classifier for {}".format(category))
        clf_for_category[category] = clf

    utils.save_file(filename, clf_for_category)
    return clf_for_category


def fill_in_domain_performance_dict(category_list, clf_for_category, ngram, min_freq):
    filename = "flies/in_domain_results"
    if os.path.isfile(filename):
        in_domain_results = utils.load_file(filename)
        return in_domain_results
    in_domain_results = {}
    print("########## In-Domain Performance ##########")
    for category in category_list:
        X_train, y_train, X_test, y_test = utils.load_existing_dataset(category)
        clf = clf_for_category[category]
        fne = NgramFeatureExtractor(X_train, ngram, min_freq)
        fne.fit_vectorizer()
        score = predict_linear_classifier(fne, clf, X_test, y_test)
        in_domain_results[category] = score
        print("trained classifier for {} got {} accuracy".format(category, round(score, 3)))
    utils.save_file(filename, in_domain_results)
    return in_domain_results


def fill_cross_domain_performance_dict(category_list, clf_for_category, ngram, min_freq):
    filename = "flies/cross_domain_results"
    if os.path.isfile(filename):
        cross_domain_results = utils.load_file(filename)
        return cross_domain_results
    cross_domain_results = {}
    print("########## Cross-Domain Performance ##########")
    for src in category_list:
        for trg in category_list:
            if src != trg:
                X_train_src, y_train_src, _, _ = utils.load_existing_dataset(src)
                _, _, X_test_trg, y_test_trg = utils.load_existing_dataset(trg)
                clf = clf_for_category[src]
                fne = NgramFeatureExtractor(X_train_src, ngram, min_freq)
                fne.fit_vectorizer()
                score = predict_linear_classifier(fne, clf, X_test_trg, y_test_trg)
                cross_domain_results[(src, trg)] = round(score, 3)
                print("trained on {} tested on {} got {} accuracy".format(src, trg, round(score, 3)))
    utils.save_file(filename, cross_domain_results)
    return cross_domain_results


def calc_relative_loss(category_list, in_domain_results, cross_domain_results):
    relative_loss = {}
    print("########## Calculating relative loss ##########")
    for src in category_list:
        for trg in category_list:
            if src != trg:
                score = cross_domain_results[(src, trg)] / in_domain_results[trg]
                relative_loss[(src, trg)] = round(score, 3)
                print("trained on {} tested on {} got {} from in domain performance".format(src, trg, round(score, 3)))
    return relative_loss











