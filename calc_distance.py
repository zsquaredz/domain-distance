from sklearn.metrics import hinge_loss
from sklearn import svm
from get_features import *
from experiments import train_linear_classifier
import os
import utils


def calc_distance(category_list, ngram, min_freq, proxy_name, clf_name=None):
    if proxy_name == "cls":
        if clf_name == "LinearSVC":
            clf = svm.LinearSVC(random_state=42, loss="hinge")
        return cls_a_distance(category_list, ngram, min_freq, clf)


def cls_a_distance(category_list, ngram, min_freq, clf):
    filename = "flies/cls_a_distance"
    if os.path.isfile(filename):
        cls_a_distance = utils.load_file(filename)
        return cls_a_distance
    cls_a_distance = {}
    print("########## Calculating cls-a-distance ##########")
    for src in category_list:
        for trg in category_list:
            if src != trg:
                if (trg, src) in cls_a_distance:
                    cls_a_distance[(src, trg)] = cls_a_distance[(trg, src)]
                    continue
                X_train_src, y_train_src, X_test_src, y_test_src = utils.load_existing_dataset(src)
                X_train_trg, y_train_trg, X_test_trg, y_test_trg = utils.load_existing_dataset(trg)
                fne = NgramFeatureExtractor(X_train_src + X_train_trg, ngram, min_freq)
                fne.fit_vectorizer()
                domain_labels = [0] * len(X_train_src) + [1] * len(X_train_trg)
                train_linear_classifier(fne, clf, domain_labels)
                pred_decision = clf.decision_function(fne.transform_vectorizer(X_test_src + X_test_trg))
                a_distance = hinge_loss(domain_labels, pred_decision)
                cls_a_distance[(src, trg)] = 1-a_distance
                print("cls-a-distance between {} and {} is {}".format(src, trg, round(1-a_distance, 3)))

    utils.save_file(filename, cls_a_distance)
    return cls_a_distance
