import pickle
import os
from sklearn.metrics import mutual_info_score


def load_existing_dataset(data_dir, category):
    reviews_pattern = os.path.join(data_dir, "/X_" + category + "_5.pkl")
    labels_pattern = os.path.join(data_dir, "y_" + category + "_5.pkl")
    with open(reviews_pattern, 'rb') as f:
        reviews = pickle.load(f)
    with open(labels_pattern, 'rb') as f:
        labels = pickle.load(f)
    X_train, y_train, X_test, y_test = [], [], [], []
    pos_count = 0
    neg_count = 0
    for i in range(len(reviews)):
        if labels[i] == 0:
            if neg_count % 2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
            else:
                X_test.append(reviews[i])
                y_test.append(labels[i])
            neg_count += 1
        elif labels[i] == 1:
            if pos_count % 2:
                X_train.append(reviews[i])
                y_train.append(labels[i])
            else:
                X_test.append(reviews[i])
                y_test.append(labels[i])
            pos_count += 1
    return X_train, y_train, X_test, y_test


def save_file(path, file):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file


def get_sorted_loss(src_category, category_list, loss, to_reverse):
    loss_list = []
    for category in category_list:
        if category != src_category:
            src_trg_relative_loss = loss[(src_category, category)]
            loss_list.append((category, src_trg_relative_loss))
    loss_list.sort(key=lambda x: x[1], reverse=to_reverse)
    print("The relative loss from {} is :".format(src_category))
    for i in range(len(loss_list)):
        print(loss_list[i])
    return loss_list


def get_counts(X, i):
    return (sum(X[:,i]))


def get_top_NMI(n, X, target):
    MI = []
    length = X.shape[1]
    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs, MI


def overlap(list1, list2):
    return list(set(list1) & set(list2))

