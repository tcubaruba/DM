import numpy as np
import math
from scipy.io import arff
from io import StringIO
from bisect import bisect_left


def verify_counter(counter, all):
    sum = 0
    for e in counter:
        sum += counter[e]
    return sum == all


def chi_sq_statistics(counter_l, counter_r, all_l, all_r):
    chi_sq_stat = 0.
    sum_col = {}
    for key in counter_r:
        sum_col.update({key: counter_r[key] + counter_l[key]})
    n = all_r + all_l
    for key in sum_col:
        e_l = (sum_col[key] * all_l) / n
        e_r = (sum_col[key] * all_r) / n
        chi_sq_stat += ((counter_l[key] - e_l) ** 2) / e_l + ((counter_r[key] - e_r) ** 2) / e_r
    return chi_sq_stat


# compute entropy of given the dictionary counter (class, count) and the complete size of the set all
def compute_entropy(counter, all):
    entropy = 0.
    for key in counter:
        val = counter[key]
        entropy += -val / all * np.log(val / all)
    return entropy


# counts the occurence of each class label in labels
def class_counter(Y, class_labels):
    counter = {}  # empty dictionary
    for i in range(0, len(class_labels)):
        counter.update({class_labels[i]: (Y == class_labels[i]).sum()})
    return counter


# load data from file: meta contains attribute descriptions
# data is an array of record objects
data, meta = arff.loadarff("iris.arff")

# extract label colums
Y = np.asarray([item.decode("utf-8") for item in data['class']])

# select l attributes
l = 2
# data set as n objects
n = float(len(Y))
# data set has d features
d = len(meta.names()) - 1
# list of all class labels
class_labels = meta['class'][1]
# list of all attribute names
attribute_list = meta.names()[0:d]

# compute entropy of the complete data set
label_dict = class_counter(Y, class_labels)
entropy_all = compute_entropy(label_dict, n)

# Note, D is not needed here.
print('Entropy: %s' % entropy_all)
# collect the in info gain for each feature

Z = np.zeros(d)
# determine info gain for each dimension
i = 0
for attr in attribute_list:
    print(attr)
    # extract the i th column and sort it
    X = data[attr]
    srt = np.argsort(X)
    X = X[srt]
    # apply the same order to Y and build pairs of value and column
    Y2 = Y[srt]
    X2 = list(zip(X, Y2))
    print(X2)
    # Find all split values based on changing labels
    last_x = X2[0]
    splits = []
    for x in X2:
        if x[1] != last_x[1]:
            splits.append(x[0])
        last_x = x
    # Compute the Information Gain for all splits and keep the best
    bestinfogain = 0
    bestx2 = 0
    last_index = -1
    for split in splits:
        print('Dim: %s, Split at %s' % (i, split))
        # split the data and count the quantity for each label in each partition
        split_index = bisect_left(X, split)
        if split_index == last_index:
            continue
        # determine size of each partition
        all_l = float(split_index)
        all_r = float(n - (split_index))
        if all_l == 0 or all_r == 0: continue
        print('Split value: %s, Split index: %s' % (split, split_index))
        # count entries per class and then compute entropy
        counter_l = class_counter(Y2[0:split_index], class_labels)
        counter_r = class_counter(Y2[split_index:int(n)], class_labels)

        # compute entropy on both sides of the split
        # compute left and right entropy
        x2 = chi_sq_statistics(counter_l, counter_r, all_l, all_r)
        bestx2 = max(x2, bestx2)
        print('X2 : %s' % x2)
        # compute the info gain
        entropy_l = compute_entropy(counter_l, all_l)
        entropy_r = compute_entropy(counter_r, all_r)
        print('l: %s r:%s' % (entropy_l * all_l, entropy_r * all_r))
        # infogain = 0
        infogain = entropy_all - entropy_l * all_l - entropy_r * all_r
        # keep the highest gain
        bestinfogain = max(infogain, bestinfogain)
        print('Information Gain: %s' % infogain)
        last_index = split_index
    # collect the info gain for each attribute
    print('X2 : %s' % bestx2)
    print('Information Gain: %s' % bestinfogain)
    # Z[i]= bestinfogain
    Z[i] = bestx2
    i += 1
# determine order w.r.t. to the highest infogain
Z = list(zip(attribute_list, Z))
Z.sort(key=lambda item: item[1], reverse=True)
print(Z)
Z = Z[0:l]
Z = [z[0] for z in Z]
data_out = data[Z]
