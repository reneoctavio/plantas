import numpy as np
import sklearn.metrics
from scipy.io import loadmat

import numpy as np

top5 = loadmat('/Users/reneoctavio/Dropbox/Plantas/vlfeat/Full/ex-plantas-vlad-aug/prediction-top-5-test.mat')['pred5'].transpose() - 1
true = loadmat('/Users/reneoctavio/Dropbox/Plantas/vlfeat/Full/ex-plantas-vlad-aug/true-label-test.mat')['true_labels'][0] - 1
scores = loadmat('/Users/reneoctavio/Dropbox/Plantas/vlfeat/Full/ex-plantas-vlad-aug/prediction-score-test.mat')['test_scores'].transpose() - 1
# top5 = np.load('/Users/reneoctavio/Dropbox/Plantas/Training/Full/CaffeNet-SVM/prediction-top-5-test.npy')
#
# true = np.load('/Users/reneoctavio/Dropbox/Plantas/Training/Full/CaffeNet-SVM/true-label-test.npy')
#
# scores = np.load('/Users/reneoctavio/Dropbox/Plantas/Training/Full/CaffeNet-SVM/prediction-score-tes.npy')

top1 = top5[:, 0]

num_classes = len(np.unique(true))

y_true = np.zeros((len(true), num_classes)) - 1.0

for img in np.arange(len(true)):
    y_true[img, true[img]] = 1.0
    # max_num = np.amax(scores[img, :])
    # min_num = np.amin(scores[img, :])
    # diff = max_num - min_num
    # scores[img, :] = scores[img, :] + diff
    # scores[img, :] = scores[img, :] / np.amax(scores[img, :])


print(scores)
print(y_true)

print(sklearn.metrics.average_precision_score(y_true, scores, average='macro'))
print(sklearn.metrics.average_precision_score(y_true, scores, average='weighted'))



# print(sklearn.metrics.precision_recall_fscore_support(true, top1))

# print(sklearn.metrics.precision_recall_fscore_support(true, top1, average='micro'))
#
# print(sklearn.metrics.precision_recall_fscore_support(true, top1, average='macro'))
#
# print(sklearn.metrics.precision_recall_fscore_support(true, top1, average='weighted'))
#
# ttop5 = []
# for i in range(0, len(true)):
#     if true[i] in top5[i, :]:
#         ttop5.append(true[i])
#     else:
#         ttop5.append(top5[i, 0])

# print(sklearn.metrics.precision_recall_fscore_support(true, ttop5, average='weighted'))

# prec, rec, f1, sup = sklearn.metrics.precision_recall_fscore_support(true, top1)
# print(np.mean(prec), np.mean(rec))
#
# print(np.sum(prec * sup)/np.sum(sup), np.sum(rec * sup)/np.sum(sup))

# print(sklearn.metrics.precision_recall_fscore_support(true, ttop5, average='samples'))
