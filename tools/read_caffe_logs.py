from caffe_extra.parse_log import parse_log
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

arch_path = '/Users/reneoctavio/Dropbox/Plantas/Training/'
archictetures_core_path = '/Users/reneoctavio/Dropbox/Plantas/Training/Core'
archictetures_full_path = '/Users/reneoctavio/Dropbox/Plantas/Training/Full'
vlfeat_path = '/Users/reneoctavio/Dropbox/Plantas/vlfeat/'
vlfeat_core_path = '/Users/reneoctavio/Dropbox/Plantas/vlfeat/core'
vlfeat_full_path = '/Users/reneoctavio/Dropbox/Plantas/vlfeat/complete'


labels_file = '/Users/reneoctavio/Dropbox/Plantas/labels.txt'

def read_labels(labels_file):
    f = open(labels_file, 'r')
    labels = []
    for line in f.readlines():
        labels.append(line.rstrip('\n'))
    return labels

def logs_to_dataframes(arch_path, networks):
    for arch, derivs in networks.iteritems():
        for deriv in derivs:
            net_name = arch + '-' + deriv
            print("Reading log of: " + net_name)
            path = os.path.join(arch_path, net_name)
            logfile_path = os.path.join(path, 'caffe_output.log')
            train_dict_list, test_dict_list = parse_log(logfile_path)

            df_train = pd.DataFrame(train_dict_list)
            df_test = pd.DataFrame(test_dict_list)

            df_train.to_csv(os.path.join(path, net_name + '-train-log.csv'))
            df_test.to_csv(os.path.join(path, net_name + '-test-log.csv'))


def plot_confusion_matrix(cm, labels, output_file, title='Confusion matrix', cmap=plt.cm.Blues):
    # Resize plots
    plt.rcParams['figure.figsize'] = (12.0, 10.0)
    # matplotlib.rcParams.update({'font.size': 8})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=60, ha='right')
    # plt.yticks(tick_marks, labels)

    plt.xticks(tick_marks, rotation=60)
    plt.yticks(tick_marks)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()


def calc_precision_recall(arch_path, set_name, networks, labels_file, mode='caffe'):
    # All net stats
    net_stats = {}
    net_stats['accuracy_top_1'] = {}
    net_stats['accuracy_top_5'] = {}

    net_stats['map_macro'] = {}
    net_stats['map_weighted'] = {}

    net_stats['precision_top_1_macro'] = {}
    net_stats['recall_top_1_macro'] = {}
    net_stats['f1_score_top_1_macro'] = {}

    net_stats['precision_top_1_weighted'] = {}
    net_stats['recall_top_1_weighted'] = {}
    net_stats['f1_score_top_1_weighted'] = {}

    net_stats['precision_top_5_macro'] = {}
    net_stats['recall_top_5_macro'] = {}
    net_stats['f1_score_top_5_macro'] = {}

    net_stats['precision_top_5_weighted'] = {}
    net_stats['recall_top_5_weighted'] = {}
    net_stats['f1_score_top_5_weighted'] = {}

    for arch, derivs in networks.iteritems():
        for deriv in derivs:
            if deriv != '':
                net_name = arch + '-' + deriv
            else:
                net_name = arch
            path = os.path.join(arch_path, set_name, net_name)

            # Read files
            if mode == 'caffe':
                predictions = np.load(os.path.join(path, 'prediction-top-5-test.npy'))
                true_values = np.load(os.path.join(path, 'true-label-test.npy'))
                scores = np.load(os.path.join(path, 'prediction-score-test.npy'))
            elif mode == 'vlfeat':
                predictions = loadmat(os.path.join(path, 'prediction-top-5-test.mat'))['pred5'].transpose() - 1
                true_values = loadmat(os.path.join(path, 'true-label-test.mat'))['true_labels'][0] - 1
                scores = loadmat(os.path.join(path, 'prediction-score-test.mat'))['test_scores'].transpose() - 1
            else:
                raise ValueError('Invalid mode')

            # Organize by label
            labels = read_labels(labels_file)
            label_pos = np.array(labels).argsort()
            pred = np.array([[label_pos[x] for x in y] for y in predictions])
            true = np.array([label_pos[x] for x in true_values])

            # Get only first prediction
            pred_top_1 = pred[:,0]

            # Confusion Matrix
            cm = sklearn.metrics.confusion_matrix(true, pred_top_1)

            # Normalize by row
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(precision=2)

            # Save figure
            output_file = os.path.join(arch_path, set_name + '-' + net_name + '.png')
            plot_confusion_matrix(cm_normalized, labels, output_file, title='Normalized Confusion Matrix of ' + net_name + '(' + set_name + ')')

    #         # Save precision recall per net
    #         f_report = open(os.path.join(path, 'class_report.csv'), 'w')
    #         f_report.write(sklearn.metrics.classification_report(true, pred_top_1, target_names=labels))
    #         f_report.close()
    #
    #         # Accuracy
    #         # Top-1
    #         acc_1 = sklearn.metrics.accuracy_score(true, pred_top_1)
    #         # Top-5
    #         top5 = []
    #         for i in range(0, len(true)):
    #             if true[i] in pred[i, :]:
    #                 top5.append(true[i])
    #             else:
    #                 top5.append(pred[i, 0])
    #         acc_5 = sklearn.metrics.accuracy_score(true, top5)
    #
    #         # Mean Average Precision
    #         map_true = np.zeros((len(true), len(np.unique(true)))) - 1.0
    #         for img_idx in np.arange(len(true)):
    #             map_true[img_idx, true[img_idx]] = 1.0
    #
    #         # Get precision recall for all nets
    #         prec_top_1_macro, rec_top_1_macro, f1_top_1_macro, _ = sklearn.metrics.precision_recall_fscore_support(true, pred_top_1, average='macro')
    #         prec_top_1_weighted, rec_top_1_weighted, f1_top_1_weighted, _ = sklearn.metrics.precision_recall_fscore_support(true, pred_top_1, average='weighted')
    #
    #         prec_top_5_macro, rec_top_5_macro, f1_top_5_macro, _ = sklearn.metrics.precision_recall_fscore_support(true, top5, average='macro')
    #         prec_top_5_weighted, rec_top_5_weighted, f1_top_5_weighted, _ = sklearn.metrics.precision_recall_fscore_support(true, top5, average='weighted')
    #
    #         # Put in the frame
    #         net_stats['accuracy_top_1'][net_name] = acc_1
    #         net_stats['accuracy_top_5'][net_name] = acc_5
    #
    #         net_stats['map_macro'][net_name] = sklearn.metrics.average_precision_score(map_true, scores, average='macro')
    #         net_stats['map_weighted'][net_name] = sklearn.metrics.average_precision_score(map_true, scores, average='weighted')
    #
    #         net_stats['precision_top_1_macro'][net_name] = prec_top_1_macro
    #         net_stats['recall_top_1_macro'][net_name] = rec_top_1_macro
    #         net_stats['f1_score_top_1_macro'][net_name] = f1_top_1_macro
    #
    #         net_stats['precision_top_1_weighted'][net_name] = prec_top_1_weighted
    #         net_stats['recall_top_1_weighted'][net_name] = rec_top_1_weighted
    #         net_stats['f1_score_top_1_weighted'][net_name] = f1_top_1_weighted
    #
    #         net_stats['precision_top_5_macro'][net_name] = prec_top_5_macro
    #         net_stats['recall_top_5_macro'][net_name] = rec_top_5_macro
    #         net_stats['f1_score_top_5_macro'][net_name] = f1_top_5_macro
    #
    #         net_stats['precision_top_5_weighted'][net_name] = prec_top_5_weighted
    #         net_stats['recall_top_5_weighted'][net_name] = rec_top_5_weighted
    #         net_stats['f1_score_top_5_weighted'][net_name] = f1_top_5_weighted
    #
    # df = pd.DataFrame(net_stats)
    # df.to_csv(os.path.join(arch_path, set_name + '-new-stats.csv'))
    # print(df)

# networks = {'AlexNet': ['Default'],
#             'CaffeNet': ['Default', 'PReLU', 'Finetuned', 'Finetuned-Zero', 'ELU'],
#             'GoogLeNet': ['Default', 'Finetuned', 'PReLU', 'Finetuned-PReLU', 'Finetuned-Zero', 'ELU', 'Finetuned-ELU'],
#             'Inception': ['Default', 'ELU', 'PReLU'],
#             'ResNet-50': ['Default', 'PReLU', 'Finetuned'],
#             'ResNet-101': ['Default']}
#
# logs_to_dataframes(archictetures_full_path, networks)
# logs_to_dataframes(archictetures_core_path, networks)

# networks = {'AlexNet': ['Default'],
#             'CaffeNet': ['Default', 'PReLU', 'Finetuned', 'Finetuned-Zero', 'ELU', 'SVM'],
#             'GoogLeNet': ['Default', 'Finetuned', 'PReLU', 'Finetuned-PReLU', 'Finetuned-Zero', 'ELU', 'SVM'],
#             'Inception': ['Default', 'ELU', 'PReLU'],
#             'ResNet-50': ['Default', 'PReLU', 'Finetuned', 'SVM'],
#             'ResNet-101': ['SVM'],
#             'ResNet-152': ['SVM']}

networks = {'ResNet-152': ['SVM']}
# calc_precision_recall(arch_path, 'Core', networks, labels_file)
calc_precision_recall(arch_path, 'Full', networks, labels_file)

# encoders = {'ex-plantas-bovw': ['aug'],
#             'ex-plantas-fv': ['', 'aug'],
#             'ex-plantas-vlad': ['aug'] }
#
# calc_precision_recall(vlfeat_path, 'Core', encoders, labels_file, mode='vlfeat')
# calc_precision_recall(vlfeat_path, 'Full', encoders, labels_file, mode='vlfeat')
