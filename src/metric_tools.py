import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from scipy import interp
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import tarfile
from sklearn.metrics import roc_curve, precision_recall_curve

sns.set(style="white", color_codes=True, context="poster")
mainpath = '/Users/jorgetil/Astro/HITS'


def bagging_PU(X, y, T=100, K=100, known_labels_ratio=0.1, plot=True):
    N = X.shape[0]
    print('pos/neg:', y[y == 1].shape[0], '/', y[y == 0].shape[0])
    rp = np.random.permutation(len(y[y == 1]))
    data_P = X[y == 1][rp[:int(len(rp)*known_labels_ratio)]]
    data_U = np.concatenate(
        (X[y == 1][rp[int(len(rp)*known_labels_ratio):]], X[y == 0]), axis=0)
    y_U = np.concatenate(
        (y[y == 1][rp[int(len(rp)*known_labels_ratio):]], y[y == 0]), axis=0)
    print("Amount of labeled samples: %d" % (data_P.shape[0]))

    NP = data_P.shape[0]
    NU = data_U.shape[0]

    train_label = np.zeros(shape=(NP+K,))
    train_label[:NP] = 1.0
    n_oob = np.zeros(shape=(NU,))
    f_oob = np.zeros(shape=(NU, 2))
    for i in range(T):
        # Bootstrap resample
        bootstrap_sample = np.random.choice(
            np.arange(NU), replace=True, size=K)
        # Positive set + bootstrapped unlabeled set
        data_bootstrap = np.concatenate(
            (data_P, data_U[bootstrap_sample, :]), axis=0)

        #plt.figure(figsize=(8, 4.5))
        # sp = plt.scatter(data_bootstrap[:, 0], data_bootstrap[:, 1],
        #                 c=train_label, marker='.',
        #                 linewidth=0, s=20, alpha=0.9, cmap=plt.cm.plasma)
        #plt.colorbar(sp, label='Class probability on Unlabeled set')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # Train model
        model = DecisionTreeClassifier(max_depth=None, max_features=None,
                                       criterion='gini', class_weight='balanced')
        model.fit(data_bootstrap, train_label)
        # Index for the out of the bag (oob) samples
        idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
        # Transductive learning of oob samples
        f_oob[idx_oob] += model.predict_proba(data_U[idx_oob])
        n_oob[idx_oob] += 1
    predict_proba = f_oob[:, 1]/n_oob

    if plot:
        '''
        plt.figure(figsize=(8, 4.5))
        plt.scatter(data_U[:, 0], data_U[:, 1], c='k', marker='.',
                    linewidth=1, s=5, alpha=0.7, label='Unlabeled')
        plt.scatter(data_P[:, 0], data_P[:, 1], c='b', marker='o',
                    linewidth=0, s=10, alpha=0.5, label='Positive')
        plt.grid()
        plt.legend()
        plt.show()
        '''
        # Plot the class probabilities for the unlabeled samples
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        sp = ax1.scatter(data_U[:, 0], data_U[:, 1], c=predict_proba,
                         linewidth=0, s=10, alpha=0.7, cmap=plt.cm.plasma, label='unlabeled')
        plt.grid()
        plt.colorbar(sp, label='Class probability on Unlabeled set')

        precision, recall, th = metrics.precision_recall_curve(
            y_U, predict_proba)
        ax2 = fig.add_subplot(1, 2, 2)
        f1s = precision[:-1]*recall[:-1]
        ax2.plot(th, f1s, linewidth=2, alpha=0.5)
        best_th = np.argmax(f1s)
        ax2.plot(th[best_th], f1s[best_th], c='r', marker='o')
        ax2.plot([th[best_th], th[best_th]], [0.0, f1s[best_th]], 'r--')
        ax2.plot([0.0, th[best_th]], [f1s[best_th], f1s[best_th]], 'r--')
        ax2.annotate('Pre: %0.3f, Rec: %0.3f' % (precision[best_th], recall[best_th]),
                     xy=(th[best_th] + 0.01, f1s[best_th]-0.05))
        ax2.set_ylabel('F1 score')
        ax2.set_xlabel('Probability threshold')
        # ax1.legend()
        plt.grid()
        plt.show()
        print(precision[best_th], recall[best_th])


def ROC_F1_thresh_curves_binary(y_tru, y_prob):
    fpr, tpr, _ = roc_curve(y_tru.ravel(), y_prob)
    precision, recall, th = metrics.precision_recall_curve(
        y_tru.ravel(), y_prob)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(fpr, tpr, 'b-')
    ax1.set_ylabel('TPR')
    ax1.set_xlabel('FPR')
    ax1.grid()
    ax2 = fig.add_subplot(1, 2, 2)
    f1s = precision[:-1]*recall[:-1]
    ax2.plot(th, f1s, 'r', linewidth=3, alpha=1., label='f1')
    ax2.plot(th, precision[:-1], 'g', linewidth=2, alpha=.7, label='precision')
    ax2.plot(th, recall[:-1], 'k', linewidth=2, alpha=.7, label='recall')

    best_th = np.argmax(f1s)
    ax2.plot(th[best_th], f1s[best_th], c='r', marker='o')
    ax2.plot([th[best_th], th[best_th]], [0.0, f1s[best_th]],
             'r--', label='best_th = % %.3f' % (best_th))
    ax2.plot([0.0, th[best_th]], [f1s[best_th], f1s[best_th]], 'r--')
    ax2.annotate('Pre: %0.3f, Rec: %0.3f' % (precision[best_th], recall[best_th]),
                 xy=(th[best_th] - 0.22, f1s[best_th]-0.075), size='small')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Probability threshold')
    ax2.legend(loc='lower left')
    ax2.set_ylim(.5, 1.1)
    ax2.grid()
    plt.show()


def ROC_F1_thresh_curves_multiclass(sample, y_score, classes, fontsize=24):
    import matplotlib.cm as cm
    # matrix with labels, shape (n_obj,n_class)
    y_test = np.zeros(y_score.shape, dtype=int)
    classes = np.asarray(classes)
    # print y_test
    # print classes
    for k in range(len(y_test)):
        mask = sample[k] == classes
        y_test[k, mask] = 1
    # print y_test
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        precision[i], recall[i], threshold[i] = metrics.precision_recall_curve(y_test[:, i],
                                                                               y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_test[:, i],
                                                               y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute micro-average P-R curve and ROC area
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(),
                                                                            y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_score,
                                                                 average="micro")

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= float(len(classes))

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all P-R and ROC curves
    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    ax[0].plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = %.4f)'
               % (roc_auc["micro"]), linewidth=4, color='b')

    ax[0].plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = %.4f)'
               % (roc_auc["macro"]), linewidth=4, color='g')

    colors = cm.rainbow(np.linspace(0, 1, len(classes)))
    for i in range(len(classes)):
        ax[0].plot(fpr[i], tpr[i], label='ROC curve of class %s (area = %.4f)'
                   % (classes[i], roc_auc[i]), linewidth=2, color=colors[i])
        ax[1].plot(threshold[i], precision[i][:-1] * recall[i][:-1], label='F1-thresh curve of class %s'
                   % (classes[i]), linewidth=2, color=colors[i], alpha=.9)

    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].set_xlabel('False Positive Rate', fontsize='large')
    ax[0].set_ylabel('True Positive Rate', fontsize='large')
    ax[0].set_title('ROC curves to multi-class', fontsize='large')
    ax[0].legend(loc="lower right", fontsize='small')
    if np.max(roc_auc['micro']) > .99:
        ax[0].set_ylim(.5, 1.)
        ax[0].set_xlim(0., .5)

    ax[1].set_xlabel('Probability threshold', fontsize='large')
    ax[1].set_ylabel('F1 score', fontsize='large')
    ax[1].set_title('F1 score as Probability threshold', fontsize='large')
    ax[1].legend(loc="lower left", fontsize='small')
    ax[1].set_ylim(0, 1.1)

    plt.show()


def PR_ROC_curves_multiclass(sample, y_score, classes, fontsize=24):
    # matrix with labels, shape (n_obj,n_class)
    y_test = np.zeros(y_score.shape, dtype=int)
    classes = np.asarray(classes)
    print(classes)
    print(y_score.shape)
    # print y_test
    # print classes
    for k in range(len(y_test)):
        mask = sample[k] == classes
        y_test[k, mask] = 1
    # print y_test
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_test[:, i],
                                                                    y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_test[:, i],
                                                               y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute micro-average P-R curve and ROC area
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(),
                                                                            y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_score,
                                                                 average="micro")

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= float(len(classes))

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all P-R and ROC curves
    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    ax[0].plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = %.4f)'
               % (roc_auc["micro"]), linewidth=4, color='b')
    ax[1].plot(recall["micro"], precision["micro"],
               label='micro-average Precision-recall curve (area = %.2f)'
               % (average_precision["micro"]), linewidth=4, color='b')

    ax[0].plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = %.4f)'
               % (roc_auc["macro"]), linewidth=4, color='g')

    colors = cm.rainbow(np.linspace(0, 1, len(classes)))
    for i in range(len(classes)):
        ax[0].plot(fpr[i], tpr[i], label='ROC curve of class %s (area = %.4f)'
                   % (classes[i], roc_auc[i]), linewidth=2, color=colors[i])
        ax[1].plot(recall[i], precision[i], label='P-R curve of class %s (area = %.4f)'
                   % (classes[i], average_precision[i]), linewidth=2, color=colors[i])

    ax[0].plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 0.4])
    #plt.ylim([0.6, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize='large')
    ax[0].set_ylabel('True Positive Rate', fontsize='large')
    ax[0].set_title('ROC curves to multi-class', fontsize='large')
    ax[0].legend(loc="lower right", fontsize='small')
    if np.max(roc_auc['micro']) > .99:
        ax[0].set_ylim(.5, 1.)
        ax[0].set_xlim(0., .5)

    #plt.xlim([0.0, 0.4])
    #plt.ylim([0.6, 1.05])
    ax[1].set_xlabel('Recall', fontsize='large')
    ax[1].set_ylabel('Precision', fontsize='large')
    ax[1].set_title(
        'Extended Precision-Recall curves to multi-class', fontsize='large')
    ax[1].legend(loc="lower left", fontsize='small')

    plt.show()


def conf_matrix(true=[], predict=[], classes=[], normalized=True, save=False,
                matrix_ready=False, matrix=[], class_names='same'):
    classes = sorted(classes)
    if not matrix_ready:
        cm = metrics.confusion_matrix(true, predict, labels=classes)
    else:
        cm = matrix
    if len(classes) < 4:
        font_size = 'large'
        figsize = (9, 7)
    else:
        font_size = 'small'
        figsize = (9, 7)
    fig, ax = plt.subplots(figsize=figsize)
    if class_names != 'same':
        classes = class_names
    if normalized:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', linewidths=.5,
                    xticklabels=classes, yticklabels=classes, cmap="GnBu",
                    annot_kws={'size': font_size}, ax=ax)
    else:
        if not matrix_ready:
            sns.heatmap(cm, annot=True, fmt='d', linewidths=.5,
                        xticklabels=classes, yticklabels=classes, cmap="GnBu",
                        annot_kws={'size': font_size}, ax=ax)
        else:
            sns.heatmap(cm, annot=True, fmt='.2f', linewidths=.5,
                        xticklabels=classes, yticklabels=classes, cmap="GnBu",
                        annot_kws={'size': font_size}, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    if save != False:
        plt.savefig(save, format='pdf', dpi=600,
                    bbox_inches='tight')
    else:
        plt.title('Confusion Matrix for RF classifier')
    plt.show()


def conv_into_binary_class(y, pos_label=None, neg_bool=False, neg_label=None):
    if type(pos_label) == str:
        pos_label = [pos_label]
    binary_class = np.zeros(shape=y.shape, dtype=np.int)
    if not neg_label:
        pos_idx = [x for x in range(len(y)) if y[x] in pos_label]
        binary_class[pos_idx] = 1
    else:
        neg_idx = np.where(y == neg_label)
        binary_class += 1
        binary_class[neg_idx] = 0

    return binary_class


def give_me_lc(field, CCD, X, Y, extract=False):
    year = field[:-3]
    try:
        tar = tarfile.open("%s/lightcurves/%s/%s/%s/%s_%s_LC_50.tar.gz"
                           % (mainpath, year, field, CCD, field, CCD))
        fil = tar.extractfile('%s_%s_%s_%s_g.dat' % (field, CCD, X, Y))
        if extract:
            tar.extract('%s_%s_%s_%s_g.dat' % (field, CCD, X, Y),
                        path='/Users/jorgetil/Astro/HITS/lightcurves/samples/.')
    except:
        print('No tar file or element in tar file')
        return None

    time, mag, err = [], [], []
    for line in fil:
        if line[0] == '#':
            continue
        values = line.split()
        time.append(float(values[1]))
        mag.append(float(values[2]))
        err.append(float(values[3]))
    time = np.asarray(time)
    mag = np.asarray(mag)
    err = np.asarray(err)

    try:
        fil = tar.extractfile('%s_%s_%s_%s_r.dat' % (field, CCD, X, Y))
        # tar.extract('%s_%s_%s_%s_r.dat' % (field, CCD, X, Y)
        #                , path='/Users/jorgetil/Downloads/.')
        time2, mag2, err2 = [], [], []
        for line in fil:
            if line[0] == '#':
                continue
            values = line.split()
            time2.append(float(values[1]))
            mag2.append(float(values[2]))
            err2.append(float(values[3]))
        time2 = np.asarray(time2)
        mag2 = np.asarray(mag2)
        err2 = np.asarray(err2)
        return time, mag, err, time2, mag2, err2
    except:
        print('No lightcurve for other filter')
        return time, mag, err, None, None, None
