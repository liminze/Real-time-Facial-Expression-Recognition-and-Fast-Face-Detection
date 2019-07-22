"""Callbacks: utilities called at certain points during model training.
"""
# https://blog.csdn.net/lulujiang1996/article/details/81540321

import keras
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


def paintRoc(y_true,y_score):
    fpr,tpr,thresholds=roc_curve(y_true.ravel(),y_score.ravel())
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=5,alpha=0.8,color='r',label='Roc(AUC=%0.2f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC_auc(AUC=%0.2f)'%(roc_auc))
    plt.legend(loc="lower right")
    plt.show()

def paintConfusion_float(ture_labels , pred_labels, labels_name, save_path=None):
    lmr_matrix = confusion_matrix(np.argmax(pred_labels, 1), np.argmax(ture_labels, 1))
    print('test_samples:', ture_labels.shape[0])
    print(lmr_matrix)
    lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    plt.imshow(lmr_matrix,interpolation='nearest',cmap=plt.cm.Oranges)
    # plt.imshow(lmr_matrix, interpolation='nearest', cmap='Greys')
    # plt.title('confusion matrix')
    plt.colorbar()
    tick_marks=np.arange(len(labels_name))
    # plt.xticks(tick_marks,labels_name,rotation=45)
    plt.xticks(tick_marks, labels_name)
    plt.yticks(tick_marks,labels_name)
    plt.xlabel('Pred label')
    plt.ylabel('True label')

    fmt='.2f'
    thresh=lmr_matrix.max()/2.
    for i,j in itertools.product(range(lmr_matrix.shape[0]),range(lmr_matrix.shape[1])):
        plt.text(j, i, format(lmr_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if lmr_matrix[i, j] > thresh else "black")
        plt.tight_layout()
    ################## 一般折线图的dpi设置为600，而图像的dpi设置为300############
    if save_path is not None:
        fig.savefig(save_path, dpi=300, format='png')

    plt.show()
    return 0