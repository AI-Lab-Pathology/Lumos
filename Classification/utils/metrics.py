import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def compute_classification_report(y_true, y_pred, target_names):
    return classification_report(y_true, y_pred, target_names=target_names, digits=2)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def compute_multiclass_roc_auc(all_probs, all_labels, class_names):
    y_true = label_binarize(all_labels, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc
