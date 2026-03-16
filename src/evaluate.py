"""
evaluate.py
-----------
Model evaluation utilities for the PowerCo churn prediction project.
Extracted from 03_modeling.ipynb so the same plots and metrics can be
reproduced without copying notebook cells.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def print_classification_summary(y_true, y_pred, y_pred_proba):
    """
    Print a formatted summary of classification metrics.

    Accuracy is included but flagged as misleading — on a dataset where
    90.3% of customers did not churn, predicting "no churn" for everyone
    scores 90.3% accuracy while catching zero actual churners.
    Recall, precision, F1, and ROC-AUC tell the real story.

    Parameters
    ----------
    y_true       : array-like, ground truth labels
    y_pred       : array-like, predicted class labels (0 or 1)
    y_pred_proba : array-like, predicted probability of churn (class 1)
    """
    accuracy  = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall    = metrics.recall_score(y_true, y_pred)
    f1        = metrics.f1_score(y_true, y_pred)
    roc_auc   = metrics.roc_auc_score(y_true, y_pred_proba)
    pr_auc    = metrics.average_precision_score(y_true, y_pred_proba)

    print('=' * 52)
    print('   CLASSIFICATION METRICS')
    print('=' * 52)
    print(f'  Accuracy         : {accuracy:.4f}  (misleading on imbalanced data)')
    print(f'  Precision        : {precision:.4f}')
    print(f'  Recall           : {recall:.4f}')
    print(f'  F1 Score         : {f1:.4f}')
    print(f'  ROC-AUC          : {roc_auc:.4f}  (random baseline = 0.50)')
    print(f'  PR-AUC (Avg Prec): {pr_auc:.4f}  (random baseline = {y_true.mean():.4f})')
    print('=' * 52)
    print()
    print(metrics.classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc
    }


def plot_confusion_matrix(y_true, y_pred, figsize=(5, 4)):
    """
    Plot an annotated confusion matrix.

    How to read it:
      Top-left  = True Negatives  (correctly predicted no churn)
      Top-right = False Positives (predicted churn, customer stayed — wasted outreach)
      Bot-left  = False Negatives (predicted stayed, customer churned — missed opportunity)
      Bot-right = True Positives  (correctly predicted churn)

    Parameters
    ----------
    y_true  : array-like, ground truth labels
    y_pred  : array-like, predicted class labels
    figsize : tuple
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted: No Churn', 'Predicted: Churn'],
        yticklabels=['Actual: No Churn', 'Actual: Churn'],
        ax=ax
    )
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.show()

    print(f'True Negatives  : {tn:,}  (non-churners correctly cleared)')
    print(f'False Positives : {fp:,}  (non-churners incorrectly flagged)')
    print(f'False Negatives : {fn:,}  (churners the model missed)')
    print(f'True Positives  : {tp:,}  (churners correctly identified)')

    return cm


def plot_roc_curve(y_true, y_pred_proba, figsize=(6, 5)):
    """
    Plot the ROC curve with AUC annotated.

    The ROC curve shows how recall and false positive rate trade off
    across all possible classification thresholds. A perfect model hugs
    the top-left corner. The diagonal represents random chance (AUC = 0.50).

    Parameters
    ----------
    y_true       : array-like, ground truth labels
    y_pred_proba : array-like, predicted probability of churn
    figsize      : tuple
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    roc_auc = metrics.roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='#d73027', lw=2,
            label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--',
            label='Random classifier (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve — Churn Prediction')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(6, 5)):
    """
    Plot the Precision-Recall curve.

    For imbalanced datasets this is often more informative than the ROC curve
    because it focuses entirely on the minority class. The dashed horizontal line
    shows the precision of a random classifier — equal to the base churn rate.

    Parameters
    ----------
    y_true       : array-like, ground truth labels
    y_pred_proba : array-like, predicted probability of churn
    figsize      : tuple
    """
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_pred_proba)
    pr_auc = metrics.average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rec, prec, color='#4575b4', lw=2,
            label=f'Random Forest (AP = {pr_auc:.3f})')
    ax.axhline(y=y_true.mean(), color='grey', lw=1, linestyle='--',
               label=f'Random baseline (AP = {y_true.mean():.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Churn Prediction')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, figsize=(10, 6)):
    """
    Plot the top-N feature importances from a trained tree-based model.

    Importance is measured as mean decrease in Gini impurity across all trees.
    Higher means the model relies on that feature more heavily when making splits.

    Note: Gini importance can favour high-cardinality features. Permutation
    importance is a more reliable check if the results look suspicious.

    Parameters
    ----------
    model         : fitted sklearn model with feature_importances_ attribute
    feature_names : list of feature names in training column order
    top_n         : how many features to display
    figsize       : tuple
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    top[::-1].plot(kind='barh', ax=ax, color='#4575b4', edgecolor='white')
    ax.set_xlabel('Mean decrease in Gini impurity')
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest')
    plt.tight_layout()
    plt.show()

    print(f'\nTop 10 features:')
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    return importances
