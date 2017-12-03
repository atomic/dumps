import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
from sklearn.metrics         import confusion_matrix, roc_curve, roc_auc_score, auc, classification_report, f1_score, classification_report
from sklearn.externals       import joblib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import KFold


# Helper Functions
def analyze_scores(clf, X, y, name='default', with_auc=True, with_oob=True):
    # function to analyze scoring on classifier and test sets
    y_pred = clf.predict(X)
    score = clf.score(X, y)
    f1 = f1_score(y, y_pred, average='weighted')
    auc_score = np.NaN
    oob_score = np.NaN
    if with_auc:
        y_pred_prob = clf.predict_proba(X)[:, 1]
        fpr, tpr, thresh = roc_curve(y, y_pred_prob, pos_label=1)
        auc_score = auc(fpr, tpr)
        roc_auc = roc_auc_score(y, y_pred_prob, average='weighted')
    if with_oob:
        oob_score = clf.oob_score_
    overall = pd.Series([auc_score, score, f1, oob_score], index=['auc', 'accuracy', 'f1', 'oob'], name=name)
    cv_report = classifaction_report_df(y_true=y, y_pred=y_pred)
    cv_report['name'] = name
    return overall, cv_report


def full_analysis(clf, _X_train, _X_rus, _X_val, yt, yr, yv, with_oob=False):
    scores_trainRUS, report_trainRUS = analyze_scores(clf, _X_rus, yr, 'trainRUS', with_oob=with_oob)
    scores_train, report_train = analyze_scores(clf, _X_train, yt, 'train', with_oob=with_oob)
    scores_val, report_val = analyze_scores(clf, _X_val, yv, 'val', with_oob=with_oob)
    _A = pd.concat([report_trainRUS, report_train, report_val]).loc[0]
    _B = pd.concat([report_trainRUS, report_train, report_val]).loc[1]
    _C = pd.concat([scores_trainRUS, scores_train, scores_val], axis=1)
    results.append([_A, _B, _C])
    display_side_by_side(_C, _A, _B)


def get_splits_description(splits, names, normalized=True):
    label_counts = [pd.Series(s).value_counts() for s in splits]
    splits = df(label_counts, index=names)
    if normalized:
        return splits.div(splits.sum(axis=1), axis=0)
    return splits


def display_splits(splits, names, normalized=True):
    split_desc = get_splits_description(splits, names, normalized)
    display(split_desc)


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def classifaction_report_df(y_true, y_pred, target_names=None):
    report = classification_report(y_true, y_pred, target_names=target_names)
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

def report_cv(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")