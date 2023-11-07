from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np


def get_tensorboard_data(train_loc, val_loc):
        train_df = pd.read_csv(train_loc)[1:]
        val_df = pd.read_csv(val_loc)[1:]
        l1 = plt.plot(train_df['Step'], train_df['Value'])
        l2 = plt.plot(val_df['Step'], val_df['Value'])
        plt.show()

def plot_curve_with_ci(X, 
                    y_mean, 
                    y_low, 
                    y_high, 
                    label, 
                    x_label, 
                    y_label, 
                    fig_title, 
                    color='grey',
                    fig_path=False):
    # plot line
    plt.plot(X, y_mean, label=label)
    # plot ci
    plt.fill_between(X, y_low, y_high, color=color, alpha=0.2,
                    label='95% CI')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # setting fig title
    plt.title(fig_title)
    plt.legend(loc='best')
    if fig_path:
        plt.savefig(fig_path)
        plt.clf()
    else:
        plt.show()


def ceiling_division(n, d):
    return -(n // -d)

def compute_metrics(y_true, y_pred, ci95=True, string_target=False):
    if string_target:
        # Convert 'Yes' and 'No' to binary values
        y_true_bin = [1 if val == 'Yes' else 0 for val in y_true]
        y_pred_bin = [1 if val == 'Yes' else 0 for val in y_pred]
    else:
        y_true_bin = y_true
        y_pred_bin = y_pred
    # Define metric functions
    def specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (1, 1):  # Only one class
            return float(cm[0][0] == len(y_true))
        TN, FP, _, _ = cm.ravel()
        return TN / (TN + FP)

    def negative_predictive_value(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (1, 1):  # Only one class
            return float(cm[0][0] == len(y_true))
        TN, _, FN, _ = cm.ravel()
        return TN / (TN + FN)

    metrics_functions = {
        'PPV': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score,
        'specificity': specificity,
        'NPV': negative_predictive_value,
        'f1_score': f1_score,
        'accuracy': accuracy_score,

    }

    results = {}

    if ci95:
        for metric, func in metrics_functions.items():
            value, lower, upper = bootstrap_ci(y_true_bin, y_pred_bin, func)
            results[metric] = {
                'value': round(value*100,2),
                '95% CI low': round(lower*100,2), 
                '95% CI high':round(upper*100,2)
            }
        return results
    else:
        for metric, func in metrics_functions.items():
            value = func(y_true_bin, y_pred_bin)
            results[metric] = round(value*100, 2)

            results['roc_auc_score'] = round(roc_auc_score(y_true_bin, y_pred_bin)*100, 2)

        return results

def bootstrap_ci(y_true, y_pred, metric_function, n_iterations=1000, alpha=0.05):
    bootstrap_samples = []
    
    for _ in range(n_iterations):
        boot_true, boot_pred = resample(y_true, y_pred)
        
        # Check if both classes are present
        if len(set(boot_true)) > 1 and len(set(boot_pred)) > 1:
            bootstrap_samples.append(metric_function(boot_true, boot_pred))
    
    if not bootstrap_samples:
        return (np.nan, np.nan)
    
    lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    score = metric_function(y_true, y_pred)
    return score, lower, upper

def emb_name(emb_name):
    name_in_path = emb_name.replace("/", "_")
    name_in_path = name_in_path.replace("_", "_") # why i need this ?
    return name_in_path

def calculate_stratifiedKfold_res(X, y, emb_name, clf):
    ss = StandardScaler()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_cv_res = []
    sm = SMOTE(random_state=42)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # print(f"Fold {i}:")
        test_count = Counter([y[i] for i in test_index])

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_norm = ss.fit_transform(X_train)
        X_res, y_res = sm.fit_resample(X_norm, y_train)
        clf.fit(X_res, y_res)
        X_test_norm = ss.transform(X_test)
        y_pred = clf.predict(X_test_norm)
        res = compute_metrics(y_test, y_pred, False)
        models_cv_res.append(np.asarray(list(res.values())))

    r = pd.DataFrame(np.vstack(models_cv_res).mean(axis=0)).T
    r.columns = list(res.keys())
    r['model_name'] = emb_name
    return r

def optimize(params, param_names, X, y):
    # here we optimize by f1 score
    params = dict(zip(param_names, params))
    model = LogisticRegression(**params)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    ss = StandardScaler()
    sm = SMOTE(random_state=88)

    for train_index, test_index in kf.split(X=X, y=y,):

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        X_norm = ss.fit_transform(X_train)
        X_res, y_res = sm.fit_resample(X_norm, y_train)

        model.fit(X_res, y_res)

        X_test_norm = ss.transform(X_test)
        preds = model.predict(X_test_norm)
        fold_f1_score = f1_score(y_test, preds)
        f1_scores.append(fold_f1_score)
    
    return -1.0*np.mean(f1_scores)


if __name__ == '__main__':
    log_dir1 = r"E:\Project\pCR_paper_code\results\experiment_3\runs\Loss_train_loss.csv"
    log_dir2 = r"E:\Project\pCR_paper_code\results\experiment_3\runs\Loss_val_loss.csv"
    ea = get_tensorboard_data(log_dir1, log_dir2)
    print(1)