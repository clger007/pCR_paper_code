from typing import Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from utils import bootstrap_ci, plot_curve_with_ci
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import pandas as pd
from collections import defaultdict
import pathlib
import numpy as np
import logging
from tqdm import tqdm
import re

class KeyWord:
    def __init__(self, list_of_text: list, stop_word_list: list | None) -> None:
        stop_words = 'english' if not stop_word_list else stop_word_list
        self.data = list_of_text
        self.bow = CountVectorizer(max_df=.8, stop_words=stop_words, max_features=5000)
        self.model = TfidfTransformer()
        logging.debug('fitting bag-of-word model!')
        self.bow_vec = self.bow.fit_transform(self.data)
        self.feature_names = self.bow.get_feature_names_out()

    def __call__(self, top_n: int, model_type: str) -> Any:
        """
        model type would be two types:
        1) bow
        2) bow + tfidf
        """
        if model_type not in ['bow', 'tfidf']:
            raise ValueError("model_type should be 'bow' or 'tfidf'")
        
        if model_type == 'tfidf':
            logging.debug('fitting tf-idf model!')
            self.tfidf_vec = self.model.fit_transform(self.bow_vec)
            sorted_items = self.sort_coo(self.tfidf_vec.tocoo())
            return self.extract_top_n(sorted_items, top_n)
        elif model_type == 'bow':
            sorted_items = self.sort_coo(self.bow_vec.tocoo())
            return self.extract_top_n(sorted_items, top_n)

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_top_n(self, sorted_items, topn: int):
        feature_names = self.feature_names
        # Use the passed sorted_items directly instead of calling sort_coo again
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
        
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
        return results

class PlotExp2:
    def __init__(self, clf, X_test, test_y, emb_name) -> None:
        self.y_pred_proba = clf.predict_proba(X_test)
        self.test_y = test_y
        self.X_test = X_test
        self.emb_name = emb_name

    def auprc(self, n_iter=1_000):
        recall_grid = np.linspace(0, 1, 100) 
        all_precisions = []
        n_iter = n_iter
        np.random.seed(300)

        for _ in tqdm(range(n_iter)):
            indices = np.random.choice(len(self.test_y), len(self.test_y), replace=True)
            sample_true = self.test_y[indices]
            sample_proba = self.y_pred_proba[:, 1][indices]

            precision, recall, _ = precision_recall_curve(sample_true, sample_proba)
            sorted_indices = np.argsort(recall)
            sorted_recall = recall[sorted_indices]
            sorted_precision = precision[sorted_indices]

            # If the smallest value in your recall_grid is less than min(sorted_recall),
            # or the largest value is more than max(sorted_recall), set the fill_value accordingly.
            interpolated_precision = interp1d(
                sorted_recall, sorted_precision,
                kind='linear',
                fill_value=(sorted_precision[0], sorted_precision[-1]),  # Use boundary values for extrapolation
                bounds_error=False
            )
            all_precisions.append(interpolated_precision(recall_grid))

        lower_precision = np.percentile(all_precisions, 2.5, axis=0)
        upper_precision = np.percentile(all_precisions, 97.5, axis=0)
        mean_precision = np.mean(all_precisions, axis=0)

        # Calculate the AUPRC
        mean_auc = auc(recall_grid, mean_precision)

        # Plotting
        res = {
            'X': recall_grid,
            'y_mean': mean_precision,
            'label':f'Mean PRC (AUC = {mean_auc:.2f})',
            'y_low': lower_precision,
            'y_high': upper_precision,
            'x_label':'Precision',
            'y_label':'Recall',
            'fig_title':f'Bootstrapped Precision-Recall curve: {self.emb_name}',
        }
        return res
    
    
    def auroc(self, n_iter):

        np.random.seed(300)
        all_auroc = []
        all_tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for _ in tqdm(range(n_iter), desc='boostrapping'):
            indices = np.random.choice(len(self.test_y), len(self.test_y), replace=True)
            sample_true = self.test_y[indices]
            sample_proba = self.y_pred_proba[:, 1][indices]

            fpr, tpr, _ = roc_curve(sample_true, sample_proba)
            roc_auc = auc(fpr, tpr)
            all_auroc.append(roc_auc)

            # Interpolate the TPR at the fixed set of FPR values
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            all_tprs.append(tpr_interp)

        # Compute the mean TPR and CI bounds
        mean_tpr = np.mean(all_tprs, axis=0)
        tpr_lower = np.percentile(all_tprs, 2.5, axis=0)
        tpr_upper = np.percentile(all_tprs, 97.5, axis=0)

        mean_auroc = np.mean(all_auroc)
        auroc_lower = np.percentile(all_auroc, 2.5)
        auroc_upper = np.percentile(all_auroc, 97.5)

        res = {
            'X': base_fpr,
            'y_mean': mean_tpr,
            'y_low': tpr_lower,
            'y_high': tpr_upper,
            'label': f'ROC curve area = {mean_auroc:.2f} [95% CI: {auroc_lower:.2f}-{auroc_upper:.2f}]',
            'x_label': 'False Positive Rate',
            'y_label': 'True Positive Rate',
            'fig_title': f'ROC 95% CI: {self.emb_name}',
        }

        return res


def plot_training_curve(train_res_dir, 
                        val_res_dir, 
                        train_label, 
                        val_label, 
                        title, 
                        ylabel, 
                        save_fig=False, 
                        dpi=300, 
                        save_dir=None,
                        smooth=True,
                        sm_weight=0.5,
                        swap=True,
                        ):
      
    """
    Plot training and validation curves from CSV data.

        This function reads the training and validation results from CSV files,
        plots the raw and optionally smoothed curves, and saves or displays the figure.

        Parameters:
        - train_res_dir (str): Path to the CSV file containing the training results.
        - val_res_dir (str): Path to the CSV file containing the validation results.
        - train_label (str): Label for the training curve in the legend.
        - val_label (str): Label for the validation curve in the legend.
        - title (str): Title of the plot.
        - ylabel (str): Label for the Y-axis.
        - save_fig (bool): If True, save the figure to a file instead of displaying.
        - dpi (int): Dots per inch (resolution) for the saved figure.
        - save_dir (str): Directory path where the figure should be saved. If None and save_fig is True,
                        a default path is used.
        - smooth (bool): If True, apply smoothing to the curves.
        - sm_weight (float): The smoothing weight to be used if smooth is True.
        - swap (bool): If True, swap the training and validation data. Useful if the labels were
                    accidentally swapped during data recording.

        The function logs the start and end of the plotting process. The raw data is plotted with a
        lower alpha for transparency. If smoothing is applied, the smoothed curves are plotted over
        the raw data. The function supports saving the figure in high resolution or displaying it
        interactively.
    """
      
    logging.info("start plotting the fig")
    train_df = pd.read_csv(train_res_dir)[1:]
    val_df = pd.read_csv(val_res_dir)[1:]
    if swap: 
        train_df, val_df = val_df, train_df
    plt.plot(train_df['Step'], train_df['Value'], linewidth=2, color='skyblue', alpha=0.1)
    plt.plot(val_df['Step'], val_df['Value'], linewidth=2, color='tomato', alpha=0.1)

    if smooth:
        sm_train = EMA_smooth(train_df['Value'].to_list(), sm_weight)
        sm_val = EMA_smooth(val_df['Value'].to_list(), sm_weight)
        plt.plot(train_df['Step'], sm_train, label=train_label, linewidth=1, color='blue')
        plt.plot(val_df['Step'], sm_val, label=val_label, linewidth=1, color='red')


    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linewidth=0.3)
    if save_fig:
        out_dir = r"E:\Project\pCR_paper_code\results\experiment_3\runs" + f"\{title}.png" if not save_dir else save_dir
        plt.savefig(out_dir, dpi=dpi)
        plt.cla()
    else:
        plt.show()
# Remember to configure logging, if not already done in the rest of your application
# logging.basicConfig(level=logging.DEBUG)

def EMA_smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_exp3(output_dir):
    p = pathlib.Path(r'E:\Project\pCR_paper_code\results\experiment_3')
    metric_mapper = {
        'Recall': 'Sensitivity',
        'Precision': 'PPV',
        'f1': 'F1 Score',
        'accuracy': "Accuracy"
    }
    metrics = defaultdict(list)
    for file_path in p.glob("*.csv"):
        name = file_path.stem
        r = re.match(r"^[^_]+", name)
        
        metric_name = r.group(0)
        if metric_name in metric_mapper:
            metric_name = metric_mapper[metric_name]
        metrics[metric_name].append(file_path)
    
    for metric, file_path in metrics.items():
        train_res_dir = file_path[0] if 'train' in file_path[0].stem else file_path[1]
        val_res_dir = file_path[1] if 'val' in file_path[1].stem else file_path[0]
        output_name = f"\\{metric}_figure.png"
        plot_args = {
            'train_res_dir': train_res_dir, 
            'val_res_dir': val_res_dir, 
            'train_label': f"Training {metric}", 
            'val_label': f"Validation {metric}", 
            'title': f"{metric}: Training vs. Validation", 
            'ylabel': metric, 
            'save_fig':True, 
            'dpi':400, 
            'save_dir':output_dir + output_name,
            'sm_weight':0.55
        }

        plot_training_curve(**plot_args)



