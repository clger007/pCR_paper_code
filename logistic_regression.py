from collections import Counter
from datetime import datetime
from functools import partial
from imblearn.over_sampling import SMOTE
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt import space
import pathlib
import os
import joblib
import re
import yaml
from utils import compute_metrics, emb_name, calculate_stratifiedKfold_res, optimize, plot_curve_with_ci
from data_visulization import PlotExp2

class LeaderBoard():

    def __init__(self, data_path, working_dir, param_dir, lr_dir, embeddings=[], target='label', optmize=False,) -> None:
        df = pd.read_parquet(data_path)
        embeddings = [emb_name(name_with_slash) for name_with_slash in list(df.columns)[2:]]
        df.columns = ['text', 'label'] + embeddings
        
        current = datetime.now()
        self.date_string = current.strftime("%m_%d")
        self.data = df
        self.working_dir = working_dir
        self.embeddings = embeddings if embeddings else self.data[self.data.columns[2:]]
        self.target = target
        self.train, self.test = self.train_test_split()
        self.param_dir = pathlib.Path(pathlib.PurePath(working_dir,param_dir))
        self.lr_dir = pathlib.Path(pathlib.PurePath(working_dir, lr_dir))
        self.models = {}
        self.model_init(optimize)
        

    def train_test_split(self):
        train, test = train_test_split(
            self.data, 
            test_size=0.2,
            random_state=3, 
            shuffle=True, 
            stratify=self.data[self.target]
            )
        return train, test

    def model_init(self, optimize=False):
        # params = {'solver': 'lbfgs', 'tol': 1e-07, 'C': 44001.8663722268, 'max_iter': 1926} 
     
        """
        In this condition, we don't have trained CLF for EMBDs, so we need to train them with the selected params.
        The method is only for the experiment purpose few parts are not ready for the production:
        1. If there are missing params for the embds, it will do no noting just record it into log
        2. If there's no params at all it will not invoke the self.optimize model to get the optimal params.
        """
      
        lr_checklist = []
        if self.lr_dir.exists():
            for clf_path in self.lr_dir.iterdir():
                if clf_path.stem[:-4] in self.embeddings:
                    self.models[clf_path.stem[:-4]] = joblib.load(clf_path)
                else:
                    logging.error(f"{clf_path} is incorrect!")
        else:
            logging.info("There's no existing models! Looking for params.")

            param_path = pathlib.Path(self.param_dir)
            check_list = []
            if param_path.exists():
                logging.info("Params file is found, now setting models!")

                for param_file in param_path.iterdir():
                    params = joblib.load(param_file)
                    for emb_name, param in params.items():
                        if emb_name in self.embeddings:
                            self.models[emb_name] = LogisticRegression(**param) # Here to load the selected parameters.
                            
                            logging.debug(f"optimized CLF for {emb_name} has been intialized!")
                            
                            try:
                                # save model at desired destination
                                target_dir = pathlib.Path(pathlib.PurePath(self.working_dir, 
                                                                            f"results/intermediate/models/{self.date_string}_{emb_name}_clf.joblib"))
                                joblib.dump(self.models[emb_name], target_dir)
                                logging.debug(f"Model succesfully saved at results/intermediate/models folder!")
                            except (FileNotFoundError, PermissionError) as e:
                                logging.error(f"File error: {str(e)}")

                        else:
                            check_list.append(emb_name)
            else:
                logging.info("There's no existing models and params! Looking for params.")

                if optimize:
                    self.optimize_model_embds()
                    self.model_init(optimize=False)
                else:
                    for emb_name in self.embeddings:
                        self.models[emb_name] = LogisticRegression(max_iter=2000) # The params are not selected ...

            if check_list:
                logging.debug(f"Following embedding does not have an optimized model for classification: {check_list}")
            else:
                logging.debug("All embeddings have optimized classifier!")
 
    def optimize_model_embds(self, target_dir=None):
        """
        This is just for logistic regression. For other classifier, you may extend this method.
        """
        model_params = {}

        param_space = [
            space.Real(1e-7, 1e-3, name='tol'),
            space.Real(1e-6, 1e6, name='C'),
            space.Integer(100, 3000, name='max_iter'),
            space.Categorical(['l2', None], name='penalty'),
        ]

        param_names = [
            "tol", "C", "max_iter","penalty",
        ]

        for emb in self.embeddings:
            X = np.vstack(self.data[emb].to_list())
            ss = StandardScaler()
            y = self.data[self.target].to_list()
            y = np.asarray([1 if a =='Yes' else 0 for a in y ] )
            
            optimization_function = partial(
                        optimize, 
                        param_names=param_names,
                        X=X,
                        y=y
                        )
            logging.debug(f"start searching parameters for LR wiht {emb} embeddings!")

            result = gp_minimize(
                optimization_function, 
                dimensions=param_space,
                verbose=0
            )
            try:
                params_res = {x:y for x, y in zip(param_names, result.x)}
                model_params[emb] = params_res
                logging.debug(f"obtained the param for {emb}: {params_res}")
            except:
                logging.debug("param reformatting failed, please debug and check the reason!")
                logging.debug("The result is stored in 'model param'!")
                model_params[emb] = result
        try:
            target_dir = f"results/intermediate/model_params/{self.date_string}_params.joblib" if not target_dir else target_dir
            joined_path = pathlib.PurePath(self.working_dir, target_dir)
            joblib.dump(model_params, pathlib.Path(joined_path))
        except:
            logging.debug(f"param saved failed due to the undefined path, the params are saved on temporary location {pathlib.Path().resolve()}")
            joblib.dump(model_params, f'temp_{emb}_{self.date_string}.joblib')

        logging.debug("The parameter selection is finished!")

    def make_80_20_eval(self, res_dir):
        
        res = {}
        for emb in self.embeddings:
            logging.debug(f"Now makeing prediction with : {emb} embeddings!")
            model = self.models[emb]
            X_train, X_test, y_train, y_test = self.get_80_20_data(emb)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[emb] = {
                'actual': y_test,
                'predicted': y_pred,
            }
        output_path = pathlib.Path(pathlib.PurePath(self.working_dir, 'results_80_20_{self.date_string}','prediction.joblib'))
        joblib.dump(res, output_path)
        logging.debug(f"All result is successfully generated, and stored under: {res_dir}")

    def get_80_20_data(self, embedding_name, over_sample=True):
        
        X_train = np.vstack(self.train[embedding_name].to_list())
        X_test = np.vstack(self.test[embedding_name].to_list())
        y_train = self.train[self.target].to_list()
        y_train = np.asarray([1 if a =='Yes' else 0 for a in y_train])
        y_test = self.test[self.target].to_list()
        y_test = np.asarray([1 if a =='Yes' else 0 for a in y_test])
        if over_sample:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def get_cv5_leaderboard(self):
        results = []
        for emb in self.embeddings:
            X_train, X_test, y_train, y_test = self.get_80_20_data(emb, True)
            res_df = calculate_stratifiedKfold_res(X_train, y_train, emb, self.models[emb])
            results.append(res_df)

        cv5_res_82_split = pd.concat(results, axis=0)
        self.leaderboard = cv5_res_82_split.sort_values(['f1_score'], ascending=False).reset_index()
        self.leaderboard.to_csv(pathlib.Path(pathlib.PurePath(self.working_dir, f'results/cv5_{self.date_string}.csv')))
        logging.debug(f"The result is saved at {pathlib.PurePath(self.working_dir, f'results/cv5_{self.date_string}.csv')}")

    def get_80_20_leaderboard(self, filename=None):
        """
        This leaderboard including 95%CI with boostrapping method.
        This method will save joblib
        """
        results  = {}
        for emb in self.embeddings:
            X_train, X_test, y_train, y_test = self.get_80_20_data(emb, True)
            clf = self.models[emb]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            r = compute_metrics(y_test, y_pred, True, False)
            results[emb] = r
            # save_auroc(clf, X_test, y_test)
            # save_auprc(clf, X_test, y_test)
            plt = PlotExp2(clf, X_test, y_test, emb)
            prc_args = plt.auprc(10_000)
            prc_args['color'] = "skyblue"
            prc_args['fig_path'] = pathlib.Path(pathlib.PurePath(self.working_dir, f'results/experiment_2/prc_plot/{emb}_prc.png'))
            plot_curve_with_ci(**prc_args)

            roc_args = plt.auroc(1_000)
            roc_args['color'] = "#7DCD85" #pale green
            roc_args['fig_path'] = pathlib.Path(pathlib.PurePath(self.working_dir, f'results/experiment_2/roc_plot/{emb}_roc.png'))

            plot_curve_with_ci(**roc_args)



        self.leaderboard_95ci = results

        file_name = f"{self.date_string}_exp2_80_20_95ci.joblib" if not filename else filename

        try:
            joblib.dump(results, pathlib.Path(pathlib.PurePath(self.working_dir, f'results/{file_name}')))
            logging.debug(f"The result is successfully saved at results folder!")

        except:
            joblib.dump(results, f'temp_{self.date_string}.joblib')
            logging.debug(f"The result is temporary saved at {pathlib.PurePath(self.working_dir, file_name)}")

def setup_logging(log_dir):
    logging.basicConfig(
                        filename=log_dir,
                        filemode='w',
                        level=logging.DEBUG, 
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )

def parse_argument():
    parser = argparse.ArgumentParser(description="Arguments for running experiment 2")
    parser.add_argument('-w', '--working-dir', type=str, required=True, help='Path to the working directory.')
    parser.add_argument('-p', '--param-dir', type=str, required=True, help='Path to the parameter directory.')
    parser.add_argument('-m', '--lr-dir', type=str, required=True, help='Path to the logistic regression directory.')
    parser.add_arguemnt('-l', '--log-dir', type=str, required=True, help='Path to the experiment log.')
    parser.add_argument('-d', '--data-dir', type=str, required=True, help='Path to the data.')
           
    # Add more arguments as needed
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def main(args):
    
    lb = LeaderBoard(**args)
    # lb.optimize_model_embds()

    lb.get_80_20_leaderboard('oversampled.joblib')
    # lb.get_cv5_leaderboard()


    logging.debug("Work done!")

def get_datastr():
    return datetime.now().strftime("%m_%d")

if __name__ == '__main__':
    config = load_config('E:\Project\pCR_paper_code\exp_2_config.yaml')
    args = {
        'data_path': config.get("data-dir"),
        "working_dir": config.get('working-dir'),
        "param_dir": config.get('param-dir'),
        "lr_dir": config.get('lr-dir'),
            }
    log_dir =  args["working_dir"] + f'/results/logs/exp2_on_{get_datastr()}.log'
    setup_logging(log_dir)
    main(args)
