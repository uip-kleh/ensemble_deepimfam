import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt  
import json
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from new_deepimfam.component import Ensemble, Draw, AAindex1

class Split3Ensemble(Ensemble):
    def __init__(self, top, columns, config_path) -> None:
        self.top = top
        self.columns = columns
        super().__init__(config_path)
        
    def predict(self, is_xgboost=True):
        train_df, val_df, test_df = self.load_data(is_predict=False)
        
        print(val_df.head())
        
        sampler = RandomOverSampler(random_state=42)
        sampled_val_df, _ = sampler.fit_resample(val_df, val_df["labels"])

        train_df, train_labels = self.split_labels(train_df)
        sampled_val_df, sampled_val_labels = self.split_labels(val_df)
        val_df, val_labels = self.split_labels(val_df)
        test_df, test_labels = self.split_labels(test_df)
        
        # XGBoost
        if is_xgboost:
            model = XGBClassifier(
                n_estimators=1000,
                # early_stopping_rounds=15,
                )
            model.fit(
                sampled_val_df, sampled_val_labels,
                eval_set=[(val_df, val_labels), (test_df, test_labels)],
                verbose=True,
            )
        else:
        # Random Forest
            model = RandomForestClassifier(
                n_estimators=1000,
                verbose=True,
                )
            
            model.fit(
                sampled_val_df, sampled_val_labels,
                )

        train_pred = model.predict(train_df)
        val_pred = model.predict(val_df)
        test_pred = model.predict(test_df)

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        
        # MAKE PATH
        self.results = os.path.join(self.results, "TOP" + str(i))
        self.make_directory(self.results)
        if is_xgboost:
            self.results = os.path.join(self.results, "xgboost")
        else:
            self.results = os.path.join(self.results, "randomforest")
        self.make_directory(self.results)
        
        # DRAW IMPORTANCE
        draw = Draw()
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), np.array(val_df.columns)[sorted_idx])
        plt.title("Feature Importance")
        fname = os.path.join(self.results, "importance.pdf")
        draw.save_figure_as_pdf(fname)

        # print(feature_importance)

        accuracy_train = accuracy_score(train_labels, train_pred)
        accuracy_val = accuracy_score(val_labels, val_pred)
        accuracy_test = accuracy_score(test_labels, test_pred)
        f1_train = f1_score(train_labels, train_pred, average="macro")
        f1_val = f1_score(val_labels, val_pred, average="macro")
        f1_test = f1_score(test_labels, test_pred, average="macro")
        print("accuracy(train):", accuracy_train)
        print("accuracy(val): ", accuracy_val)
        print("accuracy(test): ", accuracy_test)
        

        fname = os.path.join(self.results, "metrics.json")
        with open(fname, "w") as f:
            json.dump({
                "accuracy_val": accuracy_val,
                "accuracy_test": accuracy_test,
                "f1_val": f1_val,
                "f1_test": f1_test,
            }, f, indent=2)
        
        # Draw Confusion Matrix
        fname = os.path.join(self.results, "cm_val.pdf")
        cm = confusion_matrix(val_labels, val_pred)
        draw.draw_cm(cm, fname)
        fname = os.path.join(self.results, "normed_cm_val.pdf")
        normed_cm = confusion_matrix(val_labels, val_pred, normalize="true")
        draw.draw_cm(normed_cm, fname, norm=True)

        
        fname = os.path.join(self.results, "cm.pdf")
        cm = confusion_matrix(test_labels, test_pred)
        draw.draw_cm(cm, fname)
        fname = os.path.join(self.results, "normed_cm.pdf")
        normed_cm = confusion_matrix(test_labels, test_pred, normalize="true")
        draw.draw_cm(normed_cm, fname, norm=True)

    def average(self):
        train_df, val_df, test_df = self.load_data(is_predict=True)
        
        test_predictions, test_labels = self.split_labels(test_df)
        
        freq = []
        for pred in test_predictions.iloc[:].values:
            cnt = self.count_predict(pred)
            freq.append(np.argmax(np.array(cnt)))

        acc = accuracy_score(test_labels, freq)
        f1 = f1_score(test_labels, freq, average="macro")
        print("THE NUMBER OF COMBINATIONS", self.top)
        print("accuracy:", acc)
        print("f1:", f1)
        return acc, f1
            
    def count_predict(self, pred):
        cnt = [0 for i in range(5)]
        for i in pred: 
            cnt[i] += 1
        return cnt

    def load_data(self, is_predict=True):
        if is_predict:
            train_fname = os.path.join(self.results, "train_predict_weighted.csv")
            val_fname = os.path.join(self.results, "val_predict_weighted.csv")
            test_fname = os.path.join(self.results, "test_predict_weighted.csv")
        else:
            train_fname = os.path.join(self.results, "train_proba_weighted.csv")
            val_fname = os.path.join(self.results, "val_proba_weighted.csv")
            test_fname = os.path.join(self.results, "test_proba_weighted.csv")
        train_df = pd.read_csv(train_fname)
        print(train_df.columns)
        train_df = pd.read_csv(train_fname)[self.columns]
        val_df = pd.read_csv(val_fname)[self.columns]
        test_df = pd.read_csv(test_fname)[self.columns]
        return train_df, val_df, test_df        

if __name__ == "__main__":
    # corr_path = "/home/mizuno/data/results/corr.json"
    # with open(corr_path, "r") as f:
    #     corr = json.load(f)

    # N = 12
    # columns = []
    # results = []
    # for i in range(1, 12):
    #     # USING PROBA
    #     column = "-".join(corr[i][1:])        
    #     columns.append(column)
    #     # print(column)
    #     # proba_columns = [column + "-" + str(i) for column in columns for i in range(5)]
    #     proba_columns = columns + ["labels"]


    #     config_path = "new_deepimfam/config.yaml"   
    #     # ensemble = Split3Ensemble(config_path=config_path, top=i, columns=proba_columns)
    #     # ensemble.predict(is_xgboost=True)
    #     # ensemble = Split3Ensemble(config_path=config_path, top=i, columns=proba_columns)
    #     # ensemble.predict(is_xgboost=False)
        
    #     ensemble = Split3Ensemble(config_path=config_path, top=i, columns=proba_columns)
    #     results.append(ensemble.average())

    # print(results)
    
    results = [
        [0.9007874015748032, "CHAM830102-MEEJ810101"],
        [0.9590551181102362, "DESM900102-TSAJ990101"],
        [0.9433070866141732, "FUKS010112-QIAN880132"],
        [0.8984251968503937, "GEOR030106-PRAM900104"],
        [0.9385826771653544, "HARY940101-NAKH900112"],
        [0.9236220472440945, "ISOY800101-SNEP660103"],
        [0.9527559055118110, "MANP780101-WOLS870103"],
        [0.9448818897637795, "NAKH900107-PALJ810108"],
        [0.9496062992125984, "NAKH900107-PARS000102"],
        [0.9574803149606299, "QIAN880134-YUTK870104"],
        [0.9256926774978638, "SNEP660103-SNEP660104"],
    ]
    
    results = sorted(results)
    results.reverse()
    print(results)
    columns = []
    for i in range(len(results)-1, len(results)):
        columns.append(results[i][-1])
        print(columns)
        proba_columns = [column + "-" + str(i) for column in columns for i in range(5)] + ["labels"]
        config_path = "new_deepimfam/config.yaml"   
        ensemble = Split3Ensemble(config_path=config_path, top=i+1, columns=proba_columns)
        ensemble.predict(is_xgboost=True)
        ensemble = Split3Ensemble(config_path=config_path, top=i+1, columns=proba_columns)
        ensemble.predict(is_xgboost=False)