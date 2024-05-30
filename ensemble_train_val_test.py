import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt  
import json
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from new_deepimfam.component import Ensemble, Draw

class Split3Ensemble(Ensemble):
    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        
    def predict(self):
        train_df, val_df, test_df = self.load_data(is_predict=False)
        
        print(val_df.head())
        
        sampler = RandomOverSampler(random_state=42)
        val_df, _ = sampler.fit_resample(val_df, val_df["labels"])

        train_df, train_labels = self.split_labels(train_df)
        val_df, val_labels = self.split_labels(val_df)
        test_df, test_labels = self.split_labels(test_df)
        
        # XGBoost
        model = XGBClassifier(
            n_estimators=1000,
            # early_stopping_rounds=15,
            )
        model.fit(
            val_df, val_labels,
            eval_set=[(val_df, val_labels), (test_df, test_labels)],
            verbose=True,
        )

        # Random Forest
        # model = RandomForestClassifier(
        #     n_estimators=1000,
        #     verbose=True,
        #     )
        
        # model.fit(
        #     val_df, val_labels,
        #     )

        val_pred = model.predict(val_df)
        test_pred = model.predict(test_df)

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        # DRAW IMPORTANCE
        draw = Draw()
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), np.array(val_df.columns)[sorted_idx])
        plt.title("Feature Importance")
        fname = os.path.join(self.results, "importance.pdf")
        draw.save_figure_as_pdf(fname)

        # print(feature_importance)

        accuracy_val = accuracy_score(val_labels, val_pred)
        accuracy_test = accuracy_score(test_labels, test_pred)
        f1_val = f1_score(val_labels, val_pred, average="macro")
        f1_test = f1_score(test_labels, test_pred, average="macro")
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
        fname = os.path.join(self.results, "cm.pdf")
        cm = confusion_matrix(test_labels, test_pred)
        draw.draw_cm(cm, fname)
        fname = os.path.join(self.results, "normed_cm.pdf")
        normed_cm = confusion_matrix(test_labels, test_pred, normalize="true")
        draw.draw_cm(normed_cm, fname, norm=True)

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
        val_df = pd.read_csv(val_fname)
        test_df = pd.read_csv(test_fname)
        return train_df, val_df, test_df        

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    ensemble = Split3Ensemble(config_path=config_path)
    ensemble.predict()