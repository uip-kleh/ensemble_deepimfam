import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from new_deepimfam.component import Ensemble

class AverageEnsemble(Ensemble):
    def __init__(self, config_path) -> None:
        super().__init__(config_path)
        
    def predict(self):
        # LOAD PREDIT DAT
        train_predict, test_predict = self.load_data(is_predict=True)

        # LOAD PROBA DATA
        # train_proba, test_proba = self.load_data(is_predict=False)
        
        # DROP LABELS
        train_predict.drop("ISOY800101-SNEP660103", axis=1)
        test_predict.drop("ISOY800101-SNEP660103", axis=1)
        

        # SPLIT LABELS
        train_predict, train_labels = self.split_labels(train_predict)
        test_predict, test_labels = self.split_labels(test_predict)
        # train_proba, _ = self.split_labels(train_proba)
        # test_proba, _ = self.split_labels(test_proba)

        train_frequency = []
        for pred in train_predict.iloc[:].values:
            cnt = self.count_predict(pred)
            train_frequency.append(np.argmax(np.array(cnt)))
        print("train-accuracy:", accuracy_score(train_labels, train_frequency))
        print("train-f1-score:", f1_score(train_labels, train_frequency, average="macro"))

        test_frequency = []
        for pred in test_predict.iloc[:].values:
            cnt = self.count_predict(pred)
            test_frequency.append(np.argmax(np.array(cnt)))
        print("test-accuracy:", accuracy_score(test_labels, test_frequency))
        print("test-f1-score:", f1_score(test_labels, test_frequency, average="macro"))

            
    def count_predict(self, pred):
        cnt = [0 for i in range(5)]
        for i in pred: 
            cnt[i] += 1
        return cnt
        

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    ensemble = AverageEnsemble(config_path=config_path)
    ensemble.predict()