import os
import sys
sys.path.append(os.pardir)
from sklearn.metrics import confusion_matrix
from new_deepimfam.component import *
from collections import defaultdict

class TestDeepImFam(DeepImFam):
    def __init__(self, config_path) -> None:
        super().__init__(config_path)

if __name__ == "__main__":
    test1 = TestDeepImFam(config_path="config1.yaml")
    test_labels, pred1_labels = test1.predict()  

    test2 = TestDeepImFam(config_path="config2.yaml")
    test_labels, pred2_labels = test2.predict()

    cm = confusion_matrix(pred1_labels, pred2_labels)
    
    draw = Draw()
    fname = os.path.join(test1.results, "test_compare_pred.pdf")
    draw.draw_cm(cm, fname)

    compare = defaultdict(list)
    n = len(test_labels)
    for i in range(n):
        key = tuple([
            test_labels[i] == pred1_labels[i], 
            pred1_labels[i] == pred2_labels[i],
            pred2_labels[i] == test_labels[i],
            ])
        
        if key == (True, True, True): continue
        compare[key].append([test_labels[i], pred1_labels[i], pred2_labels[i]])

    for key, val in compare.items():
        print("key: ", key)
        for c in val:
            print("\t", c)