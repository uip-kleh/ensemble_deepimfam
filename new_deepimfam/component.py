import os
import yaml
import json
import numpy as np

class Common:
    def __init__(self, config_path) -> None:
        self.set_config(config_path)

    def set_config(self, config_path):
        # TODO: THERE MAY BE BETTER WAY
        with open(config_path, "r") as f: 
            args = yaml.safe_load(f)
            self.data_direcotry = self.join_home(args["data_directory"], True)
            self.results_directory = self.join_home(args["results_directory"], True)
            self.aaindex1_path = self.join_home(args["aaindex1_path"])

    def join_home(self, fname, is_dir=False):
        fname = os.path.join(os.environ["HOME"], fname)
        if not os.path.exists(fname) and is_dir: 
            os.mkdir(fname)
        return fname
    
    def save_obj(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)

# CALC STD
class AAindex1(Common):
    def __init__(self, config_path) -> None:
        Common.__init__(self, config_path)
        self.load()

    # LOAD AAINDEX1 FROM JSON
    def load(self):
        with open(self.aaindex1_path, "r") as f:
            self.aaindex1 = json.load(f)

    def calc(self):
        results = []
        
        # ELIMINATE INDEX WHICH HAS SAME VALUE
        keys = []
        for key, values in self.aaindex1.items():
            if self.has_same_value(np.array(list(values.values()))): 
                continue
            keys.append(key)

        N = len(keys)
        print("REST OF INDEX IS ", N)
        keys.sort()

        for i in range(N):
            for j in range(i+1, N):
                key1, key2 = keys[i], keys[j]

                # TODO: THERE MAY BE BETTER WAY
                values1 = np.array(list(self.aaindex1[key1].values()))
                values2 = np.array(list(self.aaindex1[key2].values()))

                corr = np.corrcoef(values1, values2)
                results.append([abs(corr[0][1]), key1, key2])
        
        results.sort()
        fname = os.path.join(self.results_directory, "corr.json")
        self.save_obj(results, fname)

    def has_same_value(self, values):
        print(values, not values.size == np.unique(values).size)
        return not values.size == np.unique(values).size

    def disp(self, key1, key2):
        print(self.aaindex1[key1])
        print(self.aaindex1[key2])

if __name__ == "__main__":
    aaindex1 = AAindex1(config_path="config.yaml")    
    # aaindex1.calc()
    aaindex1.disp("NAKH900107", "PALJ810108")