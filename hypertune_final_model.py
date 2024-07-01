import json
from ensemble_using_index import Split3Ensemble

class HyperSplit3Ensemble(Split3Ensemble):
    def __init__(self, top, columns, config_path) -> None:
        super().__init__(top, columns, config_path)

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"

    with open("/home/mizuno/data/mizuta_NORM_INDEX/results.json", "r") as f:
        reuslts = json.load(f)

    num_entry = 32
    results = reuslts[:num_entry]

    print(reuslts[:num_entry])
    columns = [entry + '-' + str(i) for acc, entry in results for i in range(5)] + ["labels"]

    ensemble = HyperSplit3Ensemble(top=num_entry, columns=columns, config_path=config_path)
    ensemble.load_data(is_predict=False)