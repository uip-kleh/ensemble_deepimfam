from new_deepimfam.component import Ensemble

if __name__ == "__main__":
    ensemble = Ensemble(config_path="new_deepimfam/config.yaml")
    # train_df, test_df = ensemble.load_data()
    ensemble.train()