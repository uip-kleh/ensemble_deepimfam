from new_deepimfam.component import DeepImFam

if __name__ == "__main__":
    deepimfam = DeepImFam(config_path="new_deepimfam/config.yaml")
    deepimfam.train()
