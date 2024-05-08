from new_deepimfam.component import DeepImFam

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    deepimfam = DeepImFam(config_path)

    deepimfam.train()
    deepimfam.draw_history()
    deepimfam.draw_cm()

    