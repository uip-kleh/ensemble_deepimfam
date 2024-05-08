from new_deepimfam.component import ImageGenerator

if __name__ == "__main__":
    image_gen = ImageGenerator(config_path="../new_deepimfam/config.yaml")
    image_gen.calc_coordinate()