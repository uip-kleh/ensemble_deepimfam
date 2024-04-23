from new_deepimfam.component import ImageGenerator

if __name__ == "__main__":
    image_gen = ImageGenerator(config_path="new_deepimfam/config.yaml")
    vectors = image_gen.generate_normed_verctors()
    image_gen.draw_vectors(vectors)