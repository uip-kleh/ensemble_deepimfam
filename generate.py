from new_deepimfam.component import ImageGenerator

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    image_gen = ImageGenerator(config_path)
    image_gen.calc_coordinate()
    
    vectors = image_gen.generate_normed_verctors()
    # vectors = image_gen.generate_std_vectors()
    image_gen.make_images_info()

    image_gen.generate_images()
    image_gen.convert_pgm()
