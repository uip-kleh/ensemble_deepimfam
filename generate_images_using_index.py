import os
import numpy as np
from new_deepimfam.component import ImageGenerator

class ImageGeneratorUsingIndex(ImageGenerator):
    def __init__(self, config_path, index) -> None:
        super().__init__(config_path, index)
        
    def generate_vector_using_index(self):
        self.load_aaindex1()
        keys = self.aaindex1[self.index].keys()
        values = np.array(list(self.aaindex1[self.index].values()))        
        values_std = self.standarize(values)
        vectors = {}
        for i, key in enumerate(keys):
            vectors[key] = [values_std[i], 0.1]
        return vectors
        
    # CALCURATE COORDINATE
    def calc_coordinate(self):
        vectors = self.generate_vector_using_index()
        sequences = self.read_sequences(self.amino_train_path) + self.read_sequences(self.amino_test_path)

        for i, seq in enumerate(sequences):
            fname = os.path.join(self.coordinates_directory, str(i) + ".dat")
            with open(fname, "w") as f:
                x, y = 0, 0
                print("{}, {}".format(x, y), file=f)
                for aa in seq:
                    if not aa in vectors:
                        continue
                    x += vectors[aa][0]
                    y += vectors[aa][1]
                    print("{}, {}".format(x, y), file=f)
        
if __name__ == "__main__":

    entry_path = "/home/mizuno/data/aaindex_entry.txt"
    config_path = "new_deepimfam/config.yaml"
    with open(entry_path, "r") as f:
        for entry in f.read().splitlines():
            print(entry)

            image_gen = ImageGeneratorUsingIndex(config_path, entry)
            image_gen.calc_coordinate()
            
            image_gen.make_images_info()

            image_gen.generate_images()
            image_gen.convert_pgm()

    
    