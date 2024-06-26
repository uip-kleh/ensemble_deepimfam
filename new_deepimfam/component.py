import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import tqdm 
# PACKAGES FOR MACHINE LEARNING
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
import keras
from keras_preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import CategoricalFocalCrossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# PACKAGES FOR XGBOOST
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier


class Common:
    def __init__(self, config_path, index) -> None:
        self.index = index
        self.set_config(config_path)

    def set_config(self, config_path):
        # TODO: THERE MAY BE BETTER WAY
        with open(config_path, "r") as f: 
            args = yaml.safe_load(f)
            # PATH
            self.data_direcotry = self.join_home(args["data_directory"], True)
            self.results = self.join_home(args["results_directory"], True)
            self.method = args["method"]
            self.results = self.make_directory(os.path.join(self.results, self.method))
            self.results = self.make_directory(os.path.join(self.results, args["model_method"]))
            self.aaindex1_path = self.join_home(args["aaindex1_path"])
            self.amino_train_path = self.join_home(args["amino_train_data"])
            self.amino_test_path = self.join_home(args["amino_test_data"])
            self.amino_info_path = self.join_home(args["amino_info_data"])

            # EXPERIMENT INDEX
            self.DATA_NUM = args["DATA_NUM"]
            self.method_directory = self.join_home(args["method_directory"], is_dir=True)
            self.experiment_directory = self.make_directory(os.path.join(self.method_directory, self.index))
            self.coordinates_directory = self.make_directory(os.path.join(self.experiment_directory, "coordinates"))
            self.images_directory = self.make_directory(os.path.join(self.experiment_directory, "images"))
            self.results_directory = self.make_directory(os.path.join(self.experiment_directory, "results"))
            self.results_directory = self.make_directory(os.path.join(self.results_directory, args["model_method"]))
            self.metrics_path = os.path.join(self.results_directory, "metrics.json")
            self.images_info_path = os.path.join(self.images_directory, "images_info.csv")
            self.IMAGE_SIZE = args["IMAGE_SIZE"]
            self.hierarchy_label = args["hierarchy_label"]
            self.dropout_ratio = args["DROPOUT_RATIO"]
            self.BATCH_SIZE = args["BATCH_SIZE"]

    def join_home(self, fname, is_dir=False):
        fname = os.path.join(os.environ["HOME"], fname)
        if is_dir:
            self.make_directory(fname)
        return fname
    
    def make_directory(self, fname):
        if not os.path.exists(fname):
            os.mkdir(fname)
        return fname
    
    def save_obj(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)
    
    def save_dict_as_dataframe(self, obj: dict, fname):
        pd.DataFrame(obj).to_csv(fname, index_label=False)

    def load_csv_as_dict(self, fname):
        return pd.read_csv(fname).to_dict(orient="list")

    # LOAD AAINDEX1 FROM JSON
    def load_aaindex1(self):
        with open(self.aaindex1_path, "r") as f:
            self.aaindex1 = json.load(f)

class Draw:
    def __init__(self) -> None:
        pass

    def draw_history(self, result, label, fname):
        plt.figure()
        epochs = [i for i in range(len(result[label]))]
        plt.plot(epochs, result[label], label="train")
        plt.plot(epochs, result["val_" + label], label="val_" + label)
        plt.title(label)
        plt.legend()
        plt.tight_layout()
        self.save_figure_as_pdf(fname)
    
    def draw_cm(self, cm, fname, norm=False):
        plt.figure()
        if not norm: sns.heatmap(cm, cmap="Blues", annot=True, fmt="d")
        else: sns.heatmap(cm, cmap="Blues", annot=True, fmt=".2f")
        plt.xlabel("Pred")
        plt.ylabel("GT")
        self.save_figure_as_pdf(fname)

    def save_figure_as_pdf(self, fname):
        plt.tight_layout()
        plt.savefig(fname, transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

# CALC CORR
class AAindex1(Common):
    def __init__(self, config_path, index) -> None:
        Common.__init__(self, config_path, index)
        self.load_aaindex1()

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
        fname = os.path.join(self.results, "corr.json")
        self.save_obj(results, fname)

    def has_same_value(self, values):
        print(values, not values.size == np.unique(values).size)
        return not values.size == np.unique(values).size

    def disp(self, key1, key2):
        print(self.aaindex1[key1])
        print(self.aaindex1[key2])

class ImageGenerator(Common):
    def __init__(self, config_path, index) -> None:
        Common.__init__(self, config_path, index)

    def calc_coordinate(self):
        vectors = self.generate_std_vectors()
        self.draw_vectors(vectors)
        
    # GENERATE STANDRIZED VECTOR
    def generate_std_vectors(self):
        self.load_aaindex1()
        keys = self.aaindex1[self.index].keys()
        values1 = np.array(list(self.aaindex1[self.index].values()))
        # values2 = np.array(list(self.aaindex1[self.index2].values()))
        values2 = np.array([0.1 for i in range(len(self.aaindex1[self.index].values()))])
        std_values1 = self.standarize(values1)

        vectors = {}
        for i, key in enumerate(keys):
            vectors[key] = [std_values1[i], values2[i]]
        return vectors
    
    def generate_normed_verctors(self):
        self.load_aaindex1()
        keys = self.aaindex1[self.index].keys()
        values1 = np.array(list(self.aaindex1[self.index].values()))
        # values2 = np.array(list(self.aaindex1[self.index2].values()))
        values2 = np.array([0.1 for i in range(len(self.aaindex1[self.index].values()))])
        std_values1 = self.standarize(values1)
        vectors = {}
        for i, key in enumerate(keys):
            vectors[key] = [std_values1[i], values2[i]]
        return vectors
        

    def standarize(self, values: np.array):
        return (values - np.mean(values)) / np.std(values) 
    
    def normalize(self, values: np.array):
        print(values)
        return (values - np.min(values)) / (np.max(values) - np.min(values))
    
    def draw_vectors(self, vectors):
        plt.figure()
        for key, items in vectors.items():
            plt.plot([0, items[0]], [0, items[1]])
            plt.text(items[0], items[1], key)
        plt.title(self.index)
        fname = os.path.join(self.experiment_directory, "vectors.pdf")
        draw = Draw()
        draw.save_figure_as_pdf(fname)

    # CALCURATE COORDINATE
    def calc_coordinate(self):
        # vectors = self.generate_std_vectors()
        vectors = self.generate_normed_verctors()
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

    def read_sequences(self, path):
        sequences = []
        with open(path, "r") as f:
            for l in f.readlines():
                label, seq = l.split()
                sequences.append(seq)
        return sequences
    
    def read_labels(self, path):
        labels = []
        with open(path, "r") as f:
            for l in f.readlines():
                label, seq = l.split()
                labels.append(label)
        return labels
    
    # MAKE IMAGES INFORMATION
    def make_images_info(self):
        labels = self.read_labels(self.amino_train_path) + self.read_labels(self.amino_test_path)
        self.read_trans(self.amino_info_path)

        trans = self.read_trans(self.amino_info_path)

        images_info = {}
        for i, label in enumerate(labels):
            fname = os.path.join(self.images_directory, str(i) + ".png")
            images_info[i] = trans[label] + [fname]

        columns = ["subsubfamily", "family", "subfamily", "path"]
        pd.DataFrame.from_dict(images_info, orient="index", columns=columns).to_csv(self.images_info_path)

    def read_trans(self, path):
        trans = {}
        with open(path, "r") as f:
            for l in f.readlines():
                l_split = l.split()
                trans[l_split[0]] = l_split[1:]
        return trans

    def generate_images(self):
        idx = 0
        for i in tqdm.tqdm(range(self.DATA_NUM)):
            data_path = os.path.join(self.coordinates_directory, str(idx) + ".dat")
            with open(data_path, "r") as f:
                dat = np.array([list(map(float, l.split(","))) for l in f.readlines()])

                fname = os.path.join(self.images_directory, str(idx) + ".pgm")
                self.generate_image(dat, fname)

            idx += 1
    
    # SET DIRECTION
    hear, right, left, up, down, up_right = [np.array(list) for list in [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]]]

    def generate_image(self, dat, fname):

        current_pix = 0


        def setpic (point): 
            nonlocal current_pix
            pix[tuple(point)] = current_pix
            current_pix += 1
            # pix[tuple(point)] = 1 # BINARY

        def drawline(from_p, dest_p):
            nonlocal current_pix
            dx, dy = dest_p - from_p

            if dx < 0: 
                dest_p, from_p = from_p, dest_p
                dx, dy = dest_p - from_p

            n_vec = np.array([dy, -dx])

            start = np.array([int(np.floor(p)) for p in from_p]) 
            endp = np.array([int(np.floor(p)) for p in dest_p]) 

            # setpic(start)
            pix[tuple(start)] = current_pix

            movp =  np.copy(start) 
            
            mov2 = self.up if dy >= 0 else self.down

            while not (np.allclose(movp, endp)): 

                vec1 = movp + self.right - from_p
                vec2 = movp + self.up_right - from_p

                if np.dot(n_vec, vec1) * np.dot(n_vec, vec2) <= 0: 
                    movp += self.right
                else:
                    movp += mov2
                        
                # setpic(movp) 
                pix[tuple(movp)] = current_pix
                current_pix += 1

        # INITIAL 
        MAX_PIX = 255 # GRAYSCALE
        img = np.array([float(self.IMAGE_SIZE), float(self.IMAGE_SIZE)])
        pix  = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=int) 

        # RESCALE 
        max, min = np.max(dat, axis=0), np.min(dat, axis=0)
        width, height = max - min
        imgw, imgh = img - 1

        rat = np.max([width / imgw, height / imgh]) # SAME RATION OF HEIGHT AND WIDTH
        # rat = np.array([width / imgw, height / imgh])
        dat = dat / rat
        max, min = (max, min) / rat # SAME RATION OF HEIGHT AND WIDTH

        mid = (max + min) / 2.0
        dat = [row - mid + img / 2. for row in dat]

        for i in range(len(dat) - 1):
            drawline(dat[i], dat[i+1])

        # RESCALE PIX
        amin, amax = np.amin(pix), np.amax(pix)
        pix = np.interp(pix, (amin, amax), (0, MAX_PIX)).astype(int)

        with open(fname, "w") as f:
            print("P2\n %d %d\n%d" % (self.IMAGE_SIZE, self.IMAGE_SIZE, MAX_PIX), file=f) 
            # print ("P1\n %d %d" % (self.IMAGE_SIZE, self.IMAGE_SIZE), file=f)
            for row in np.flipud(pix.T):
                print(*(MAX_PIX - row), file=f)  # REVERSE: GRAYSCLE
                # print(*(row + 1) % 2, file=f)    # REVERSE: BIMARY 

    def convert_pgm(self):
        for i in tqdm.tqdm(range(self.DATA_NUM)):
            pgm_fname = os.path.join(self.images_directory, str(i) + ".pgm")
            png_fname = os.path.join(self.images_directory, str(i) + ".png")

            cv2.imwrite(png_fname, cv2.imread(pgm_fname))

class DeepImFam(Common):
    def __init__(self, config_path, index) -> None:
        Common.__init__(self, config_path, index)

    def train(self):
        df = pd.read_csv(self.images_info_path)
        train_df, test_df = train_test_split(
            df, test_size=.2, stratify=df[self.hierarchy_label], 
            shuffle=True, random_state=0)
        
        # OVERSAMPLING
        # sampler = RandomOverSampler(random_state=42)
        # train_df, _ = sampler.fit_resample(train_df, train_df[self.hierarchy_label])

        # SET ImageDataDrameGenerator
        image_data_frame_gen = self.ImageDataFrameGenerator(
            images_directory=self.images_directory,
            x_col="path",
            y_col=self.hierarchy_label,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )

        train_gen = image_data_frame_gen.get_generator(df=train_df, shuffle=True)
        test_gen = image_data_frame_gen.get_generator(df=test_df, shuffle=False)

        # CALLBACK
        # reduce_lr = ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.1,
        #     patience=10,
        #     min_lr=1e-5
        # )

        # モデル
        # early_stopping = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=0.0,
        #     patience=30,
        # )

        model = self.generate_model()
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=30,
            # callbacks=[reduce_lr, early_stopping],
            batch_size=512,
        )    

        # SAVE MODEL
        fname = os.path.join(self.results_directory, "model.h5")
        model.save(fname)

        # SAVE RESULT
        result = history.history
        fname = os.path.join(self.results_directory, "history.csv")
        pd.DataFrame(result).to_csv(fname)
        if not os.path.exists(self.metrics_path):
            metrics = {}
        else:
            with open(self.metrics_path, "r") as f:
                metrics = json.load(f)
        for key in ["loss", "accuracy"]:
            metrics[key] = result[key][-1]
            metrics["val_" + key] = result["val_" + key][-1]

        test_f1 = f1_score(test_gen.labels, np.argmax(model.predict(test_gen), axis=1), average="macro")
        metrics["test_f1"] = test_f1
        self.save_obj(metrics, self.metrics_path)

    def split_train_test(self, df, test_size=.2):
        return train_test_split(
            df, test_size=test_size, stratify=df[self.hierarchy_label], 
            shuffle=True, random_state=0)

    def generate_model(self):
        model = Sequential([
            Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(16, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(self.dropout_ratio),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(self.dropout_ratio),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(self.dropout_ratio),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(5, activation="softmax"),
            ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            # loss="categorical_crossentropy",
            loss=CategoricalFocalCrossentropy(name="CategoricalFocal"),
            metrics=["accuracy"]
        )

        return model

    class ImageDataFrameGenerator:
        image_data_gen = ImageDataGenerator(
            preprocessing_function=lambda img: 1. - img / 255.,
            # rescale = 1 / 255.
        )

        def __init__(self, images_directory, x_col, y_col, target_size, batch_size=512,) -> None:
            self.images_directory = images_directory
            self.x_col = x_col
            self.y_col = y_col
            self.target_size = target_size
            self.batch_size = batch_size,

        def get_generator(self, df, shuffle=False):
            # CALC SAMPLE WEIGHT
            sample_weight = compute_sample_weight(class_weight="balanced", y=df[self.y_col])

            # SET GENERATOR
            generator = self.image_data_gen.flow_from_dataframe(
                dataframe=df,
                directory=self.images_directory,
                x_col=self.x_col,
                y_col=self.y_col,
                shuffle=shuffle,
                seed=0,
                target_size=self.target_size,
                color_mode="grayscale",
                class_mode="categorical",
                sample_weight=sample_weight
            )
            return generator
        
    def predict(self):
        # LOAD IMAGE DATA
        df = pd.read_csv(self.images_info_path)
        train_df, test_df = train_test_split(
            df, test_size=.2, stratify=df[self.hierarchy_label], 
            shuffle=True, random_state=0)

        # SET ImageDataDrameGenerator
        image_data_frame_gen = self.ImageDataFrameGenerator(
            images_directory=self.images_directory,
            x_col="path",
            y_col=self.hierarchy_label,
            target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )

        train_gen = image_data_frame_gen.get_generator(df=train_df, shuffle=False)
        test_gen = image_data_frame_gen.get_generator(df=test_df, shuffle=False)

        # LOAD MODEL
        model = self.load_model()

        # PREDICT
        train_proba = model.predict(train_gen)
        test_proba = model.predict(test_gen)
        train_pred = np.argmax(train_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)

        print("F1-score", f1_score(test_gen.labels, test_pred, average="macro"))

        # SAVE PREDICT PROBA
        train_fname = os.path.join(self.results, "train_proba_weighted.csv")
        test_fname = os.path.join(self.results, "test_proba_weighted.csv")
        if not os.path.exists(train_fname):
            self.save_dict_as_dataframe({"labels": train_gen.labels}, train_fname)
            self.save_dict_as_dataframe({"labels": test_gen.labels}, test_fname)

        train_dict = self.load_csv_as_dict(train_fname)
        test_dict = self.load_csv_as_dict(test_fname)
        for i in range(5):
            train_dict["-".join([self.index, str(i)])] = train_proba[:, i]
            test_dict["-".join([self.index, str(i)])] = test_proba[:, i]
        self.save_dict_as_dataframe(train_dict, train_fname)
        self.save_dict_as_dataframe(test_dict, test_fname)

        # SAVE PREDICT LABELS
        train_fname = os.path.join(self.results, "train_predict_weighted.csv")
        test_fname = os.path.join(self.results, "test_predict_weighted.csv")
        if not os.path.exists(train_fname):
            self.save_dict_as_dataframe({"labels": train_gen.labels}, train_fname)
            self.save_dict_as_dataframe({"labels": test_gen.labels}, test_fname)

        train_dict = self.load_csv_as_dict(train_fname)
        train_dict[self.index] = train_pred.tolist()
        self.save_dict_as_dataframe(train_dict, train_fname)

        test_dict = self.load_csv_as_dict(test_fname)
        test_dict[self.index] = test_pred.tolist()
        self.save_dict_as_dataframe(test_dict, test_fname)

        return train_gen.labels, train_pred, test_gen.labels, test_pred

    def load_model(self):
        # LOAD MODEL
        fname = os.path.join(self.results_directory, "model.h5")
        model = keras.models.load_model(fname)
        return model
    
    def draw_history(self):
        fname = os.path.join(self.results_directory, "history.csv")
        history = pd.read_csv(fname)

        draw = Draw()
        loss_fname = os.path.join(self.results_directory, "loss.pdf")
        draw.draw_history(history, "loss", loss_fname)
        accuracy_fname = os.path.join(self.results_directory, "accuracy.pdf")
        draw.draw_history(history, "accuracy", accuracy_fname)

    def draw_cm(self):
        train_labels, train_predictions, test_labels, test_predictions = self.predict()

        print("macro-f1-score: ", f1_score(test_labels, test_predictions, average="macro"))
        print("micro-f1-score: ", f1_score(test_labels, test_predictions, average="micro"))

        draw = Draw()        
        # TRAIN
        cm_fname = os.path.join(self.results_directory, "train_cm.pdf")
        cm = confusion_matrix(train_labels, train_predictions)
        draw.draw_cm(cm, cm_fname)

        cm_normed_fname = os.path.join(self.results_directory, "train_cm_normed.pdf")
        cm_normed = confusion_matrix(train_labels, train_predictions, normalize="true")
        draw.draw_cm(cm_normed, cm_normed_fname, norm=True)

        # TEST
        cm_fname = os.path.join(self.results_directory, "cm.pdf")
        cm = confusion_matrix(test_labels, test_predictions)
        draw.draw_cm(cm, cm_fname)
        
        cm_normed_fname = os.path.join(self.results_directory, "cm_normed.pdf")
        cm_normed = cm = confusion_matrix(test_labels, test_predictions, normalize="true")
        draw.draw_cm(cm_normed, cm_normed_fname, norm=True)

    def cross_validate(self):
        df = pd.read_csv(self.images_info_path)
        index = self.validate_index(df, df[self.hierarchy_label])
        results = []
        scores = []

        for i, (train_index, test_index) in enumerate(index):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            # OVERSAMPLING
            # sampler = RandomOverSampler(random_state=42)
            # train_df, _ = sampler.fit_resample(train_df, train_df[self.hierarchy_label])

            # SET ImageDataDrameGenerator
            image_data_frame_gen = self.ImageDataFrameGenerator(
                images_directory=self.images_directory,
                x_col="path",
                y_col=self.hierarchy_label,
                target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                batch_size=self.BATCH_SIZE
            )

            train_gen = image_data_frame_gen.get_generator(train_df, shuffle=True)
            test_gen = image_data_frame_gen.get_generator(test_df, shuffle=False)

            # CALLBACK
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=20,
                min_lr=1e-5,
            )

            # モデル
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=80,
            )

            model = self.generate_model()
            history = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=1000,
                callbacks=[reduce_lr, early_stopping],
                batch_size=512,
            )  

            pred = np.argmax(model.predict(test_gen), axis=1)

            # SAVE RESULTS 
            fname = os.path.join(self.results_directory, "-".join([str(i), "crossvalidation.csv"]))
            pd.DataFrame(history.history).to_csv(fname)
            results.append(history.history["val_accuracy"][-1])
            scores.append(f1_score(test_gen.labels, pred, average="macro"))
            
        if not os.path.exists(self.metrics_path): 
            metrics = {}
        else:
            with open(self.metrics_path, "r") as f:
                metrics = json.load(f)
        metrics["results"] = results
        metrics["scores"] = scores
        self.save_obj(metrics, self.metrics_path)
            
    def validate_index(self, df, labels):
        index = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in kf.split(df, labels):
            index.append((train_idx, test_idx))
        return index
    

class Ensemble(Common):
    def __init__(self, config_path, index) -> None:
        Common.__init__(self, config_path, index)

    def train(self):
        train_df, test_df = self.load_data(is_predict=False)

        sampler = RandomOverSampler(random_state=42)
        train_df, _ = sampler.fit_resample(train_df, train_df["labels"])

        train_df, train_labels = self.split_labels(train_df)
        test_df, test_labels = self.split_labels(test_df)

        # print(train_df.head())
        # train_df = self.dummy_columns(train_df)
        # test_df = self.dummy_columns(test_df)

        # XGBoost
        model = XGBClassifier(
            n_estimators=1000,
            # early_stopping_rounds=15,
            )
        model.fit(
            train_df, train_labels,
            eval_set=[(train_df, train_labels), (test_df, test_labels)],
            verbose=True,
        )

        # Random Forest
        # model = RandomForestClassifier(
        #     n_estimators=1000,
        #     verbose=True,
        #     )
        
        # model.fit(
        #     train_df, train_labels,
        #     )

        # CatBoost
        # model = CatBoostClassifier(
        #     iterations=1000,
        #     use_best_model=True,
        # )
        # model.fit(
        #     train_df, train_labels,
        #     eval_set=(test_df, test_labels),
        # )

        train_pred = model.predict(train_df)
        test_pred = model.predict(test_df)

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)

        # DRAW IMPORTANCE
        draw = Draw()
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), np.array(train_df.columns)[sorted_idx])
        plt.title("Feature Importance")
        fname = os.path.join(self.results, "importance.pdf")
        draw.save_figure_as_pdf(fname)

        # print(feature_importance)

        accuracy_train = accuracy_score(train_labels, train_pred)
        accuracy_test = accuracy_score(test_labels, test_pred)
        f1_train = f1_score(train_labels, train_pred, average="macro")
        f1_test = f1_score(test_labels, test_pred, average="macro")
        print("accuracy(train): ", accuracy_train)
        print("accuracy(test): ", accuracy_test)
        

        fname = os.path.join(self.results, "metrics.json")
        with open(fname, "w") as f:
            json.dump({
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "f1_train": f1_train,
                "f1_test": f1_test,
            }, f, indent=2)
        
        # Draw Confusion Matrix
        fname = os.path.join(self.results, "cm.pdf")
        cm = confusion_matrix(test_labels, test_pred)
        draw.draw_cm(cm, fname)
        fname = os.path.join(self.results, "normed_cm.pdf")
        normed_cm = confusion_matrix(test_labels, test_pred, normalize="true")
        draw.draw_cm(normed_cm, fname, norm=True)

    def split_labels(self, df: pd.DataFrame):
        labels = df["labels"]
        df = df.drop("labels", axis=1)
        return df, labels
    
    def drop_column(self, df: pd.DataFrame, column: str):
        return df.drop(column, axis=1)
    
    def dummy_columns(self, df: pd.DataFrame):
        for column in df.columns:
            # print(pd.get_dummies(df[column]))
            df = pd.concat([self.drop_column(df, column), pd.get_dummies(df[column], prefix=column, prefix_sep='-')], axis=1)
        return df

    def load_data(self, is_predict=True):
        if is_predict:
            train_fname = os.path.join(self.results, "train_predict_weighted.csv")
            test_fname = os.path.join(self.results, "test_predict_weighted.csv")
        else:
            train_fname = os.path.join(self.results, "train_proba_weighted.csv")
            test_fname = os.path.join(self.results, "test_proba_weighted.csv")
        train_df = pd.read_csv(train_fname)
        test_df = pd.read_csv(test_fname)
        return train_df, test_df

if __name__ == "__main__":
    # TODO: AAindex
    # aaindex1 = AAindex1(config_path="config.yaml")    
    # aaindex1.calc()
    # aaindex1.disp("NAKH900107", "PALJ810108")

    # TODO: Generate Images
    # image_gen = ImageGenerator(config_path="config.yaml")
    # image_gen.calc_coordinate()
    # image_gen.make_images_info()
    # image_gen.generate_images()
    # image_gen.convert_pgm()

    # TODO: Train DeepImFam
    deepimfam = DeepImFam(config_path="config.yaml")
    # deepimfam.train()
    # deepimfam.load_model()
    # deepimfam.predict()
    obj = deepimfam.load_csv_as_dict("/home/mizuno/data/results/train_predict.csv")
    print(obj)
    
    # TODO: Draw Result
    # deepimfam.draw_history()
    # deepimfam.draw_cm()