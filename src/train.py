from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from pathlib import Path
import os
# To build model
from src.model_builder import get_model, DiceLoss, DiceScore
# To train model
from src.training_process import model_train
# Class for preprocessing data
from src.data_handler import DataGenerator
# Function for split image ids to train and validation set
from src.utils import train_valid_ids_split
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def train():
    """The primary module to train model."""

    # Load config
    cfg = AttrDict(load_config("config.yaml"))

    # Setting train/test dir & ground truth path
    train_dir = Path(cfg.dataset.train_dir)
    gt_path = Path(cfg.dataset.gt_path)

    # Setting ground truth dataframe
    gt_df = pd.read_csv(gt_path)
    # Defining filtered lists of ids
    train_list_ids = os.listdir(train_dir)

    # Splitting dataset on train & validation list with ids
    train_list_ids, valid_list_ids = train_valid_ids_split(train_list_ids, train_split=0.8, shuffle=True)
    valid_list_ids, test_list_ids = train_valid_ids_split(valid_list_ids, train_split=0.8)
    # Set 30% of data in train_list_ids and valid_list_ids
    train_list_size = np.size(train_list_ids)
    train_list_ids = train_list_ids[:int(train_list_size * 0.30)]
    # valid_list_size = np.size(valid_list_ids)
    # valid_list_ids = valid_list_ids[:int(valid_list_size * 0.30)]

    # Define some variables for preprocess
    image_size = cfg.preprocess.image_size
    batch_size = cfg.preprocess.batch_size
    color_channels = cfg.preprocess.color_channels

    # Initializing training & validation data generators
    train_data_gen = DataGenerator(train_list_ids, gt_df, mode="fit", base_path=str(train_dir),
                                   batch_size=batch_size, img_size=image_size, color_channels=color_channels)
    valid_data_gen = DataGenerator(valid_list_ids, gt_df, mode="fit", base_path=str(train_dir),
                                   batch_size=batch_size, img_size=image_size, color_channels=color_channels,
                                   shuffle=False)

    # Define saved model path (weights and entire paths)
    entire_model_path = Path(cfg.model.entire_path)
    model_weights_path = Path(cfg.model.weights_path)

    # Creates folders to model files if they not exists
    entire_model_path.mkdir(exist_ok=True)
    model_weights_path.mkdir(exist_ok=True)

    # Define file names of saved model
    entire_file = cfg.model.files.entire
    h5_file = cfg.model.files.h5
    weights_file = cfg.model.files.weights

    # Get the model
    asd_model = get_model(image_size, 1)
    # Run training process
    asd_model = model_train(asd_model, train_data_gen, valid_data_gen)
    # Run model saving process
    asd_model.save(cfg.model.trained_path.entire)
    asd_model.save_weights(cfg.model.trained_path.weights)





if __name__ == "__main__":
    train()