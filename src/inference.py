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
# To take inference from trained model
from src.get_prediction import get_prediction
# Class for preprocessing data
from src.data_handler import DataGenerator
# Function for split image ids to train and validation set
from src.utils import train_valid_ids_split
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def inference():
    """The primary module to debug any process for the model."""

    # Load config
    cfg = AttrDict(load_config("config.yaml"))

    # Setting test dir & ground truth path
    test_dir = Path(cfg.dataset.test_dir)
    gt_path = Path(cfg.dataset.gt_path)

    # Setting ground truth dataframe
    gt_df = pd.read_csv(gt_path)

    # Define some variables for preprocess
    image_size = cfg.preprocess.image_size
    batch_size = cfg.preprocess.batch_size
    color_channels = cfg.preprocess.color_channels

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

    # Load the model
    if (entire_model_path / entire_file).exists():
        # Load model from file
        asd_model = load_model(str(entire_model_path / entire_file))
    elif (entire_model_path / h5_file).exists():
        # Load model from file
        asd_model = load_model(str(entire_model_path / h5_file))
    elif (model_weights_path / (weights_file + ".index")).exists():
        # Get the model
        asd_model = get_model(image_size, 1)
        # Set hyperparameters
        lr_rate = cfg.hyper.lr_rate
        # Compile model after built
        asd_model.compile(loss=DiceLoss(),
                          optimizer=Adam(learning_rate=lr_rate),
                          metrics=[DiceScore()])
        # Load weights to built model
        asd_model.load_weights(model_weights_path / weights_file)


    # === Inference process

    # Set list of ids for predictions
    predict_list_ids = os.listdir(test_dir)

    # Initializing prediction data generator
    predict_data_gen = DataGenerator(predict_list_ids, gt_df, mode="predict", base_path=str(test_dir),
                                     img_size=image_size, color_channels=color_channels,
                                     shuffle=False)

    # Predict on batch and image
    predict_batch_number = cfg.prediction.predict_batch_number
    predict_image_number = cfg.prediction.predict_image_number

    # Run inference process
    get_prediction(asd_model, predict_data_gen, predict_data_gen, predict_batch_number, predict_image_number)


if __name__ == "__main__":
    inference()
