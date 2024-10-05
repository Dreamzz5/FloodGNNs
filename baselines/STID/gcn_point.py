import os
import sys
import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import *
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj
import numpy as np
from .arch import Seq, Point, GCN_Point, GCN_Seq
import random

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = "Flood"  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]  # Length of input sequence
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings[
    "TRAIN_VAL_TEST_RATIO"
]  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings[
    "NORM_EACH_CHANNEL"
]  # Whether to normalize each channel of the data
RESCALE = regular_settings["RESCALE"]  # Whether to rescale the data
NULL_VAL = regular_settings["NULL_VAL"]  # Null value in the data
# Model architecture and parameters
MODEL_ARCH = GCN_Point
dense = True
expand = False
if dense:
    adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "original")
    adj_mx = torch.Tensor(adj_mx[0])
else:
    adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/topology_adj.pkl", "original")
    adj_mx = torch.Tensor(adj_mx[0])


conv_type = 'GAT'
identity = False
if identity:
    adj_mx = torch.eye(adj_mx[0].shape[0])


MODEL_PARAM = {
    "num_nodes": 358,
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 32,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    "if_node": 1,
    "node_dim": 32,
    "if_T_i_D": 0,
    "if_D_i_W": 0,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "adj_mx": adj_mx,
    "conv_type": conv_type,
    "expand": expand
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = "An Example Config"
CFG.GPU_NUM = 1  # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG._ = random.randint(-1e6, 1e6)
############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        # 'mode' is automatically set by the runner
    }
)

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler  # Scaler class
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
    }
)

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4, 5, 6]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict(
    {"MAE": masked_mae, "MAPE": masked_mape, "RMSE": masked_rmse, "NSE": masked_nse}
)
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS

suffixes = []
if identity:
    suffixes.append("identity")
if dense:
    suffixes.append("dense")
if expand:
    suffixes.append("expand")

if suffixes:
    suffix = "-".join(suffixes)
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        f"{suffix}-point-{conv_type}",
    )
else:
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        f"{MODEL_ARCH.__name__}-{conv_type}",
    )
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 50, 80], "gamma": 0.5}
CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = list(np.arange(OUTPUT_LEN) + 1)
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True
