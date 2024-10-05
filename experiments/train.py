# Run a baseline model in BasicTS framework.


import os
import sys
from argparse import ArgumentParser

sys.dont_write_bytecode = True

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
import basicts
from easytorch.config import import_config

torch.set_num_threads(4)  # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser(
        description="Run time series forecasting model in BasicTS framework!"
    )
    parser.add_argument(
        "-c", "--cfg", default="baselines/STID/gcn_seq.py", help="training config"
    )
    parser.add_argument("-ct", "--conv_type", default="GCN", type=str)
    parser.add_argument("-d", "--dense", action="store_true", help="evaluation only")
    parser.add_argument("-i", "--identity", action="store_true", help="evaluation only")
    parser.add_argument("-g", "--gpus", default="1", type=str)

    return parser.parse_args()


def modify_config_file(
    config_file: str,
    conv_type: str,
    identity: bool,
    dense: bool,
):
    """
    Modifies the configuration file to update the 'conv_type', 'expand', and 'identity' parameters.
    """
    with open(config_file, "r") as file:
        lines = file.readlines()

    # Find and replace the lines containing 'conv_type', 'expand', and 'identity'
    with open(config_file, "w") as file:
        for line in lines:
            if "conv_type =" in line:
                file.write(f"conv_type = '{conv_type}'\n")
            elif "identity =" in line:
                file.write(f"identity = {str(identity).capitalize()}\n")
            elif "dense =" in line:
                file.write(f"dense = {str(dense).capitalize()}\n")
            else:
                file.write(line)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    modify_config_file(
        args.cfg, args.conv_type, args.identity, args.dense
    )
    cfg = import_config(args.cfg)
    basicts.launch_training(cfg, args.gpus, node_rank=0)
