import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from dataset import CustomVAEDataset, VAEDataset


# Argument parsing
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c', dest="filename", metavar='FILE',
                    help='path to the config file', default='configs/vae.yaml')
args = parser.parse_args()

# Load config
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

# Setup logger
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name']
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], workers=True)

# Model and experiment setup
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# Data setup
# data = CustomVAEDataset(**config["data_params"], pin_memory=True)
data = VAEDataset(**config["data_params"], pin_memory=True)
data.setup()

# Callbacks
callbacks = [
    LearningRateMonitor(logging_interval='epoch'),
    ModelCheckpoint(
        save_top_k=2,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        monitor="val_loss",
        save_last=True
    )
]

# Trainer setup
trainer = Trainer(
    logger=tb_logger,
    callbacks=callbacks,
    strategy=DDPStrategy(find_unused_parameters=False),
    **config['trainer_params']
)

# Create output dirs
Path(f"{tb_logger.log_dir}/Samples").mkdir(parents=True, exist_ok=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(parents=True, exist_ok=True)

# Run training
print(f"======= Training {config['model_params']['name']} =======")
trainer.fit(experiment, datamodule=data)
