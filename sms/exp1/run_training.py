import argparse
import os
import shutil
import yaml
import json
from datetime import datetime
import torch
import torch.optim as optim
from functools import partial
from typing import Optional
import logging

from sms.exp1.config_classes import load_config_from_launchplan, LaunchPlanConfig
from sms.exp1.training.trainer import Trainer
from sms.exp1.data.dataloader import get_dataloader

import sms.exp1.models.encoders as encoders
import sms.exp1.models.projector as projectors
import sms.exp1.models.siamese as siamese
import sms.exp1.training.loss_functions as loss_functions

import logging
from sms.src.log import configure_logging

configure_logging(console_level=logging.INFO)

def build_encoder(dumped_lp_config: dict):
    return getattr(encoders, dumped_lp_config["encoder"]["type"])(
        **dumped_lp_config["encoder"]["params"],
        input_shape=dumped_lp_config["dims"]["input_shape"],
        d_latent=dumped_lp_config["dims"]["d_latent"]
    )

def build_projector(dumped_lp_config: dict):
    return projectors.ProjectionHead(
        **dumped_lp_config["projector"]["params"],
        d_latent=dumped_lp_config["dims"]["d_latent"],
        d_projected=dumped_lp_config["dims"]["d_projected"]
    )

def main(lp_path: str, mode: str, run_folder: Optional[str] = None):
    """
    Run the training for the Siamese network experiment.
    Pretraining/finetuning/both can be run with one call. 
    run_folder determines where logs and models are saved. It defaults to sms/exp1/runs/run_{timestamp}.
    If mode is 'both' or 'pretrain', only lp_path is required. If a run_folder is given and it already exists, we use f"{run_folder}_2".
    If mode is 'finetune', then lp_path and run_folder are required. It is assumed that the pretrained model 
        is saved in the run_folder, with the name specified in the config if something other than the default.

    Args:
        lp_path (str): Path to the launchplan yaml file.
        mode (str): Mode to run the training. Can be 'pretrain', 'finetune', or 'both'.
        run_folder (str): Path to the folder where the run will be saved.
    """
    logger = logging.getLogger(__name__)
    config = load_config_from_launchplan(lp_path)
    # create unique folder for this run
    if run_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"sms/exp1/runs/run_{timestamp}"
    if mode in ['both', 'pretrain']:
        i = 1
        while os.path.exists(run_folder):
            logger.info(f"Folder {run_folder} already exists. Trying {run_folder}_{i}.")
            run_folder = f"{run_folder}_{i}"
            i += 1
    os.makedirs(run_folder)

    # save the original launchplan and the loaded config (in case pointed-to files are changed)
    shutil.copy(lp_path, os.path.join(run_folder, "original_launchplan.yaml"))
    with open(os.path.join(run_folder, "loaded_config.yaml"), 'w') as f:
        yaml.dump(config.model_dump(), f)    

    if mode in ['both', 'pretrain']:
        run_training(config=config, mode='pretrain', run_folder=run_folder)
    
    if mode in ['both', 'finetune']:
        run_training(config=config, mode='finetune', run_folder=run_folder)

def run_training(config: LaunchPlanConfig, mode: str, run_folder: str):

    config = config.model_dump()

    if mode == 'pretrain':
        dl_cfg = config["pt_dl"]
        loss_cfg = config["pt_loss"]
        opt_cfg = config["pt_optimizer"]
        sch_cfg = config["pt_scheduler"]
        train_cfg = config["pt_training"]
        save_path = config["mod_paths"]["pt_model_path"] if config["mod_paths"] else os.path.join(run_folder, 'pretrain_saved_model.pth')
    elif mode == 'finetune':
        dl_cfg = config["ft_dl"]
        loss_cfg = config["ft_loss"]
        opt_cfg = config["ft_optimizer"]
        sch_cfg = config["ft_scheduler"]
        train_cfg = config["ft_training"]
        save_path = config["mod_paths"]["ft_model_path"] if config["mod_paths"] else os.path.join(run_folder, 'finetune_saved_model.pth')

    train_loader = get_dataloader(
        data_paths=dl_cfg["train_data_path"],
        format_config=config["input"],
        mode=mode,
        use_transposition=dl_cfg["use_transposition"],
        neg_enhance=dl_cfg["neg_enhance"],
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        use_sequence_collate_fn=dl_cfg["use_sequence_collate_fn"],
        shuffle=dl_cfg["shuffle"]
    )
    
    val_loader = get_dataloader(
        data_paths=dl_cfg["val_data_path"],
        format_config=config["input"],
        mode=mode,
        use_transposition=dl_cfg["use_transposition"],
        neg_enhance=dl_cfg["neg_enhance"],
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        use_sequence_collate_fn=dl_cfg["use_sequence_collate_fn"],
        shuffle=dl_cfg["shuffle"]
    )

    encoder = build_encoder(config)
    projector = build_projector(config)
    model = siamese.SiameseModel(encoder, projector)

    pt_path = config["mod_paths"]["pt_model_path"] if config["mod_paths"] else os.path.join(run_folder, 'pretrain_saved_model.pth')
    if mode == 'finetune':
        model.load_state_dict(torch.load(pt_path, weights_only=True))
        model.set_use_projection(False)

    loss = partial(getattr(loss_functions, loss_cfg["type"]), **loss_cfg["params"])

    optimizer = getattr(optim, opt_cfg["type"])(model.parameters(), **opt_cfg["params"])
    scheduler = getattr(optim.lr_scheduler, sch_cfg["type"])(optimizer, **sch_cfg["params"])

    trainer = Trainer(
        config=config,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        run_folder=run_folder,
        epochs=train_cfg["epochs"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
        model_save_path=save_path
    )

    metrics = trainer.train()
    metrics_path = os.path.join(run_folder, f'{mode}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Training for Siamese Network")
    parser.add_argument(
        '--lp', '--launchplan',
        type=str,
        required=True,
        help='Path to the training launchplan yaml file.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['both', 'pretrain', 'finetune'],
        default='both',
        help='Training mode: both, pretrain, or finetune'
    )
    parser.add_argument(
        '--rf', '--run_folder',
        type=str,
        default=None,
        help='Path to the folder where the run will be saved.',
        required=False
    )
    args = parser.parse_args()

    main(lp_path=args.lp, mode=args.mode, run_folder=args.rf)