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

def main(lp_path: str, run_folder: Optional[str] = None):
    config = load_config_from_launchplan(lp_path)
    # create unique folder for this run
    if run_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"sms/exp1/runs/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    # save the original launchplan and the loaded config (in case pointed-to files are changed)
    shutil.copy(lp_path, os.path.join(run_folder, "original_launchplan.yaml"))
    with open(os.path.join(run_folder, "loaded_config.yaml"), 'w') as f:
        yaml.dump(config, f)    

    run_training(config=config, mode='pretrain', run_folder=run_folder)
    run_training(config=config, mode='finetune', run_folder=run_folder)

def run_training(config: LaunchPlanConfig, mode: str, run_folder: str):
    config = config.model_dump()

    if mode == 'pretrain':
        dl_cfg = config["pt_dl"]
        loss_cfg = config["pt_loss"]
        opt_cfg = config["pt_optimizer"]
        sch_cfg = config["pt_scheduler"]
        train_cfg = config["pt_training"]
    elif mode == 'finetune':
        dl_cfg = config["ft_dl"]
        loss_cfg = config["ft_loss"]
        opt_cfg = config["ft_optimizer"]
        sch_cfg = config["ft_scheduler"]
        train_cfg = config["ft_training"]
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

    encoder = getattr(encoders, config["encoder"]["type"])(
        **config["encoder"]["params"],
        input_shape=config["dims"]["input_shape"],
        d_latent=config["dims"]["d_latent"]
    )
    projector = projectors.ProjectionHead(
        **config["projector"]["params"],
        d_latent=config["dims"]["d_latent"],
        d_projected=config["dims"]["d_projected"]
    )
    model = siamese.SiameseModel(encoder, projector)

    if mode == 'finetune':
        model.load_state_dict(torch.load(config["model_paths"]["pretrained_model_path"]))
        model = model.get_encoder()

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
        model_save_path=os.path.join(run_folder, f'{mode}_saved_model.pth'),
        epochs=train_cfg["epochs"],
        early_stopping_patience=train_cfg["early_stopping_patience"]
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
        default='sms/exp1/launchplans/01.yaml',
        help='Path to the training launchplan yaml file.'
    )
    parser.add_argument(
        '--rf', '--run_folder',
        type=str,
        default=None,
        help='Path to the folder where the run will be saved.',
        required=False
    )
    args = parser.parse_args()
    main(args.lp, args.rf)