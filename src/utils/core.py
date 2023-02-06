from dataset.nok import NokDataModule

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def assert_batch_shape(batch, shape):
    assert batch.shape[1:] == shape

def train(model_class: type, early_stopping_delta=None, max_epochs=None, config=None, test=False):

    with wandb.init(name=model_class.__name__.lower(), config=config, save_code=True):
        logger = WandbLogger(log_model=False)
        config = wandb.config

        nok_datamodule = NokDataModule(batch_size=config.batch_size, num_workers=4)

        model = model_class(**config)
        logger.watch(model, log=None)

        onnx_file = f"{logger.save_dir}/model.onnx"
        torch.onnx.export(
            model,
            model.example_input_array,
            onnx_file,
            opset_version=9,
            export_params=False,
            do_constant_folding=False,
            input_names=["father", "mother"],
            output_names=["child"],
            dynamic_axes={
                "father": {0: "batch_size"},
                "mother": {0: "batch_size"},
                "child": {0: "batch_size"},
            }
        )

        trainer = Trainer(
            logger=logger,
            gpus=-1,
            precision=16,
            callbacks= [
                ModelCheckpoint(monitor="loss", save_top_k=1, mode="min", every_n_val_epochs=1, filename="best"),
                * ([] if early_stopping_delta is None else [EarlyStopping(monitor="loss", min_delta=early_stopping_delta, patience=3)])
            ],
            max_epochs=max_epochs
        )

        trainer.fit(model, datamodule=nok_datamodule)

        if(test):
            trainer.test(model, datamodule=nok_datamodule)
    
