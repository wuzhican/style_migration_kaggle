import os,models
import loaders
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from utils.Functions import *
from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

batch_size = 2
root_dir='./data'
data_save_root='./checkpoint'
style_image_path='./data/style.jpeg'
models_choice_from = [
    "ImfwNet"
]
    

parser = ArgumentParser()
parser.add_argument("--model", type=str, default='DeeplabUpsampleModel')
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--auto_resume", type=bool, default=True)
parser.add_argument("--root_dir", type=str, default=root_dir)
parser.add_argument("--data_save_root", type=str, default=root_dir)
parser.add_argument("--style_image_path", type=str, default=style_image_path)
parser.add_argument("--batch_size", type=int, default=batch_size)

# parser = models.DeeplabUpsampleModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
# parser.set_defaults(resume_from_checkpoint=GetResumePath(data_save_root))
args = parser.parse_args()
arg_v = vars(args)
root_dir = arg_v['root_dir']
data_save_root = arg_v['data_save_root']
style_image_path = arg_v['style_image_path']
batch_size = arg_v['batch_size']
lr = arg_v['learning_rate']

if arg_v['model'] not in models_choice_from:
    print("model choice is not in %s,exit"%(str(models_choice_from)))
    exit(-1)
else:
    if arg_v['model'] == "ImfwNet":
        module = models.FWNetModule(
            load_image(style_image_path,shape=(256,256)),
            automatic_optimization=False,
            lr=lr
        )
        train_dataset = loaders.styleLoader(root_dir,augment_ratio=2)
        loader = (
            DataLoader(train_dataset, batch_size=batch_size,num_workers=2,drop_last=True),
        )
        hooks = [EarlyStopping(monitor="train_loss", min_delta=0.00, patience=9, verbose=False, mode="min")]
        logger = TensorBoardLogger("./data/lightning_logs", name="ImfwNet")
        models_args={
            'logger':logger
        }

if __name__ == '__main__':
    trainer = pl.Trainer.from_argparse_args(args, callbacks=hooks, default_root_dir=data_save_root,max_epochs = -1,**models_args)
    if arg_v['auto_scale_batch_size']:
        trainer.tune(module, *loader)
    else:
        trainer.fit(module, *loader)

