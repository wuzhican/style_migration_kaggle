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

batch_size = 2
root_dir='./data'
data_save_root='./checkpoint'
models_choice_from = [
    "ImfwNet"
]
    

parser = ArgumentParser()
parser.add_argument("--model", type=str, default='DeeplabUpsampleModel')
parser.add_argument("--auto_resume", type=bool, default=True)
# parser = models.DeeplabUpsampleModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(resume_from_checkpoint=GetResumePath(data_save_root))
args = parser.parse_args()
arg_v = vars(args)

if arg_v['model'] not in models_choice_from:
    print("model choice is not in %s,exit"%(str(models_choice_from)))
    exit(-1)
else:
    if arg_v['model'] == "ImfwNet":
        module = models.FWNetModule(
            load_image(os.path.join(root_dir,'style.jpeg'),shape=(256,256)),
            automatic_optimization=False
        )
        train_dataset = loaders.styleLoader(os.path.join(root_dir,'train'))
        loader = (
            DataLoader(train_dataset, batch_size=batch_size,num_workers=2,drop_last=True),
        )
        hooks = [EarlyStopping(monitor="train_loss", min_delta=0.00, patience=3, verbose=False, mode="min")]

if __name__ == '__main__':
    trainer = pl.Trainer.from_argparse_args(args, callbacks=hooks, default_root_dir=data_save_root,max_epochs = -1)
    if arg_v['auto_scale_batch_size']:
        trainer.tune(module,*loader)
    else:
        trainer.fit(module,*loader)

