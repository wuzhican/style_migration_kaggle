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

# import tensorboard

batch_size = 2
root_dir='./data'
data_save_root='./checkpoint'
style_image_path='./data/style.jpeg'
models_choice_from = [
    "ImfwNet",
    "SMNet"
]
default_args = [
    ['--model',str,'DeeplabUpsampleModel'],
    ['--learning_rate',float,1e-3],
    ['--root_dir',str,root_dir],
    ['--data_save_root',str,root_dir],
    ['--style_image_path',str,style_image_path],
    ['--batch_size',int,batch_size],
    ['--style_weight',float,1e3],
    ['--content_weight',float,1],
    ['--tv_weight',float,1e-3],
]
    

parser = ArgumentParser()
parser.add_argument("--auto_resume", action="store_true")
for arg in default_args:
    parser.add_argument(arg[0],type=arg[1],default=arg[2])

# parser = models.DeeplabUpsampleModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
arg_v = vars(args)
root_dir = arg_v['root_dir']
data_save_root = arg_v['data_save_root']
style_image_path = arg_v['style_image_path']
batch_size = arg_v['batch_size']
lr = arg_v['learning_rate']
style_weight = arg_v['style_weight']
content_weight = arg_v['content_weight']
tv_weight = arg_v['tv_weight']


if arg_v['model'] not in models_choice_from:
    print("model choice is not in %s,exit"%(str(models_choice_from)))
    exit(-1)
else:
    if arg_v['model'] == "ImfwNet":
        module = models.FWNetModule(
            load_image(style_image_path,shape=(256,256)),
            style_weight=style_weight,
            content_weight=content_weight,
            tv_weight=tv_weight,
            automatic_optimization=False,
            lr=lr
        )
        train_dataset = loaders.styleLoader(root_dir,augment_ratio=2)
        loader = (
            DataLoader(train_dataset, batch_size=batch_size,num_workers=2,drop_last=True),
        )
        hooks = [EarlyStopping(monitor="train_loss", min_delta=0.1, patience=3, verbose=False, mode="min")]
        logger = TensorBoardLogger("./data/lightning_logs", name="ImfwNet")
        models_args={
            'logger':logger
        }
    elif arg_v['model'] == 'SMNet':
        module = models.SMNet(
            load_image(style_image_path,shape=(512,512)),
            style_weight=style_weight,
            content_weight=content_weight,
            automatic_optimization=False
        )
        train_dataset = loaders.styleLoader(root_dir,augment_ratio=1)
        loader = (
            DataLoader(train_dataset, batch_size=batch_size,num_workers=2,drop_last=True),
        )
        hooks = [EarlyStopping(monitor="loss", min_delta=1e-2, patience=9, verbose=False, mode="min")]
        logger = TensorBoardLogger("./data/lightning_logs", name="SMNet")
        models_args={
            'logger':logger
        }

if arg_v['auto_resume'] or arg_v['resume_path']:
    models_args['resume_from_checkpoint'] = GetResumePath(data_save_root if not arg_v['resume_path'] else arg_v['resume_path'])
    hooks=[]

if __name__ == '__main__':
    trainer = pl.Trainer.from_argparse_args(args, callbacks=hooks, default_root_dir=data_save_root,**models_args)
    if arg_v['auto_scale_batch_size']:
        trainer.tune(module, *loader)
    else:
        trainer.fit(module, *loader)

