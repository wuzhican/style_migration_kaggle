import pytorch_lightning as pl
from tester.AbstractTester import AbstractTester
import models,loaders,utils
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class MSNetClassTester(AbstractTester):
    
    train_arg = ['content_layers','style_layers']
    
    def __init__(self,root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.content_layers = ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3']
        self.style_layers = ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3']
        
    def train(self,**args):
        arg = {}
        for key in self.train_arg:
            if key in args.keys():
                arg[key] = args[key]
        module = models.SMNet(
            utils.load_image('./data/style4.jpg', shape=(512, 512)),
            style_weight=1e6,
            content_weight=1,
            automatic_optimization=False,
            max_steps=6000,
            **arg
        )
        train_dataset = loaders.styleLoader(self.root_dir,augment_ratio=1)
        loader = (
            DataLoader(train_dataset, batch_size=1,num_workers=2,drop_last=True),
        )
        logger = TensorBoardLogger("./data/lightning_logs", name="SMNet")
        models_args={
            'logger':logger
        }
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir='./checkpoint',**models_args)
        trainer.fit(module, *loader)
        
    def run(self,**args):
        for i in range(len(self.content_layers)):
            print("计算从%s到%s层的内容损失"%(1,i+1))
            self.train(content_layers=self.content_layers[0:i+1])
        for i in range(len(self.content_layers)):
            print("计算从%s层的内容损失"%(i+1))
            self.train(content_layers=self.content_layers[i])
        for i in range(len(self.style_layers)):
            print("计算从%s到%s层的风格损失"%(1,i+1))
            self.train(style_layers=self.style_layers[0:i+1])
        for i in range(len(self.style_layers)):
            print("计算从%s层的风格损失"%(i+1))
            self.train(style_layers=self.style_layers[i])
