import torch,numpy
import torch.nn.functional as F
import  torchvision.transforms as transforms
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import densenet121,vgg16
from torchvision.models._utils import IntermediateLayerGetter
from loaders import *
from utils import *
import utils

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return F.relu(self.conv(x)+x)

class ImfwNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # 下采样
        downsample = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )
        res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 9, 1)
        )
        self.model = nn.Sequential(
            downsample,
            res_blocks,
            upsample
        )

    def forward(self, x):    
        return self.model(x)
        

class FWNetModule(pl.LightningModule):
    def __init__(self,**args) -> None:
        super().__init__()
        args_v = {
            'content_weight': 1e3,
            'style_weight': 1,
            'tv_weight': 1e-5,
            'lr':1e-3,
            'automatic_optimization': False,
            'content_layers': ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3'],
            'style_layers': ['layer1_2', 'layer2_2', 'layer3_3', 'layer4_3', 'layer5_3'],
            'train_epochs':100,
            'test_image_path':'./data/MSNet/train/trans.jpg',
            'style':None,
        }
        for key in args_v.keys():
            if key in args.keys():
                setattr(self,key,args[key])
            else:
                setattr(self,key,args_v[key])
        self.style = nn.Parameter(self.style)
        self.test_img = nn.Parameter(utils.load_image(self.test_image_path,shape=(512,512)))
        self.fwNet = ImfwNet()
        vgg = vgg16(pretrained=True).features
        vgg.eval()
        self.vgg = vgg
        self.feature_net = IntermediateLayerGetter(vgg,{
            '3':'layer1_2',
            '8':'layer2_2',
            '15':'layer3_3',
            '22':'layer4_3',
            '29':'layer5_3'
        })
        self.trans = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.CenterCrop(512),
                transforms.ToTensor(),  # 转为0-1的张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        style_features = self.feature_net(self.style)
        self.style_features = nn.ParameterDict({key :nn.Parameter(style_features[key]) for key in style_features})
        self.style_grams = nn.ParameterDict({layer: nn.Parameter(gram_matrix(self.style_features[layer])) for layer in self.style_features})
    
    def forward(self, batch, batch_index) :
        x = batch
        transformed_images = self.fwNet(x).clamp(-2.1, 2.7)
        transformed_features = self.feature_net(transformed_images)
        content_features = self.feature_net(x)
        # 内容损失
        # 使用F.mse_loss函数计算预测(transformed_images)和标签(content_images)之间的损失
        content_loss = 0
        for layer in self.style_grams:
            if layer in self.content_layers:
                content_loss += self.content_weight * F.mse_loss(
                    transformed_features[layer], content_features[layer])
        content_loss =  content_loss / len(self.content_layers)
        # 全变分损失
        # total variation图像水平和垂直平移一个像素，与原图相减
        # 然后计算绝对值的和即为tv_loss
        _tv_loss = tv_loss(transformed_images, self.tv_weight) 
        # 风格损失
        style_loss = 0
        for layer in self.style_grams:
            if layer in self.style_layers:
                transformed_gram = gram_matrix(transformed_features[layer])
                # 是针对一个batch图像的Gram
                style_gram = self.style_grams[layer]
                # 是针对一张图像的，所以要扩充style_gram
                # 并计算计算预测(transformed_gram)和标签(style_gram)之间的损失
                style_loss += self.style_weight * F.mse_loss(transformed_gram,
                                    style_gram.expand_as(transformed_gram))
        style_loss = style_loss / len(self.style_layers)
        return style_loss , content_loss , _tv_loss
    
    def training_step(self, batch, batch_index):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self(batch,batch_index)
        self.log_dict({
            'loss': loss[0]+loss[1]+loss[2],
            'style_loss': loss[0],
            '_tv_loss': loss[2],
            'content_loss': loss[1]
        }, prog_bar=True)
        print('finished log loss')
        loss[0].backward(retain_graph=True)
        loss[1].backward(retain_graph=True)
        loss[2].backward()
        opt.step()
        
    def on_train_batch_end(self, outputs, batch, batch_idx,*args) -> None:
        torch.cuda.empty_cache()
        if batch_idx%self.train_epochs == self.train_epochs - 1:
            with torch.no_grad():
                title = 'epoch %s'%(int((batch_idx+1)/self.train_epochs))
                target = self.fwNet(self.test_img)
                utils.show_tensor(target,utils.show_image,title)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.fwNet.parameters(), self.lr)
        
        
if __name__ == '__main__':
    dense=densenet121()