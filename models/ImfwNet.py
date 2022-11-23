import torch
import torch.nn.functional as F
import  torchvision.transforms as transforms
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import densenet121,vgg16
from torchvision.models._utils import IntermediateLayerGetter
from loaders import *
from utils import *
import utils

class MyModuleList(nn.ModuleList):
    def __init__(self, modules) -> None:
        super().__init__(modules)
    
    def forward(self,x):
        for i in range(self.__len__()):
            x = self.__getitem__(i)(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = MyModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        return F.relu(self.conv(x)+x)


class ImfwNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # 下采样
        downsample = MyModuleList([
            nn.Conv2d(3, 32, 9, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        ])
        res_blocks = MyModuleList([
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        ])
        upsample = MyModuleList([
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 9, 1)
        ])
        self.model = nn.Sequential(
            nn.Conv2d(3,128,1),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128,3,1),
            # nn.ConvTranspose2d(128,3,1),
            # nn.ReLU()
        )

    def forward(self, x):    
        return F.relu(self.model(x))
        

class FWNetModule(pl.LightningModule):
    
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
    }
    
    def __init__(self,style:torch.Tensor,**args) -> None:
        super().__init__()
        self.style = style
        for key in self.args_v.keys():
            if key in args.keys():
                setattr(self,key,args[key])
            else:
                setattr(self,key,self.args_v[key])
        self.save_hyperparameters()
        # print('init FWNetModule class ')
        self.fwNet = ImfwNet()
        vgg = vgg16(pretrained=True).features
        # vgg.eval()
        # vgg.classifier = nn.Sequential()
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
        
        # 内容表示的图层,均使用经过relu激活后的输出
        self.style_features = self.feature_net(self.style)
        # 为我们的风格表示计算每层的格拉姆矩阵，使用字典保存
        self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser
    
    def training_step(self, batch,batch_index):
        opt = self.optimizers()
        opt.zero_grad()
        if(str(self.device).find('cuda') != -1 and str(self.style.device) != str(self.device)):
            self.style = self.style.to(self.device)
            self.style_features = self.feature_net(self.style)
            self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}
        x = batch
        transformed_images = self.fwNet(x)
        
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
        # 3个损失加起来，梯度下降
        loss = style_loss + content_loss + _tv_loss
        if batch_index%self.train_epochs == self.train_epochs - 1:
            with torch.no_grad():
                test_img = self.trans(Image.open(self.test_image_path))
                title = 'epoch %s'%(int((batch_index+1)/self.train_epochs))
                target = self.fwNet(test_img)
                utils.show_tensor(target,utils.show_image,title)
        self.log('train_loss', loss, prog_bar=True)
        self.log('style_loss', style_loss, prog_bar=True)
        self.log('_tv_loss', _tv_loss, prog_bar=True)
        self.log('content_loss', content_loss, prog_bar=True)
        self.manual_backward(loss,retain_graph = True)
        opt.step()
        
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        torch.cuda.empty_cache()
        # if batch_idx%self.train_epochs == self.train_epochs - 1:
        #     with torch.no_grad():
        #         test_img = self.trans(Image.open(self.test_image_path)).to(self.device)
        #         title = 'epoch %s'%(int((batch_idx+1)/self.train_epochs))
        #         target = self.fwNet(test_img)
        #         utils.show_tensor(target,utils.show_image,title)
    
    def configure_optimizers(self):
        # print("start configure_optimizers with device: %s"%(self.device))
        if(str(self.device).find('cuda') != -1 and str(self.style.device) != str(self.device)):
            self.fwNet.to(self.device)
            opt= torch.optim.SGD(self.fwNet.parameters(), self.lr)
        # print("finish configure_optimizers")
        return opt
        
        
if __name__ == '__main__':
    dense=densenet121()