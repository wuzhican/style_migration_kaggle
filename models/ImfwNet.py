import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import densenet121,vgg16
from loaders import *
from utils import *


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


class ImfwNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 下采样
        self.downsample = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x
        

class FWNetModule(pl.LightningModule):
    def __init__(self,style_wuzhican:torch.Tensor, content_weight:int=1,style_weight:int=1e5,automatic_optimization=True,lr=1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        print('init FWNetModule class ')
        self.automatic_optimization = automatic_optimization
        self.fwNet = ImfwNet()
        vgg = vgg16(pretrained=True)
        vgg.eval()
        vgg.classifier = nn.Sequential()
        self.vgg = vgg
        self.content_weight, self.style_weight, self.style, self.lr = content_weight, style_weight, style_wuzhican, lr
        self.feature_net = None
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser
    
    def training_step(self, batch,batch_index):
        print("start train batch_index:%s "%(batch_index))
        if(str(self.device).find('cuda') != -1 and str(self.style.device) != str(self.device)):
            # print('start convery device')
            self.style = self.style.to(self.device)
            self.vgg.to(self.device)
        
        if(self.feature_net==None):
            self.feature_net = InterMediateLayerGatter(self.vgg,{
                'features/3':'layer1_2',
                'features/8':'layer2_2',
                'features/15':'layer3_3',
                'features/22':'layer4_3',
            })
            # 内容表示的图层,均使用经过relu激活后的输出
            self.style_features = self.feature_net(self.style)
            # 为我们的风格表示计算每层的格拉姆矩阵，使用字典保存
            self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}

        print("start zero grad batch_index:%s "%(batch_index))
        opt=self.optimizers()
        opt.zero_grad()
        x = batch
        print("start calauate transformed_images batch_index:%s "%(batch_index))
        transformed_images = self.fwNet(x).clamp(-2.1, 2.7)
        
        print("start calauate transformed_features batch_index:%s "%(batch_index))
        transformed_features = self.feature_net(x)
        print("start calauate content_features batch_index:%s "%(batch_index))
        content_features = self.feature_net(transformed_images)

        # 内容损失
        # 使用F.mse_loss函数计算预测(transformed_images)和标签(content_images)之间的损失
        print("start calauate content_loss batch_index:%s "%(batch_index))
        content_loss = F.mse_loss(
            transformed_features['layer3_3'], content_features['layer3_3'])
        content_loss = self.content_weight*content_loss
        # print("batch %s: content_loss:%s "%(batch_index,content_loss))

        # 全变分损失
        # total variation图像水平和垂直平移一个像素，与原图相减
        # 然后计算绝对值的和即为tv_loss
        print("start calauate _tv_loss batch_index:%s "%(batch_index))
        _tv_loss = tv_loss(transformed_images)
        # print("batch %s: _tv_loss:%s "%(batch_index,_tv_loss))

        # 风格损失
        print("start calauate style_loss batch_index:%s "%(batch_index))
        style_loss = 0
        transformed_grams = {
            layer: gram_matrix(transformed_features[layer]) for layer in transformed_features.keys()
        }
        for layer in self.style_grams:
            transformed_gram = transformed_grams[layer]
            # 是针对一个batch图像的Gram
            style_gram = self.style_grams[layer]
            # 是针对一张图像的，所以要扩充style_gram
            # 并计算计算预测(transformed_gram)和标签(style_gram)之间的损失
            style_loss += F.mse_loss(transformed_gram,
                                style_gram.expand_as(transformed_gram))
        style_loss = self.style_weight * style_loss
        # print("batch %s: style_loss:%s "%(batch_index,style_loss))
        # 3个损失加起来，梯度下降
        print("start calauate loss batch_index:%s "%(batch_index))
        loss = style_loss + _tv_loss + content_loss
        print("start log loss value batch_index:%s "%(batch_index))
        self.log('train_loss', int(loss), prog_bar=True)
        self.log('style_loss', int(style_loss), prog_bar=True)
        self.log('_tv_loss', int(_tv_loss), prog_bar=True)
        self.log('content_loss', int(content_loss), prog_bar=True)
        print("start manual_backward batch_index:%s "%(batch_index))
        self.manual_backward(loss,retain_graph = True)
        print("start step batch_index:%s "%(batch_index))
        opt.step()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.fwNet.parameters(), self.lr)
        
        
if __name__ == '__main__':
    dense=densenet121()