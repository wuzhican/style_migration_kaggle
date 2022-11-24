import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch,os
import numpy as np

if __name__ != '__main__':
    plt.ion()

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b, c, h*w)
    tensor_t = tensor.transpose(1, 2)
    gram = torch.matmul(tensor, tensor_t)/(b*c*h*w)
    return gram


def tv_loss(y, tv_weight=1e-5):
    res = torch.sum(tv_weight*torch.abs(y[:, :, :, :-1]-y[:, :, :, 1:])) + \
        torch.sum(tv_weight*torch.abs(y[:, :, :-1, :]-y[:, :, 1:, :]))
    return res


def load_image(image_path, shape=None):
    image = Image.open(image_path)
    size = image.size
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image = in_transform(image)
    image = image[:3, :, :].unsqueeze(dim=0)
    return image

def show_pil(img,title=None):
    print(title)
    img = img.clone().detach().cpu()
    un = UnNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    to_img = transforms.ToPILImage()
    img = to_img(un(img).clamp(0,1))
    img.show()
    

def show_image(img,title=None):
    print(title)
    img = img.clone().detach().cpu()
    un = UnNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    img = un(img).clamp(0,1).data.numpy()
    img = img.transpose(1,2,0)
    plt.figure(title)
    plt.imshow(img)
    plt.pause(0.01)

def show_tensor(image:torch.Tensor,show_image = show_image,title=None):
    if len(image.size())==3:
        show_image(image,title)
    elif len(image.size()) == 4:
        for i in range(image.size()[0]):
            show_image(image[i],title+'_%s'%(i))
    else:
        raise ValueError("the tensor size is not in [3D,4D]")

class UnNormalize(object):
    def __init__(self,mean,std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
 
    def __call__(self,tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        res = None
        if len(tensor.size())==3:
            tmp = []
            for i in range(3):
                tmp.append(tensor[i] * self.std[i] + self.mean[i])
            res = torch.stack(tmp)
        elif len(tensor.size()) == 4:
            tmp1 = []
            for i in range(tensor.size()[0]):
                tmp2 = []
                for j in range(3):
                    tmp2.append(tensor[i][j] * self.std[j] + self.mean[j])
                tmp1.append(torch.stack(tmp2))
            res = torch.stack(tmp1)
        else:
            print('the tensor shape is not correct')
        return res

# 定义一个将标准化后的图像转化为便于利用matplotlib可视化的函数
def im_convert(tensor):
    '''
    将[1, c, h, w]维度的张量转为[h, w, c]的数组
    因为张量进行了表转化，所以要进行标准化逆变换
    '''
    tensor = tensor.cpu()
    image = tensor.data.numpy().squeeze() # 去除batch维度的数据
    image = image.transpose(1, 2, 0) # 置换数组维度[c, h, w]->[h, w, c]
    # 进行标准化的逆操作
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1) # 将图像的取值剪切到0-1之间
    return image

def GetResumePath(data_save_root,):
    max_record = {-100:'-100'}
    max, max_tag = -100, '-100'
    save_dir = data_save_root
    for i in os.popen('ls %s' % (save_dir)):
        assert i.find('version') != -1
        t = i.replace('version_', '').replace('\n', '')
        max_record[int(t)] = t
        if(int(t) > max):
            max, max_tag = int(t), t
    for t in range(max,-1,-1):
        if os.popen('ls %s'%os.path.join(save_dir, 'version_%s' % (max_tag))).read().find('checkpoints') != -1:
            for i in os.popen('ls -r %s' % (os.path.join(save_dir, 'version_%s' % (max_tag), 'checkpoints'))):
                return os.path.join(save_dir, 'version_%s' % (max_tag), 'checkpoints', i.replace('\n', ''))
    return None

