import torch,os
from torch.utils import data
from torchvision import transforms
from PIL import Image
from loaders.styleLoader import styleLoader

class adainLoader(styleLoader):
    def __init__(self, root_dir,style_path , transform=None, augment_ratio=10) -> None:
        super().__init__(root_dir, transform, augment_ratio)
        try:
            self.style_img = self.transform(Image.open(style_path))
        except Exception as e:
            print('catch exception when load style image:%s ,exception message:%s '%(style_path,e))
        
    def __getitem__(self, index) :
        idx = index % len(self.images)
        while(True):
            try:
                img = self.transform(
                    Image.open(os.path.join(self.root_dir, self.images[idx]))
                )
            except Exception as e:
                print('catch exception when load image:%s ,exception message:%s '%(self.images[idx],e))
                idx = (idx+1) % len(self.images)
                continue
            break
        return (img,self.style_img)
    