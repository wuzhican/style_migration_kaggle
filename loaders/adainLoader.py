import torch,os,random
from torch.utils import data
from torchvision import transforms
from PIL import Image
from loaders.styleLoader import styleLoader

class adainLoader(styleLoader):
    def __init__(self, root_dir,style_path , transform=None, augment_ratio=10) -> None:
        super().__init__(root_dir, transform, augment_ratio)
        self.style_path = style_path
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x:x.convert('RGB')),
            transforms.Resize((512,512)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])
        self.style_images = []
        for i in os.popen('ls %s'%(style_path)):
            i = i.replace('\n','')
            self.style_images.append(i)
        assert len(self.style_images) > 0
        assert len(self.images) > 0
        
    def __getitem__(self, index) :
        img = super().__getitem__(index)
        idx = (random.randint(0,len(self.style_images))) % len(self.style_images)
        while(True):
            try:
                style_img = self.transform(
                    Image.open(os.path.join(self.style_path, self.style_images[idx]))
                )
            except Exception as e:
                print('catch exception when load style image:%s ,exception message:%s '%(self.style_images[idx],e))
                idx = (idx+1) % len(self.images)
                continue
            break
        return (img,style_img)
    