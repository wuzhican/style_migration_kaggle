import models,utils
import matplotlib.pyplot as plt
from torchvision import transforms

# fwnet=models.FWNetModule(utils.load_image('./data/style.jpeg'))
checkpoint_path = utils.GetResumePath('./data')
fwnet = models.FWNetModule.load_from_checkpoint(
    checkpoint_path, style_wuzhican=utils.load_image('./data/style.jpeg'))

un = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
toImg = transforms.ToPILImage()

img=utils.load_image('./data/style1.jpg')[0]
target=toImg(un(fwnet.fwNet(img).clamp(-2.1, 2.7)))
target.show()



