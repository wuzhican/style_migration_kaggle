import models,utils
import matplotlib.pyplot as plt
from torchvision import transforms

checkpoint_path = utils.GetResumePath('./data/lightning_logs/SMNet')
smnet = models.SMNet.load_from_checkpoint(
    checkpoint_path, style=utils.load_image('./data/style.jpeg'))

show_image = smnet.input_image

toImg = transforms.ToPILImage()
un = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

target_image = toImg(show_image[0])
target_image.show()
