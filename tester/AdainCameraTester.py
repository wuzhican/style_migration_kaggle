import models,utils,cv2,numpy,time,torch,threading
from tester.AbstractTester import AbstractTester
from torchvision import transforms

class AdainCameraTester(AbstractTester):
    args_v = {
        'video':0,
        'video_tag':'adain video',
        'style_path':'./data/styles/style5.jpeg',
        'transform': transforms.Compose([
            transforms.Lambda(lambda x:cv2.flip(x,1)),
            transforms.Lambda(lambda x:cv2.resize(x, (512,512))),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'to_numpy': transforms.Compose([
            # transforms.Lambda(lambda x:utils.UnNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(x)),
            transforms.Lambda(lambda x:x.clamp(0,1)),
            transforms.ToPILImage(),
            transforms.Lambda(lambda x:cv2.cvtColor(numpy.asarray(x),cv2.COLOR_RGB2BGR)),
            transforms.Lambda(lambda x:cv2.resize(x, (1280,720)))
        ])
    }
    
    def __init__(self,checkpoint_path,**args) -> None:
        super().__init__()
        for key in self.args_v:
            if key in args.keys():
                setattr(self,key,args[key])
            else:
                setattr(self,key,self.args_v[key])
        self.style_img = utils.load_image(self.style_path,shape=(256,256)).cuda()
        self.adainNet = models.AdainNetModule.load_from_checkpoint(checkpoint_path)
        self.cap = cv2.VideoCapture(self.video)
        self.adainNet.cuda()
        self.adainNet.eval()
        
    def run(self):
        while True:
            a = time.time()
            ret,frame = self.cap.read()
            if(ret):
                torch.cuda.empty_cache()
                img = self.transform(frame).unsqueeze(dim=0).cuda()
                target = self.adainNet.trans_image([img,self.style_img])[0].cpu()
                show_frame = self.to_numpy(target)
                seconds = time.time() - a
                cv2.putText(show_frame,"fps:%s"%(1/seconds),(5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow(self.video_tag,show_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                continue
            else:
                break
        cv2.destroyAllWindows()
        
