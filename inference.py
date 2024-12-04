from model import BiSeNet
import torch
import os.path as osp
import torchvision.transforms as transforms


class FaceParser():
    def __init__(self,model_name = "79999_iter.pth",n_classes = 19,device="cuda") -> None:
        self.net = BiSeNet(n_classes=n_classes)
        self.net.to(device)
        save_pth = osp.join('res/cp', model_name)
        self.net.load_state_dict(torch.load(save_pth))

    def create_mask(self,image):
        self.net.eval()
        to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = self.net(img)[0]
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
        return parsing
    