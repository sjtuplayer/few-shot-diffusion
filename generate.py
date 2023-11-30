import torch
from denoising_diffusion_pytorch import Unet
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image,make_grid
import os
from argparse import ArgumentParser
from torchvision import transforms
import torchvision
#diffusion直接采样
parser = ArgumentParser()
parser.add_argument('--img_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--save_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--ckpt_path', type=str,default=None, help='Path to ckpt')
opts = parser.parse_args()
image_size=256
model = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=False,
).cuda()
model.prepare(two_stage_step=300,style_condition=True)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=len(self.file_names)
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image
    def __len__(self):
        return self.l
device='cuda'
batch_size=8
guid_step=400
filter_size=8
diffusion.load_state_dict(torch.load(opts.ckpt_path),strict=False)
real_data=Train_Data(opts.img_path)
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=8,
                                   drop_last=False)
os.makedirs(opts.save_path,exist_ok=True)
with torch.no_grad():
    for batch_idx,imgs in enumerate(real_dataloader):
        imgs=imgs.to(device)
        noise_step=600
        t=torch.ones(len(imgs)).long().to(device)*noise_step
        noises = diffusion.p_losses(imgs, t,return_x=True)
        _,mid_imgages = diffusion.ddim_sample(imgs.shape, sample_step=25, max_step=noise_step,
                                                                       return_middle=True,start_img=noises,condition=imgs,guid=imgs,guid_step=guid_step,filter_size=filter_size)
        save_image(torch.cat([imgs,mid_imgages[imgs.size(0)*2:imgs.size(0)*3]],dim=0),os.path.join(opts.save_path,"%d.png"%batch_idx),nrow=imgs.size(0))
        save_image(mid_imgages, os.path.join(opts.save_path, "%d-mid.png" % batch_idx),nrow=imgs.size(0))
