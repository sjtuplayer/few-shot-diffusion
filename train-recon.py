import torch
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--ckpt_path', type=str,default=None, help='Path to ckpt')
parser.add_argument('--save_dir', type=str,default='output-recon', help='Path to ckpt')
opts = parser.parse_args()
image_size=256
model_target = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=False,
).cuda()

model_target.prepare(style_condition=True,two_stage_step=300)
diffusion = GaussianDiffusion(
    model_target,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
ckpt=torch.load(opts.ckpt_path,map_location='cpu')
diffusion.load_state_dict(ckpt,strict=False)

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
batch_size =8
real_data=Train_Data(opts.data_path)
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
global_step=0
output_dir=opts.save_dir
os.makedirs(os.path.join(output_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(output_dir,'images'),exist_ok=True)
optimizer = Adam(diffusion.model.c_content_condition.parameters(), lr = 1e-4, betas =(0.9, 0.99))
for epoch in range(100):
    for batch_idx,batch in enumerate(real_dataloader):
        if global_step%10==0:
            print(global_step,len(real_dataloader))
        real_image=batch.cuda()
        condition = real_image
        t, (x, loss_diffusion) = diffusion.few_shot_forward(real_image, x_self_cond=condition,max_step=1000)
        loss_diffusion=loss_diffusion.mean()
        loss_diffusion.backward()
        optimizer.step()
        optimizer.zero_grad()

        if global_step%200==0:
            noise_step = 800
            t = torch.ones(len(real_image)).long().to('cuda') * noise_step
            noises = diffusion.p_losses(real_image, t, return_x=True)
            output_image_cond=diffusion.ddim_sample(real_image.shape,condition=condition,sample_step=50,max_step=noise_step)
            save_image(torch.cat([condition,noises,output_image_cond],dim=0),os.path.join(output_dir,'images/%d.png'%global_step),nrow=batch_size)
            torch.save(diffusion.state_dict(),os.path.join(output_dir,'models/%d.pt'%global_step))
        global_step += 1