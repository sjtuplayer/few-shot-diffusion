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
import numpy as np
import clip
from style_loss import loss
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--source_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--target_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--save_dir', type=str,default='output-whole', help='Path to input directory')
parser.add_argument('--source_feature_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--target_feature_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--ckpt_path', type=str,default=None, help='Path to ckpt')
opts = parser.parse_args()
image_size=256
model_target = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=False,
).cuda()
model_target.prepare(two_stage_step=300,style_condition=True)
diffusion_target = GaussianDiffusion(
    model_target,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()
class Clip:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        #print(self.preprocess)
        self.transfroms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    def encode_text(self,text_input):
        return self.model.encode_text(clip.tokenize(text_input).to(self.device))
    def encode_img(self,img):
        return self.model.encode_image(self.transfroms(img))
    def forward(self,img,text):
        image = self.transfroms(img)
        text = clip.tokenize([text]).to(self.device)
        logits_per_image, logits_per_text = self.model(image, text)
        #probs = logits_per_image.softmax(dim=-1)
        return -logits_per_image
class Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224])
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
class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l=max(10000,len(self.file_names))
    def __getitem__(self, idx):
        idx=idx%len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l
batch_size =4
dir=opts.target_path
loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
style_imgs=[]
for i in os.listdir(dir):
    image = Image.open(os.path.join(dir, i)).convert('RGB')
    style_imgs.append(loader(image))
style_imgs=torch.stack(style_imgs,dim=0).cuda()
clip_model=Clip()
style_loss = loss.VGGStyleLoss(transfer_mode=1, resize=True).cuda()
train_data=Train_Data(opts.target_path)
real_data=Train_Data(opts.source_path)
features_source=torch.from_numpy(np.load(opts.source_feature_path)).cuda().mean(0)
features_target=torch.from_numpy(np.load(opts.target_feature_path)).cuda().mean(0)
feature_dir=(features_target-features_source).type(torch.HalfTensor).cuda().unsqueeze(0)
print(feature_dir.shape,feature_dir.dtype)
real_dataloader = DataLoader(real_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
real_dataloader_iter=iter(real_dataloader)
train_dataloader = DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)
diffusion_target.load_state_dict(torch.load(opts.ckpt_path),strict=True)


optimizer = Adam(filter(lambda p: p.requires_grad, diffusion_target.parameters()), lr = 1e-4, betas =(0.9, 0.99))
global_step=0
mse_loss=torch.nn.MSELoss(reduction='none')
mse_loss_reduce=torch.nn.MSELoss()
print(mse_loss(torch.zeros_like(feature_dir).cuda(),feature_dir).mean())
cos_loss=torch.nn.CosineSimilarity(dim=1)
save_dir=opts.save_dir
os.makedirs(os.path.join(save_dir,'models'),exist_ok=True)
os.makedirs(os.path.join(save_dir,'images'),exist_ok=True)
opts.beta_f=1
opts.beta_style=1
loss_diffusion=0
loss_diffusion2=0
loss_feature=0
loss_style=0
filter_N=4
for epoch in range(100):
    for batch_idx,batch in enumerate(train_dataloader):
        if batch_idx%10==0:
            print(batch_idx)
        image=batch.cuda()
        real_image = next(real_dataloader_iter, None)
        if real_image is None:
            real_dataloader_iter = iter(real_dataloader)
            real_image = next(real_dataloader_iter, None)
        real_image = real_image.cuda()
        condition=real_image
        # if random.random()<0.5:
        #     condition=None
        if global_step%2==0:
            with torch.no_grad():
                t, (x, _) = diffusion_target.few_shot_forward(real_image,step=300,x_self_cond=condition)
                feature_source=clip_model.encode_img(real_image)
            x_start_target = (diffusion_target.batch_p_sample(x, t, x_self_cond=condition) + 1) / 2
            feature_target = clip_model.encode_img(x_start_target)
            if opts.beta_f == 0 and opts.beta_style == 0:
                x_start_target=x_start_target.detach()
                feature_target=feature_target.detach()
            if opts.beta_f != 0:
                feature_source_to_target = feature_source + feature_dir.repeat(batch_size, 1)
                loss_feature = mse_loss(feature_target, feature_source_to_target).mean(-1)*opts.beta_f
            if opts.beta_style!=0:
                loss_style = torch.zeros(x_start_target.size(0)).cuda()
                for i in range(x_start_target.size(0)):
                    loss_style[i] = style_loss(x_start_target[i:i + 1].repeat(style_imgs.size(0), 1, 1, 1),
                                               style_imgs).mean()
                loss_style = loss_style * opts.beta_style
            dishu = 20
            alpha = dishu ** (t / 1000)
            loss_style = (alpha * loss_style).mean()
            loss_feature = (alpha * loss_feature).mean()
            loss = loss_feature + loss_style
            if opts.beta_f!=0 or opts.beta_style!=0:
                loss.backward()
            t2, (x2, loss_diffusion) = diffusion_target.few_shot_forward(image, t=t,x_self_cond=None)
            loss_diffusion = ((dishu ** 0.9 - alpha) * loss_diffusion).mean()
            loss_diffusion.backward()
        else:
            t = torch.randint(0, 300, (batch_size,)).long().cuda()
            t2, (x2, loss_diffusion2) = diffusion_target.few_shot_forward(image, t=t,x_self_cond=None)
            dishu = 20
            alpha = dishu ** (t / 1000)
            loss_diffusion2 = ((dishu ** 0.9 - alpha) * loss_diffusion2).mean()/5
            loss=loss_diffusion2
            loss.backward()
        if global_step%10==0 and global_step!=0:
            print('step=%d,dif1=%.4f, dif2=%.4f, fea=%.4f, sty=%.4f'%(global_step,float(loss_diffusion2),float(loss_diffusion),float(loss_feature),float(loss_style)))
        if global_step%2==0:
            optimizer.step()
            optimizer.zero_grad()
        if global_step%50==0:
            noise_step = 600
            t = torch.ones(len(real_image)).long().to('cuda') * noise_step
            noises = diffusion_target.p_losses(real_image, t, return_x=True)
            sampled_images, sampled_middle_images = diffusion_target.ddim_sample(real_image.shape, sample_step=25,
                                                                                 return_middle=True, start_img=noises,
                                                                                 max_step=noise_step,
                                                                                 min_step=-1, condition=condition,
                                                                                 guid_step=300,guid=condition)
            save_image(torch.cat((real_image, noises, sampled_middle_images, sampled_images), dim=0),
                       os.path.join(save_dir, 'images/%d-sample.jpg' % global_step), nrow=batch_size, normalize=False)
            torch.save(diffusion_target.state_dict(), save_dir + '/models/%d.pth' % global_step)
        global_step += 1
