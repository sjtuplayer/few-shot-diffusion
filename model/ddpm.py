import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torchvision.utils import save_image
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def print_range(x):
    return (round(float(x.min()),2),round(float(x.mean()),2),round(float(x.max()),2))
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
def my_unnormalize_to_zero_to_one(t):
    return (t-t.min())/(t.max()-t.min())
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels =3
        self.self_condition = False

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        #print(betas)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)          #显存大量占用
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    # def model_predictions_merge(self, x, t, x_self_cond = None, clip_x_start = False):
    #     # 中间的model_output随机进行batch内的线性组合
    #     model_output0 = self.model(x, t, x_self_cond)
    #     b=x.size(0)
    #     if t[0]>100:
    #         model_output = torch.zeros_like(model_output0).to(x.device).float()
    #         for i in range(b):
    #             z=torch.rand(b)
    #             z=z/z.sum()
    #             for j in range(b):
    #                 model_output[i]+=z[j]*model_output0[j]
    #     else:
    #         model_output = model_output0
    #     maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
    #
    #     if self.objective == 'pred_noise':
    #         pred_noise = model_output
    #         x_start = self.predict_start_from_noise(x, t, pred_noise)
    #         x_start = maybe_clip(x_start)
    #
    #     elif self.objective == 'pred_x0':
    #         x_start = model_output
    #         x_start = maybe_clip(x_start)
    #         pred_noise = self.predict_noise_from_start(x, t, x_start)
    #     return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start



    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True,loss_fn=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, model_variance, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        if loss_fn is not None and t<200:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                loss=loss_fn(x_in)
                grad= torch.autograd.grad(loss, x_in)[0]
                print(loss,grad.mean())
            model_mean=model_mean-grad*model_variance*0.5
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_sample_loop(self, shape=None,loss_fn=None,img=None,start_t=None,condition=None):
        # batch, device = shape[0], self.betas.device
        # img = torch.randn(shape, device=device)
        if start_t is None:
            start_t=self.num_timesteps
        if img is None:
            batch, device = shape[0], self.betas.device
            img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, start_t)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = condition if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond,loss_fn=loss_fn)
            # if t>start_t-20:
            #     print(img.min(),img.max(),x_start.min(),x_start.max())
            #     save_image(x_start, 'results/%d-x0.jpg' % t, normalize=True)
            #     save_image(img, 'results/%d.jpg'%t,normalize=True)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True,loss_fn=None,sample_step=None,max_step=None,min_step=None,start_img=None,return_middle=False,condition=None,guid=None,middle_step=0,guid_step=700,style_enhance=0,filter_size=8):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if sample_step is not None:
            sampling_timesteps=sample_step
        if max_step is None:
            max_step=total_timesteps
        if min_step is None:
            min_step=-1
        times = torch.linspace(min_step, max_step - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        if start_img is not None:
            #print(start_img.min(), start_img.max())
            if start_img.min()>=0 and start_img.max()<=1:
                img=start_img*2-1
            else:
                img=start_img
            batch=img.size(0)
        else:
            img = torch.randn(shape, device=device)
        x_start = None
        if guid is not None:
            filter_N = filter_size
            shape = (img.size(0), 3,img.size(-1),img.size(-1))
            shape_d = (img.size(0), 3, img.size(-1) // filter_N, img.size(-1) // filter_N)
            down = Resizer(shape, 1 / filter_N).cuda()
            up = Resizer(shape_d, filter_N).cuda()
        middle_img=None
        self_cond=condition
        for iter,(time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            #self_cond = x_start if self.self_condition else None
            if time_next>=middle_step:
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)
            else:
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, None, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            #sigma=0
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if time_next>guid_step and guid is not None:
                tmp_noise = self.p_losses(guid, torch.ones(len(img)).long().to(device) * time_next, return_x=True)
                tmp_img, tmp_x_start = self.p_sample(tmp_noise, time_next, self_cond, loss_fn=None)
                for ii in range(style_enhance-1):
                    tmp_noise = self.p_losses(tmp_x_start, torch.ones(len(img)).long().to(device) * time_next, return_x=True)
                    tmp_img, tmp_x_start = self.p_sample(tmp_noise, time_next, self_cond, loss_fn=None)
                # tmp_noise = self.p_losses(tmp_x_start, torch.ones(len(img)).long().to(device) * time_next,return_x=True)
                # tmp_img, tmp_x_start = self.p_sample(tmp_noise, time_next, self_cond, loss_fn=None)
                img = img - up(down(img)) + up(down(tmp_img))
            #save_image(pred_noise,'results/test/%d.jpg'%iter,normalize=True)
            if return_middle and (iter%5==0):
                if middle_img is None:
                    middle_img = my_unnormalize_to_zero_to_one(x_start.clone())
                else:
                    middle_img=torch.cat((middle_img,my_unnormalize_to_zero_to_one(x_start.clone())),dim=0)
            #save_image(img,'results/%d.jpg'%time,normalize=True)
        if min_step==-1:
            img = unnormalize_to_zero_to_one(img)
        #print('img range', img.min(), img.max())
        if return_middle:
            return img,middle_img
        return img

    def sample(self, batch_size = 16,loss_fn=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size),loss_fn)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    def p_losses(self, x_start, t, noise = None,return_x=False,x_self_cond=None):
        #输入的x_start属于(-1,1)
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        if x_start.min()>=0 and x_start.max()<=1:
            x_start=x_start*2-1
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        if return_x:
            return x
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()
    def few_shot_p_losses(self, x_start, t, return_x,noise = None,x_self_cond=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        if return_x:
            return x
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return x,loss.mean(-1)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)
    def few_shot_forward(self,img, *args,step=0,max_step=800,t=None,return_x=False, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # t = torch.randint(600, self.num_timesteps, (1,), device=device).long().repeat(b)
        if t is None:
            t = torch.randint(step, max_step, (b,), device=device).long()
        # t=torch.ones_like(t).cuda().long()*900
        img = normalize_to_neg_one_to_one(img)
        return t,self.few_shot_p_losses(img, t, return_x,*args,**kwargs)
    def batch_p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    def batch_p_sample(self, x, t, x_self_cond = None, clip_denoised = True,loss_fn=None,return_model_mean=False):
        b, *_, device = *x.shape, x.device
        batched_times = t
        model_mean, _, model_log_variance, x_start = self.batch_p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        if return_model_mean:
            return x_start,model_mean
        return x_start