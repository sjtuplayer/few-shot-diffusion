import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
# Decide which device we want to run on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cuda'

class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas-gt)**self.p)
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.resize = resize

    def forward(self, input, target, ignore_color=False):
        self.mean = self.mean.type_as(input)
        self.std = self.std.type_as(input)
        if ignore_color:
            input = torch.mean(input, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss =torch.zeros(input.size(0)).to(input.device)
        x = input
        y = target.repeat(x.size(0)//target.size(0),1,1,1)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.abs(x-y).mean(-1).mean(-1).mean(-1)
        return loss



class VGGStyleLoss(torch.nn.Module):
    def __init__(self, transfer_mode, resize=True,style_img=None):
        super(VGGStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        blocks = []
        if transfer_mode == 0:  # transfer color only
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
        else: # transfer both color and texture
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
            blocks.append(vgg.features[9:16].eval())
            blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.style_img=style_img
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram
    def get_gram_matrix(self,input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        x=self.blocks[0](input)
        x=self.blocks[1](x)
        gram=self.gram_matrix(x)
        gram=gram.unsqueeze(1).repeat(1,3,1,1)
        return gram
    def forward(self, input, target=None):
        if target is None:
            target=self.style_img.clone()
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # print(input.min(),input.max())
        # print(target.min(),target.max())
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = torch.zeros(input.size(0)).to(input.device)
        x = input
        y = target.repeat(x.size(0) // target.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            gm_x = self.gram_matrix(x)
            gm_y = self.gram_matrix(y)
            loss += ((gm_x-gm_y)**2).sum(dim=-1).sum(dim=-1)
        return loss

class MyStyleLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(MyStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        print(blocks)
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def get_matrix(self, x,last_x):
        (b, ch, h, w) = last_x.size()
        x=F.interpolate(x,size=(h,w),mode='bilinear')
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        last_x=self.blocks[0](x)
        last_y=self.blocks[0](y)
        for block in self.blocks[1:]:
            x = block(last_x)
            y = block(last_y)
            gm_x = self.get_matrix(x,last_x)
            gm_y = self.get_matrix(y,last_y)
            loss += torch.sum((gm_x-gm_y)**2)
            last_x,last_y=x,y
        return loss
if __name__ == '__main__':
    net=MyStyleLoss().cuda()
    x1=torch.randn(1,3,128,128).float().cuda()
    x2=torch.randn(1,3,128,128).float().cuda()
    y=net(x1,x2)
    print(y.shape)
