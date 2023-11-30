import clip
import torch
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.optim import Adam
from torchvision.utils import save_image
from argparse import ArgumentParser
class Clip:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        #self.model.eval()
        self.transfroms=transforms.Compose([
            transforms.Resize([224,224]),
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
parser = ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to input directory')
parser.add_argument('--save_path', type=str, help='Path to save the results')

opts = parser.parse_args()
model=Clip()
dataset=Data(opts.data_path)
test_dataloader = DataLoader(dataset,
                              batch_size=64,
                              shuffle=False,
                              num_workers=8,
                              drop_last=True)
features=[]
with torch.no_grad():
    for index,batch in enumerate(test_dataloader):
        print(index)
        batch=batch.cuda()
        feature=model.encode_img(batch)
        features.append(feature.cpu())
    features=torch.cat(features,dim=0)
    print(features.shape)
    features=features.numpy()
    np.save(opts.save_path,features)