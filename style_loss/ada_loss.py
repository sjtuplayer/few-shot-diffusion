# import sys
# sys.path.append(".")
# sys.path.append("..")
import torch
import torch.nn as nn
import itertools
import torch.optim as optim
from . import networks as ada_n
from torchvision import transforms
image_encoder_path = "style_loss/vgg_normalised.pth"

class AdaAttNModel(nn.Module):

    def __init__(self):
        super(AdaAttNModel, self).__init__()
        self.device = torch.device("cuda:0")
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        self.resize_512=transforms.Resize((128,128))
        image_encoder.load_state_dict(torch.load(image_encoder_path))
        enc_layers = list(image_encoder.children())
        # enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(self.device), device_ids=[0, 1])
        # enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(self.device), device_ids=[0, 1])
        # enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(self.device), device_ids=[0, 1])
        # enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(self.device), device_ids=[0, 1])
        # enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(self.device), device_ids=[0, 1])
        enc_1 = nn.Sequential(*enc_layers[:4]).to(self.device)
        enc_2 = nn.Sequential(*enc_layers[4:11]).to(self.device)
        enc_3 = nn.Sequential(*enc_layers[11:18]).to(self.device)
        enc_4 = nn.Sequential(*enc_layers[18:31]).to(self.device)
        enc_5 = nn.Sequential(*enc_layers[31:44]).to(self.device)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'transformer']
        parameters = []
        self.max_sample = 64 * 64

        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        # self.optimizers = None
        self.lambda_content = 0
        self.lambda_global = 10
        self.lambda_local = 3
        # self.optimizer = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

        self.loss_names = ['content', 'global', 'local']
        self.criterionMSE = torch.nn.MSELoss().to(self.device)
        # self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
        # self.optimizers.append(self.optimizer_g)
        self.loss_global = torch.tensor(0., device=self.device)
        self.loss_local = torch.tensor(0., device=self.device)
        self.loss_content = torch.tensor(0., device=self.device)

    def set_input(self, c, s):
        self.s = s.to(self.device)  # s.shape:[1, 3, 512, 512]
        self.c = c.to(self.device)

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(ada_n.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(ada_n.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return ada_n.mean_variance_norm(feats[last_layer_idx])
    def forward(self,cs):
        self.cs = self.resize_512(cs).to(self.device)
        return self.compute_losses()
    # def forward(self):
    #     self.c_feats = self.encode_with_intermediate(self.c)
    #     self.s_feats = self.encode_with_intermediate(self.s)
    #     if self.opt.skip_connection_3:
    #         c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], self.get_key(self.c_feats, 2, self.opt.shallow_layer),
    #                                                self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
    #     else:
    #         c_adain_feat_3 = None
    #     cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
    #                               self.get_key(self.c_feats, 3, self.opt.shallow_layer),
    #                               self.get_key(self.s_feats, 3, self.opt.shallow_layer),
    #                               self.get_key(self.c_feats, 4, self.opt.shallow_layer),
    #                               self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
    #     self.cs = self.net_decoder(cs, c_adain_feat_3)

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(ada_n.mean_variance_norm(stylized_feats[i]),
                                                       ada_n.mean_variance_norm(self.c_feats[i]))

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = ada_n.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = ada_n.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_local = torch.tensor(0., device=self.device)
        if self.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i)
                s_key = self.get_key(self.s_feats, i)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i], std * ada_n.mean_variance_norm(self.c_feats[i]) + mean)

    def compute_losses(self):
        self.c_feats = self.encode_with_intermediate(self.c)
        self.s_feats = self.encode_with_intermediate(self.s)
        stylized_feats = self.encode_with_intermediate(self.cs)
        #self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.loss_content = self.loss_content * self.lambda_content
        self.loss_local = self.loss_local * self.lambda_local
        self.loss_global = self.loss_global * self.lambda_global
        #print('loss_style:%.2f,loss_content:%.2f,loss_content2:%.2f'%(float(self.loss_global.cpu().detach()),
                                                   #float(self.loss_local.cpu().detach()),float(self.loss_content.cpu().detach())))
        return self.loss_global + self.loss_local
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_global + self.loss_local
        loss.backward()
        self.optimizer_g.step()

