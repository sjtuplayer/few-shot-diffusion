# Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption (ICCV 2023)

###  [Paper](https://arxiv.org/abs/2309.03729)
<!-- <br> -->
[Teng Hu](https://github.com/sjtuplayer), [Jiangning Zhang](https://zhangzjn.github.io/), [Liang Liu](https://scholar.google.com/citations?hl=zh-CN&user=Kkg3IPMAAAAJ), [Ran Yi](https://yiranran.github.io/), Siqi Kou, [Haokun Zhu](https://github.com/zwandering), [Xu Chen](https://scholar.google.com/citations?hl=zh-CN&user=1621dVIAAAAJ), [Yabiao Wang](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), [Chengjie Wang](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ) and [Lizhuang Ma](https://dmcv.sjtu.edu.cn/) 
<!-- <br> -->

![image](imgs/framework.jpg)

## Abstract
>Training a generative model with limited number of samples is a challenging task. Current methods primarily rely on few-shot model adaption to train the network. However, in scenarios where data is extremely limited (less than 10), the generative network tends to overfit and suffers from content degradation. To address these problems, we propose a novel phasic content fusing few-shot diffusion model with directional distribution consistency loss, which targets different learning objectives at distinct training stages of the diffusion model. Specifically, we design a phasic training strategy with phasic content fusion to help our model learn content and style information when t is large, and learn local details of target domain when t is small, leading to an improvement in the capture of content, style and local details. Furthermore, we introduce a novel directional distribution consistency loss that ensures the consistency between the generated and source distributions more efficiently and stably than the prior methods, preventing our model from overfitting. Finally, we propose a cross-domain structure guidance strategy that enhances structure consistency during domain adaptation. Theoretical analysis, qualitative and quantitative experiments demonstrate the superiority of our approach in few-shot generative model adaption tasks compared to state-of-the-art methods

This work has been accpected by ICCV 2023.


## Visualization Results

The follows are the results trained on Cartoon and Van Gogh painting dataset:

![image](imgs/visualization%20result.jpg)

## Todo (Latest update: 2023/09/08)
- [x] **Release the training code
- [ ] **Release the pretrained models and training data


## The Pretrained Models

Coming Soon!


## Training Steps


### Overview

This project is based on [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

We show the training method of our model here.
The training process can be divided into three steps:

(0) Install the dependency:
```
pip install denoising_diffusion_pytorch
```


(1) Train the basic diffusion model on the source dataset;

(2) Train the phasic content fusing module on the source dataset;

(3) Train the whole model on both the source and target dataset;

### (1) Train the basic diffusion model on the source dataset


To train the basic diffusion model, you can run the following code
```
python3 train.py --data_path=path_to_dataset 
```

### (2) Train the phasic content fusing diffusion model on the source dataset

After the basic diffusion model is trained, you can train phasic content fusing module by:
```
python3 train-recon.py --data_path=path_to_dataset --ckpt=path_to_checkpoint
```

### (3) Train the whole model on both the source and target dataset

Finally, train the whole model on both the source and target dataset:
```
python3 train-whole.py --source_path=path_to_source_dataset --target_path=path_to_target_dataset --ckpt_path=path_to_checkpoint
```

## Citation

If you find this code helpful for your research, please cite:

```
@inproceedings{hu2023phasic,
      title={Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption}, 
      author={Teng Hu and Jiangning Zhang and Liang Liu and Ran Yi and Siqi Kou and Haokun Zhu and Xu Chen and Yabiao Wang and Chengjie Wang and Lizhuang Ma},
      booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
      year={2023}
      }
```