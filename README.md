# Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption

## Abstract
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

## Train the basic diffusion model on the source dataset


To train the basic diffusion model, you can run the following code
```
python3 train.py --data_path=path_to_dataset 
```

## Train the phasic content fusing diffusion model on the source dataset

After the basic diffusion model is trained, you can train phasic content fusing module by:
```
python3 train-recon.py --data_path=path_to_dataset --ckpt=path_to_checkpoint
```

## Train the whole model on both the source and target dataset

Finally, train the whole model on both the source and target dataset:
```
python3 train-whole.py --source_path=path_to_source_dataset --target_path=path_to_target_dataset --ckpt_path=path_to_checkpoint
```
