<div align="center"> <h1>The Official Pytorch and JAX implementation of "Efficient-VDVAE: Less is more" <a href="">Arxiv preprint</a></h1> </div>  
  
<div align="center">    
  <a>Louay&nbsp;Hazami</a>     
  &emsp; <b>&middot;</b> &emsp;    
  <a>Rayhane&nbsp;Mama</a>     
  &emsp; <b>&middot;</b> &emsp;    
  <a>Ragavan&nbsp;Thurairatnam</a>    
</div>    
<br>    
<br>   
 
[Efficient-VDVAE]() is a memory and compute efficient very deep hierarchical VAE. It converges faster and is more stable than current hierarchical VAE models. It also achieves SOTA likelihood-based performance on several image datasets.    
    
<p align="center">  
    <img src="images/unconditional_samples.png" width="1200">  
</p>  
  
## Pre-trained model checkpoints  
  
We provide checkpoints of pre-trained models on MNIST, CIFAR-10, Imagenet 32x32, Imagenet 64x64, CelebA 64x64, CelebAHQ 256x256 (5-bits and 8-bits), FFHQ 256x256 (5-bits and 8bits), CelebAHQ 1024x1024 and FFHQ 1024x1024 in the links in the table below. All provided models are the ones trained for table 4 of the [paper]().

<table align="center">
    <thead align="center">
        <tr>
            <th rowspan=2 align="center"> Dataset </th>
            <th colspan=2 align="center"> Pytorch </th>
            <th colspan=2 align="center"> JAX </th>
            <th rowspan=2 align="center"> Negative ELBO </th>
        </tr>
        <tr>
	        <th align="center"> Logs </th>
	        <th align="center"> Checkpoints </th>
	        <th align="center"> Logs </th>
	        <th align="center"> Checkpoints </th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="center">MNIST</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">79.09 nats</td>
        </tr>
        <tr>
            <td align="center">CIFAR-10</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">2.87 bits/dim</td>
        </tr>
        <tr>
            <td align="center">Imagenet 32x32</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">3.58 bits/dim</td>
        </tr>
        <tr>
            <td align="center">Imagenet 64x64</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">3.30 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebA 64x64</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">1.83 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 256x256 (5-bits)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">0.51 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 256x256 (8-bits)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">1.35 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 256x256 (5-bits)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">0.53 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 256x256 (8-bits)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">2.17 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 1024x1024</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">1.01 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 1024x1024</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center">2.30 bits/dim</td>
        </tr>
    </tbody>
</table>
 
### Notes: 

- Downloading from the *"Checkpoints"* link will download the minimal required files to resume training/do inference. The minimal files are the model checkpoint file and the saved hyper-parameters of the run (explained further below).
- Downloading from the *"Logs"* link will download additional pre-training logs such as tensorboard files or saved images from training. *"Logs"* also holds the saved hyper-parameters of the run.
- Downloaded *"Logs"* and/or *"Checkpoints"* should be always unzipped in their implementation folder (`efficient_vdvae_torch` for Pytorch checkpoints and `efficient_vdvae_jax` for JAX checkpoints).
- Some of the model checkpoints are missing in either Pytorch or JAX for the moment. We will update them soon.
  
## Pre-requisites   
To run this codebase, you need:  
  
- Machine that runs a linux based OS (tested on Ubuntu 20.04 (LTS))  
- GPUs (preferably more than 16GB)  
- [Docker](https://docs.docker.com/engine/install/ubuntu/)  
- Python 3.7 or higher  
- CUDA 11.1 or higher (can be installed from [here](https://developer.nvidia.com/cuda-11.1.0-download-archive))  

We recommend running all the code below inside a `Linux screen` or any other terminal multiplexer, since some commands can take hours/days to finish and you don't want them to die when you close your terminal.
  
## Installation  
  
To create the docker image used in both the Pytorch and JAX implementations:  
  
```  
cd build  
docker build -t efficient_vdvae_image .  
```  
  
All code executions should be done within a docker container. To start the docker container, we provide a utility script:  
  
```  
sh docker_run.sh  # Starts the container and attaches terminal
cd /workspace/Efficient-VDVAE  # Inside docker container
```  
## Setup datasets  
  
All datasets can be automatically downloaded and pre-processed from the convenience script we provide:

```
cd data_scripts
sh download_and_preprocess.sh <dataset_name>
```

### Notes:
- `<dataset_name>` can be one of `(imagenet32, imagenet64, celeba, celebahq, ffhq)`. MNIST and CIFAR-10 datasets will get automatically downloaded later when training the model, and they do no require any dataset setup.
- For the `celeba` dataset, a manual download of `img_align_celeba.zip` and  `list_eval_partition.txt` files is necessary. Both files should be placed under `<project_path>/dataset_dumps/`.
- `img_align_celeba.zip` download [link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ).
- `list_eval_partition.txt` download [link](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg).
  
## Setting the hyper-parameters  
  
In this repository, we use [hparams](https://github.com/Rayhane-mamah/hparams) library (already included in the Dockerfile) for hyper-parameter management:  
  
- Specify all run parameters (number of GPUs, model parameters, etc) in one `.cfg` file
- Hparams evaluates any expression used as "value" in the `.cfg` file. "value" can be any basic python object `(floats, strings, lists, etc)` or any python basic expression `(1/2, max(3, 7), etc.)` as long as the evaluation does not require any library importations or does not rely on other values from the `.cfg`.
- Hparams saves the configuration of previous runs for reproducibility, resuming training, etc.  
- All hparams are saved by name, and re-using the same name will recall the old run instead of making a new one.  
- The `.cfg` file is split into sections for readability, and all parameters in the file are accessible as class attributes in the codebase for convenience.  
- The HParams object keeps a global state throughout all the scripts in the code.  
  
We highly recommend having a deeper look into how this library works by reading the [hparams library documentation](https://github.com/Rayhane-mamah/hparams), the [parameters description](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg) and figures 4 and 5 in the [paper]() before trying to run Efficient-VDVAE.  

We have heavily tested the robustness and stability of our approach, so changing the model/optimization hyper-parameters for memory load reduction should not introduce any drastic instabilities as to make the model untrainable. That is of course as long as the changes don't negate the important stability points we describe in the paper.
  
## Training the Efficient-VDVAE  
  
To run Efficient-VDVAE in Torch:  
  
```  
cd efficient_vdvae_torch  
# Set the hyper-parameters in "hparams.cfg" file  
# Set "NUM_GPUS_PER_NODE" in "train.sh" file  
sh train.sh  
```  
  
To run Efficient-VDVAE in JAX:  
  
```  
cd efficient_vdvae_jax  
# Set the hyper-parameters in "hparams.cfg" file  
python train.py  
```  
  
If you want to run the model with less GPUs than available on the hardware, for example 2 GPUs out of 8:  
  
```  
CUDA_VISIBLE_DEVICES=0,1 sh train.sh  # For torch  
CUDA_VISIBLE_DEVICES=0,1 python train.py  # For JAX  
```  
  
Models automatically create checkpoints during training. To resume a model from its last checkpoint, set its *`<run.name>`* in *`hparams.cfg`* file and re-run the same training commands.  
  
Since training commands will save the hparams of the defined run in the `.cfg` file. If trying to restart a pre-existing run (by re-using its name in `hparams.cfg`), we provide a convenience script for resetting saved runs:  
  
```  
cd efficient_vdvae_torch  # or cd efficient_vdvae_jax  
sh reset.sh <run.name>  # <run.name> is the first field in hparams.cfg  
```  

### Note:  
  
- To make things easier for new users, we provide example `hparams.cfg` files that can be used under the [egs](https://github.com/Rayhane-mamah/Efficient-VDVAE/tree/main/egs) folder. Detailed description of the role of each parameter is also inside [hparams.cfg](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg).
  
## Monitoring the training process  
  
While writing this codebase, we put extra emphasis on verbosity and logging. Aside from the printed logs on terminal (during training), you can monitor the training progress and keep track of useful metrics using [Tensorboard](https://www.tensorflow.org/tensorboard):  
  
```  
# While outside efficient_vdvae_torch or efficient_vdvae_jax  
tensorboard --logdir . --port <port_id> --reload_multifile=True  
```  
  
## Inference with the Efficient-VDVAE  
  
Efficient-VDVAE support multiple inference modes:  
  
- "reconstruction": Encodes then decodes the test set images and computes test NLL and SSIM.  
- "generation": Generates random images from the prior distribution. Randomness is controlled by the `run.seed` parameter.  
- "div_stats": Pre-computes the average KL divergence stats used to determine turned-off variates (refer to section 7 of the [paper]()). Note: This mode needs to be run before "encoding" mode and before trying to do masked "reconstruction" (Refer to [hparams.cfg](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg) for a detailed description).  
- "encoding": Extracts the latent distribution from the inference model, pruned to the quantile defined by `synthesis.variates_masks_quantile` parameter. This latent distribution is usable in downstream tasks.  
  
To run the inference:  
  
```  
cd efficient_vdvae_torch  # or cd efficient_vdvae_jax  
# Set the inference mode in "logs-<run.name>/hparams-<run.name>.cfg"  
# Set the same <run.name> in "hparams.cfg"  
python synthesize.py  
```  
  
### Notes:  
- Since training a model with a name *`<run.name>`* will save that configuration under *`logs-<run.name>/hparams-<run.name>.cfg`* for reproducibility and error reduction. Any changes that one wants to make during inference time need to be applied on the saved hparams file (*`logs-<run.name>/hparams-<run.name>.cfg`*) instead of the main file *`hparams.cfg`*.  
- The torch implementation currently doesn't support multi-GPU inference. The JAX implementation does.  
  
## Potential TODOs

- [x] Make data loaders Out-Of-Core (OOC) in Pytorch
- [ ] Make data loaders Out-Of-Core (OOC) in JAX
- [ ] Update pre-trained model checkpoints
- [ ] Improve the format of the encoded dataset used in downstream tasks (output of `encoding` mode, if there is a need)
- [ ] Write a `decoding` mode API (if needed).

## Bibtex  
  
TODO