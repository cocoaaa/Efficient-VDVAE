# Instructions from the authors
## Set up environment: 
1. this is their dockerfile
```
FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN pip install --upgrade pip

#install hparams
RUN pip install --upgrade git+https://github.com/Rayhane-mamah/hparams

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
#pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install JAX extensions
RUN pip install --upgrade optax flax trax

ENV NUMBA_CACHE_DIR /tmp/

```

2. this is their requirement.txt
```
audioread==2.1.8
ffmpeg==1.4
inflect==3.0.2
ipywidgets==7.5.1
librosa
matplotlib==3.1.2
# numpy==1.18.1
opencv-python==4.1.2.30
pandas==0.25.3
Pillow==7.1.1
protobuf==3.11.2
scikit-image==0.16.2
scikit-learn==0.22
tqdm==4.40.1
Unidecode==1.1.1
tensorflow_probability==0.14.1
torch_optimizer==0.3.0
```
##  To run the inference:  
"generation": Generates random images from the prior distribution. Randomness is controlled by the run.seed parameter.

1. Set the parameters in hparams.cfg
   - run.seed: random seed for generation process
   - run.name: this is the name used when training a model, that is:
     - During training, parameters and logs for the training run is saved to
       `logs-<run.name>/hparams-<run.name>.cfg`
       For eg., the released checkpoint zip contains `log-celebahq256_8bits_baseline` folder 
       (in addition to `checkpoints-celebahq256_87bits_baseline` folder). 
       and this `log-<run.name>` folder (here `run.name` is `celebahq256_8bits_baseline`) contains 
       a config file called `hparams-<run.name>.cfg` (again, with `run.name` of `celebahq256_8bits_baseline`).

   - "Set the inference mode in "logs-<run.name>/hparams-<run.name>.cfg" to "generation" : synthesis_mode = 'generation'



## Note:
- what is "run.name"? 
  > Models automatically create checkpoints during training. To resume a model from its last checkpoint, set its *`<run.name>`* in *`hparams.cfg`* file and re-run the same training commands. 
- the default locations the training or sampling script will consider 
  - ckpt file: `./efficient_vdvae_torch/checkpoints-<run.name>`
  - log/config/hparams file is: `./efficient_vdvae_torch/logs-<run.name>/hparams-<run.name>.cfg`.
- outdir where the generated images will be saved:  `./logs-<run.name>/synthesis-images/generated`



### Run this code for generation:
- set the temperature_settings to [0.85] -- which is used in the fig 1 of their paper
- set batch_size to 1; tried 4 as max but then somehow there was out of memory error 
- set n_generation_batches to 100k 
  - essentially, n_genreation_batches * batch_size (nder "genration" settings in the config file) need to be n_samples we need.
  - 

``` shell 
cd efficient_vdvae_torch  # or cd efficient_vdvae_jax  
conda activate test

# Set the inference mode in "logs-<run.name>/hparams-<run.name>.cfg"  
# Set the same <run.name> in "hparams.cfg"  
# Set the random see
export run.name="celebahq256_8bits_baseline"

export CUDA_VISIBLE_DEVICES=0
# python -m pdb synthesize.py  #for debug

# !!-- WARNING --!!
# this may overwrite to already generated images b.c.
# the output dir is relative to the config file's location, i.e.,
# output images will be saved to <config_fp>/`synthesis-images/generated` folder
nohup python synthesize.py &

```  
- started: 20230301-231925
- pid: 7166
- outdir: /data/hayley-old/Github/VAEs/Efficient-VDVAE/efficient_vdvae_torch/logs-celebahq256_8bits_baseline/synthesis-images/generated

- current 491


> Since training a model with a name *`<run.name>`* will save that configuration under *`logs-<run.name>/hparams-<run.name>.cfg`* for reproducibility and error reduction. Any changes that one wants to make during inference time need to be applied on the saved hparams file (*`logs-<run.name>/hparams-<run.name>.cfg`*) instead of the main file *`hparams.cfg`*.  
>  The torch implementation currently doesn't support multi-GPU inference. The JAX implementation does. 
 
  


### Codes I ran so far -- 20230301-192224:
- To set up the run environment based off of my `test` conda-env
```shell
conda activate test
pip install git+https://github.com/Rayhane-mamah/hparams
```
####  Note/Log: arghh, this gives some errors. I hate pip...!
$ mamba install hparams

                  __    __    __    __
                 /  \  /  \  /  \  /  \
                /    \/    \/    \/    \
███████████████/  /██/  /██/  /██/  /████████████████████████
              /  / \   / \   / \   / \  \____
             /  /   \_/   \_/   \_/   \    o \__,
            / _/                       \_____/  `
            |/
        ███╗   ███╗ █████╗ ███╗   ███╗██████╗  █████╗
        ████╗ ████║██╔══██╗████╗ ████║██╔══██╗██╔══██╗
        ██╔████╔██║███████║██╔████╔██║██████╔╝███████║
        ██║╚██╔╝██║██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║
        ██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║
        ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝

        mamba (0.15.3) supported by @QuantStack

        GitHub:  https://github.com/mamba-org/mamba
        Twitter: https://twitter.com/QuantStack

█████████████████████████████████████████████████████████████


Looking for: ['hparams']

pkgs/r/linux-64          [====================] (00m:00s) No change
pkgs/r/noarch            [====================] (00m:00s) No change
pkgs/main/noarch         [====================] (00m:00s) Done
pkgs/main/linux-64       [====================] (00m:00s) Done
conda-forge/noarch       [====================] (00m:02s) Done
conda-forge/linux-64     [====================] (00m:08s) Done

Pinned packages:
  - python 3.8.*


Encountered problems while solving:
  - nothing provides requested hparams

(test) hayley@arya /data/hayley-old/Github/VAEs/Efficient-VDVAE/efficient_vdvae_torch 
$ pip install git+https://github.com/Rayhane-mamah/hparams
Collecting git+https://github.com/Rayhane-mamah/hparams
  Cloning https://github.com/Rayhane-mamah/hparams to /tmp/pip-req-build-ouraf2ok
Collecting gcsfs
  Downloading gcsfs-2023.1.0-py2.py3-none-any.whl (26 kB)
Requirement already satisfied: requests in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.25.0)
Requirement already satisfied: decorator>4.1.2 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (5.1.0)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (3.7.4.post0)
Requirement already satisfied: google-auth-oauthlib in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (0.4.6)
Requirement already satisfied: google-auth>=1.2 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.3.3)
Requirement already satisfied: attrs>=17.3.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (21.2.0)
Requirement already satisfied: async-timeout<4.0,>=3.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (3.0.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (5.2.0)
Requirement already satisfied: chardet<5.0,>=2.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (3.0.4)
Requirement already satisfied: typing-extensions>=3.6.5 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (3.10.0.2)
Requirement already satisfied: yarl<2.0,>=1.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (1.7.2)
Collecting fsspec==2023.1.0
  Downloading fsspec-2023.1.0-py3-none-any.whl (143 kB)
     |████████████████████████████████| 143 kB 49.3 MB/s 
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (4.2.4)
Requirement already satisfied: rsa<5,>=3.1.4 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (4.7.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (0.2.7)
Requirement already satisfied: six>=1.9.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (1.16.0)
Requirement already satisfied: setuptools>=40.3.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (59.1.0)
Requirement already satisfied: google-auth>=1.2 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.3.3)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth-oauthlib->gcsfs->hparams==0.2) (1.3.0)
Collecting google-cloud-storage
  Downloading google_cloud_storage-2.7.0-py2.py3-none-any.whl (110 kB)
     |████████████████████████████████| 110 kB 74.1 MB/s 
Requirement already satisfied: requests in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.25.0)
Requirement already satisfied: google-auth>=1.2 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.3.3)
Collecting google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5
  Downloading google_api_core-2.11.0-py3-none-any.whl (120 kB)
     |████████████████████████████████| 120 kB 69.4 MB/s 
Requirement already satisfied: requests in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.25.0)
Collecting google-auth>=1.2
  Downloading google_auth-2.16.1-py2.py3-none-any.whl (177 kB)
     |████████████████████████████████| 177 kB 73.8 MB/s 
Requirement already satisfied: six>=1.9.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (1.16.0)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (4.2.4)
Requirement already satisfied: rsa<5,>=3.1.4 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (4.7.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from google-auth>=1.2->gcsfs->hparams==0.2) (0.2.7)
Collecting google-cloud-core<3.0dev,>=2.3.0
  Downloading google_cloud_core-2.3.2-py2.py3-none-any.whl (29 kB)
Collecting google-resumable-media>=2.3.2
  Downloading google_resumable_media-2.4.1-py2.py3-none-any.whl (77 kB)
     |████████████████████████████████| 77 kB 26.8 MB/s 
Collecting google-crc32c<2.0dev,>=1.0
  Downloading google_crc32c-1.5.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (32 kB)
Collecting googleapis-common-protos<2.0dev,>=1.56.2
  Downloading googleapis_common_protos-1.58.0-py2.py3-none-any.whl (223 kB)
     |████████████████████████████████| 223 kB 70.8 MB/s 
Collecting protobuf!=3.20.0,!=3.20.1,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.19.5
  Downloading protobuf-4.22.0-cp37-abi3-manylinux2014_x86_64.whl (302 kB)
     |████████████████████████████████| 302 kB 66.5 MB/s 
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from pyasn1-modules>=0.2.1->google-auth>=1.2->gcsfs->hparams==0.2) (0.4.8)
Requirement already satisfied: idna<3,>=2.5 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from requests->gcsfs->hparams==0.2) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from requests->gcsfs->hparams==0.2) (2022.12.7)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from requests->gcsfs->hparams==0.2) (1.25.11)
Requirement already satisfied: chardet<5.0,>=2.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (3.0.4)
Requirement already satisfied: oauthlib>=3.0.0 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs->hparams==0.2) (3.1.1)
Requirement already satisfied: requests in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from gcsfs->hparams==0.2) (2.25.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from pyasn1-modules>=0.2.1->google-auth>=1.2->gcsfs->hparams==0.2) (0.4.8)
Requirement already satisfied: idna<3,>=2.5 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from requests->gcsfs->hparams==0.2) (2.10)
Requirement already satisfied: multidict<7.0,>=4.5 in /home/hayley/miniconda3/envs/test/lib/python3.8/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->gcsfs->hparams==0.2) (5.2.0)
Building wheels for collected packages: hparams
  Building wheel for hparams (setup.py) ... done
  Created wheel for hparams: filename=hparams-0.2-py3-none-any.whl size=8300 sha256=e35dff64c6016e58bd6ec41e52425f231f19dc0666c4dee8bd43e49a4c4412ba
  Stored in directory: /tmp/pip-ephem-wheel-cache-z2yq2ohd/wheels/ad/21/22/059bbbbe731c989eb3487a3d7889ebd0f26342a602bebf7fc4
Successfully built hparams
Installing collected packages: protobuf, googleapis-common-protos, google-auth, google-crc32c, google-api-core, google-resumable-media, google-cloud-core, google-cloud-storage, fsspec, gcsfs, hparams
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.20.1
    Uninstalling protobuf-3.20.1:
      Successfully uninstalled protobuf-3.20.1
  Attempting uninstall: googleapis-common-protos
    Found existing installation: googleapis-common-protos 1.52.0
    Uninstalling googleapis-common-protos-1.52.0:
      Successfully uninstalled googleapis-common-protos-1.52.0
  Attempting uninstall: google-auth
    Found existing installation: google-auth 2.3.3
    Uninstalling google-auth-2.3.3:
      Successfully uninstalled google-auth-2.3.3
  Attempting uninstall: google-api-core
    Found existing installation: google-api-core 1.24.1
    Uninstalling google-api-core-1.24.1:
      Successfully uninstalled google-api-core-1.24.1
  Attempting uninstall: fsspec
    Found existing installation: fsspec 2021.11.0
    Uninstalling fsspec-2021.11.0:
      Successfully uninstalled fsspec-2021.11.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
wandb 0.12.17 requires protobuf<4.0dev,>=3.12.0, but you have protobuf 4.22.0 which is incompatible.
opencensus 0.7.11 requires google-api-core<2.0.0,>=1.0.0, but you have google-api-core 2.11.0 which is incompatible.
grpcio-tools 1.37.1 requires protobuf<4.0dev,>=3.5.0.post1, but you have protobuf 4.22.0 which is incompatible.
Successfully installed fsspec-2023.1.0 gcsfs-2023.1.0 google-api-core-2.11.0 google-auth-2.16.1 google-cloud-core-2.3.2 google-cloud-storage-2.7.0 google-crc32c-1.5.0 google-resumable-media-2.4.1 googleapis-common-protos-1.58.0 hparams-0.2 protobuf-4.22.0





## Sample from their model trained on ffhq256_8bits
- date: 20230906-165602
- purpose: for creating gm-ffhq256 dataset for cvpr24 submission
- project: proj/fgm

### Commands I ran:
- follow the insturctions in their readme.md

```shell
cd efficient_vdvae_torch 


#1. donwload the released checkpoint zip file, which contains the model ckpt and the config file (needed for setting hparams for sampling from the ckpt)
src=/Users/hayley/Downloads/ffhq256_8bits_baseline_checkpoints.zip
dst=/data/hayley-old/Github/VAEs/Efficient-VDVAE/efficient_vdvae_torch
rsync -azP  $src hayley@arya.usc.edu:$dst

#2. unzip this file at `efficient_vdvae_torch` folder
unzip ./ffhq256_8bits_baseline_checkpoints.zip
mv ffhq256_8bits_baseline_checkpoints/logs_... folder to ../ 
rm ffhq256_8bits_baseline_checkpoints ffhq256_8bits_baseline_checkpoints.zip

#-note: run_name should match this file's name, e.g. logs-<run.name>/
#-also need to:
#--check: set the inference mode in "logs-<run.name>/hparams-<run.name>.cfg"  
#--check: set the same <run.name> in "hparams.cfg"  



#Run sampling
python synthesize.py

```

### Progress
- started:
- pid:
- sample_dir:

### todo:
- [ ] check if 50k kimages are sampled:   
```shell
sample_dir=$logdir/synthesis-images
ls $sample_dir | wc -l
```
- [ ] rsynced to isi.turing?
  - [ ] step1: rsync from arya to mbp
  - [ ] step2: rsync from mbp to turing