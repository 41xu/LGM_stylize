## Stylized 3DGS based on LGM

### Stylization

```bash
# use ip2p for stylization and editing
python stylize_different_lr.py big --resume pretrained/model_fp16.safetensors --workspace stylize_result --test_path style_test/

# image-based stylization
python stylize_nnfm.py big --resume pretrained/model_fp16.safetensors --workspace stylize_result --test_path style_test/

# some testing and helpful scripts
# script for lora training using a small dataset of VanGogh's painting
python train_lora.py config/VanGogh.json

```

### Install

```bash

conda create -n LGM python=3.8
conda activate LGM

pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.

# a modified gaussian splatting (+ depth, alpha rendering)
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -r requirement_threestudio.txt
# other dependencies
pip install -r requirements.txt
```

### Pretrained Weights

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/ashawkey/LGM).

For example, to download the fp16 model for inference:
```bash
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors
cd ..
```

For [MVDream](https://github.com/bytedance/MVDream) and [ImageDream](https://github.com/bytedance/ImageDream), we use a [diffusers implementation](https://github.com/ashawkey/mvdream_diffusers).
Their weights will be downloaded automatically.

### Inference

Inference takes about 10GB GPU memory (loading all imagedream, mvdream, and our LGM).

```bash
### gradio app for both text/image to 3D
python app.py big --resume pretrained/model_fp16.safetensors

### test
# --workspace: folder to save output (*.ply and *.mp4)
# --test_path: path to a folder containing images, or a single image
python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test --test_path data_test 

### local gui to visualize saved ply
python gui.py big --output_size 800 --test_path workspace_test/saved.ply

### mesh conversion
python convert.py big --test_path workspace_test/saved.ply
```

For more options, please check [options](./core/options.py).

### Training

**NOTE**: 
Since the dataset used in our training is based on AWS, it cannot be directly used for training in a new environment.
We provide the necessary training code framework, please check and modify the [dataset](./core/provider_objaverse.py) implementation!

We also provide the **~80K subset of [Objaverse](https://objaverse.allenai.org/objaverse-1.0)** used to train LGM in [objaverse_filter](https://github.com/ashawkey/objaverse_filter).

```bash
# debug training
accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# training (use slurm for multi-nodes training)
accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace
```
