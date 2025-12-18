## Requirements
```commandline
cond env create -f environment.yml
conda activate BBDM
```

## 📂 Data Preparation

### Paired Data Organization
To train the model on your custom dataset, please organize your images into a directory structure as shown below. Ensure that images in folder `A` (Source) and folder `B` (Target) share the same filenames.

```text
/path/to/your_dataset/
├── train
│   ├── A       # Input images 
│   └── B       # Ground Truth images
├── val
│   ├── A
│   └── B
└── test
    ├── A
    └── B
```

[//]: # (Configuration)
Update the dataset configuration in your config.yaml file to point to your data root:

dataset_name: 'custom_dataset'   # Give your dataset a name
dataset_type: 'custom_aligned'   # Keep this as 'custom_aligned'
dataset_config:
  dataset_path: '/path/to/your_dataset'  # Absolute or relative path



## 🚀 Training and Testing

### 1. Configuration
First, choose and modify the configuration file based on the templates provided in the `configs/` directory:

- **Pixel Space:** Use `configs/Template-BBDM.yaml`.
- **Latent Space:** Use `configs/Template-LBBDM-f4.yaml` (or f8/f16) depending on your desired latent depth.

> **⚠️ Important:** Open your chosen config file and ensure you have updated the **`dataset_path`** and **`VQGAN checkpoint path`** to match your local setup.

### 2. Training
To start training from scratch, run the following command. The `--save_top` flag ensures the best performing models are saved.

python3 main.py --config configs/Template_LBBDM_f4.yaml \
                --train \
                --sample_at_start \
                --save_top \
                --gpu_ids 0

### 3. Testing/Evaluation
To sample from the whole test dataset and evaluate metrics, use the --sample_to_eval flag and specify the model checkpoint:   

python3 main.py --config configs/Template_LBBDM_f4.yaml \
                --sample_to_eval \
                --gpu_ids 0 \
                --resume_model path/to/your/model_ckpt.pth       


## 📥 Pre-trained Models
This project utilizes pre-trained VQGAN models from [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion) as the first-stage autoencoder. All our models are trained based on these consistent VQGAN backbones.

### Download Checkpoints
Please download the corresponding VQGAN checkpoint for your desired latent depth ($f=4, 8, 16$).

| Model | Downsampling Factor | Download Link |
| :--- | :---: | :--- |
| **VQ-f4** | $f=4$ | [Download vq-f4.zip](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) |
| **VQ-f8** | $f=8$ | [Download vq-f8.zip](https://ommer-lab.com/files/latent-diffusion/vq-f8.zip) |
| **VQ-f16** | $f=16$ | [Download vq-f16.zip](https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1) |

> **Note:** After downloading, remember to update the `vqgan_ckpt_path` in your configuration file to point to these files.


## 🙏 Acknowledgments

Our code is developed based on the excellent open-source work of **BBDM**. We deeply appreciate the authors for providing their implementation.

* **BBDM:** [Brownian Bridge Diffusion Model](https://github.com/xuekt98/BBDM)

