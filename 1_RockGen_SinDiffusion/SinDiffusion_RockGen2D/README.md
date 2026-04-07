# SinDiffusion: Learning a Diffusion Model from a Single Natural Image

Official PyTorch implementation of "SinDiffusion: Learning a Diffusion Model from a Single Natural Image".
The code aims to allow the users to reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[Arxiv](https://arxiv.org/abs/2211.12445)]


# Overview

This repository implements the SinDiffusion model, leveraging denoising diffusion models to capture internal distribution of patches from a single natural image. 
SinDiffusion significantly improves the quality and diversity of generated samples compared with existing GAN-based approaches. 
It is based on two core designs. 
First, SinDiffusion is trained with a single model at a single scale instead of multiple models with progressive growing of scales which serves as the default setting in prior work. 
This avoids the accumulation of errors, which cause characteristic artifacts in generated results.
Second, we identify that a patch-level receptive field of the diffusion network is crucial and effective for capturing the image's patch statistics, therefore we redesign the network structure of the diffusion model.
Extensive experiments on a wide range of images demonstrate the superiority of our proposed method for modeling the patch distribution.

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Datasets
Here we included the data in the ./data folder which includes rock images of different morphologies.


| Rock Type | Data Sources / References |
| :--- | :--- |
| **Sandstone** | Bentheimer (Kelly et al., 2021) |
| | North Sea (Niu et al., 2020) |
| | Leman (Scott et al., 2020) |
| | North Sea (Scott et al., 2020) |
| | Berea (Safari et al., 2021) |
| **Carbonate** | Liu et al. (2022) |
| **Limestone** | Indiana (Safari et al., 2021) |
| **Mudrock** | Nankai (Milliken et al., 2010) |
| **Shale** | Landry et al. (2020) |

**Full References**

- Liu & Mukerji (2022): M. Liu and T. Mukerji, Geophysical Research Letters, 2022, 49, e2022GL098342.

- Guan et al. (2021): M. Guan, T. I. Anderson, P. Creux and A. R. Kovscek, Computers & Geosciences, 2021, 156, 104905.

- Safari et al. (2021): H. Safari, B. J. Balcom and A. Afrough, Computers & Geosciences, 2021, 156, 104895.

- Niu et al. (2020): Y. Niu, P. Mostaghimi, M. Shabaninejad, P. Swietojanski and R. T. Armstrong, Water Resources Research, 2020, 56, e2019WR026597.

- Scott et al. (2019/2020): G. Scott, K. Wu and Y. Zhou, Transport in Porous Media, 2019, 129, 855–884.

- Milliken & Reed (2010): K. L. Milliken and R. M. Reed, Journal of Structural Geology, 2010, 32, 1887–1898.

- Landry et al. (2020): C. J. Landry, B. S. Hart and M. Prodanovic, SPE/AAPG/SEG Unconventional Resources Technology Conference, 2020, p. D023S040R002.



## Training the model (single image, single GPU/CPU)
To train the model on a single reference image (e.g., `data/test2.png`) and save checkpoints/samples:
```bash
python proper_sindiffusion_train.py --data_dir data/test2.png --image_size 256 --diffusion_steps 1000 \
  --epochs 1000 --batch_size 8 --lr 5e-4 --num_channels 64 --num_res_blocks 1 \
  --channel_mult "1,2,4" --attention_resolutions "2"
```
The experimental results are saved under `./OUTPUT/proper_sindiffusion_test2/`, including `proper_generated_sample.png` after training ends.

## Testing / sampling the model
To sample multiple images from a trained model checkpoint using the proper diffusion sampler:
```bash
python test_proper_model.py --model_path OUTPUT/proper_sindiffusion_test2/model_epoch_1000.pth \
  --data_dir data/test2.png --num_samples 8 --output_dir RESULT/proper_sindiffusion_test2
```
The generated images are saved to `./RESULT/proper_sindiffusion_test2/`, alongside a `comparison.png` visualization.

# Additional information

## Acknowledge
Our code is developed based on [SinDiffusion]. If you use the original model please cite
```
@article{wang2022sindiffusion,
  title={SinDiffusion: Learning a Diffusion Model from a Single Natural Image},
  author = {Wang, Weilun and Bao, Jianmin and Zhou, Wengang and Chen, Dongdong and Chen, Dong and Yuan, Lu and Li, Houqiang},
  journal={arXiv preprint arXiv:2211.12445},
  year={2022}
}
```