# Rock image generation and texture pipeline

This repository is a four-stage workflow for **generation of 2D rock images and further preparation for microfluidic fabrications**: train a single-image diffusion model for diverse rock samples, gurantee the **percolation** of pores, **synthesize different shaped/sized textures** via image quilting, and export **DXF** geometry for microfluidic layout in ECAD and fabrication.

Each stage lives in its own folder. Several stages bundle or adapt **third-party code**; see [License](#license) and the links below.

## Repository layout

| Folder | Role |
|--------|------|
| [`1_RockGen_SinDiffusion/SinDiffusion_RockGen2D`](1_RockGen_SinDiffusion/SinDiffusion_RockGen2D) | SinDiffusion-based rock image generation (single reference image) |
| [`2_Percolation`](2_Percolation) | Pore connectivity: shrink the field until left–right or top–bottom percolation appears |
| [`3_TextureSyn`](3_TextureSyn) | Image quilting texture synthesis (Efros & Freeman) |
| [`4_ToFabPattern`](4_ToFabPattern) | Raster → DXF contours and solid hatch for tooling |

---

## 1. Rock generation with SinDiffusion

**Method.** [SinDiffusion: Train a Diffusion Model from a Single Natural Image](https://arxiv.org/abs/2211.12445) learns a diffusion model from **one** training image by modeling internal patch statistics.

**Upstream.** Official implementation: [WeilunWang/SinDiffusion](https://github.com/WeilunWang/SinDiffusion) (Apache-2.0). This project uses the same `guided_diffusion` stack with a **single-image training and sampling** workflow tailored for rock tomography–style inputs.

### Setup

```bash
cd 1_RockGen_SinDiffusion/SinDiffusion_RockGen2D
pip install -r requirements.txt
```

Use a **CUDA** build and GPU for practical training times. On some macOS setups you may need:

`KMP_DUPLICATE_LIB_OK=TRUE` (avoids OpenMP duplicate-library issues when running PyTorch).

### Train

Point `--data_dir` at one RGB (or single-image workflow as loaded by the script) **reference rock** slice. Example:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python rock_sindiffusion_train.py \
  --data_dir data/sindiff_test/Scott/Scott_NS_3.png \
  --image_size 512 \
  --diffusion_steps 2000 \
  --epochs 2200 \
  --batch_size 8 \
  --lr 5e-4 \
  --num_channels 64 \
  --num_res_blocks 1 \
  --channel_mult "1,2,4" \
  --attention_resolutions "2" \
  --use_checkpoint \
  --save_interval 100 \
  --log_interval 50
```

**Checkpoints** are written under:

`OUTPUT/proper_sindiffusion_<image_basename_without_extension>/`

For example, for `Scott_NS_3.png` the folder is `OUTPUT/proper_sindiffusion_Scott_NS_3/`, with files like `model_epoch_2000.pth`.

### Sample

Generate multiple images from a saved checkpoint (use the same `--data_dir` reference image so architecture and normalization stay consistent):

```bash
python simple_sampler.py \
  --model_path ./OUTPUT/proper_sindiffusion_Scott_NS_3/model_epoch_2000.pth \
  --data_dir data/sindiff_test/Scott/Scott_NS_3.png \
  --output_dir RESULT/Scott_NS_3 \
  --num_samples 50 \
  --batch_size 10
```

### Citation (SinDiffusion)

```bibtex
@article{wang2022sindiffusion,
  title={SinDiffusion: Learning a Diffusion Model from a Single Natural Image},
  author = {Wang, Weilun and Bao, Jianmin and Zhou, Wengang and Chen, Dongdong and Chen, Dong and Yuan, Lu and Li, Houqiang},
  journal={arXiv preprint arXiv:2211.12445},
  year={2022}
}
```

---

## 2. Percolation (connect pore space)

**Goal.** Diffusion-generated binary rocks sometimes have **isolated pores**. This notebook **shrinks the domain** while keeping each pore’s **shape relative to its centroid**, so neighboring pores overlap and can merge until a **spanning cluster** appears (connectivity from left to right **or** top to bottom).

**Workflow (`2_Percolation/percolation_test.ipynb`).**

1. Load a grayscale image and threshold: pixels **below 128** → pore (`True`).
2. Label connected components, discard regions smaller than **0.05%** of the image area.
3. For each remaining pore, store centroid and pixel offsets relative to the centroid.
4. For a list of **scale factors** from 1.0 down (default: `numpy.linspace(1.0, 0.6, 10)`), reconstruct the pore map at reduced height/width and test percolation.
5. Save a summary figure `scaled_overlap_pores.png` and, when percolation first occurs, save `perco.png` next to the input (`img_path`).

**Setup.** Install **NumPy**, **SciPy**, **scikit-image**, **matplotlib**, **Pillow**, and run the notebook in Jupyter or VS Code.

Edit the `img_path` variable at the top of the notebook to your binary-friendly rock image.

---

## 3. Texture synthesis (image quilting)

**Method.** *Image Quilting for Texture Synthesis and Transfer* ([Efros & Freeman, SIGGRAPH 2001](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf)); implementation lineage includes [trunc8/image-quilting](https://github.com/trunc8/image-quilting) and related course projects.

**Usage** (`3_TextureSyn`):

```bash
cd 3_TextureSyn
pip install numpy opencv-python matplotlib  # add others as needed if import errors
python main.py -i <texture_image> -b <block_size> -o <overlap_fraction> \
  -s <scale> -n <num_outputs> -f <output.png> -p <0|1> -t <tolerance>
```

Use `python main.py -h` for details. See `3_TextureSyn/LICENSE.md` for the MIT terms on that subtree.

---

## 4. Image to DXF (fabrication-oriented)

**Scripts** (`4_ToFabPattern`):

- **`todxf.py`** — Loads a grayscale rock image, resizes to **5001×2001** (as you wish), **Otsu** threshold, finds **outer contours and holes** (OpenCV), shifts geometry to the origin, scales by **4×**, writes a **polyline DXF** (`rock_background.dxf` by default).
- **`hatch.py`** — Reads the DXF, identifies the **outer boundary** containing a user-defined **seed point**, attaches **solid hatch** on layer `hatch`, saves `rock_background_hatched.dxf` for the future ECAD identifier.

**Dependencies:** `opencv-python`, `numpy`, `ezdxf`.

Edit `input_image`, `seed_point`, and paths at the top of each script to match your layout.

---

## License

This repository **mixes licenses**; you must comply with the license in each subtree that you use.

| Component | License | File |
|-----------|---------|------|
| `1_RockGen_SinDiffusion/SinDiffusion_RockGen2D` | Apache-2.0 | [`1_RockGen_SinDiffusion/SinDiffusion_RockGen2D/LICENSE`](1_RockGen_SinDiffusion/SinDiffusion_RockGen2D/LICENSE) |
| `2_Percolation`, `4_ToFabPattern`, root `README.md` | See root [`LICENSE`](LICENSE) | — |
| `3_TextureSyn` | MIT | [`3_TextureSyn/LICENSE.md`](3_TextureSyn/LICENSE.md) |



## Disclaimer

Software is provided **as-is** for research and prototyping. Verify outputs for your own metrology, fabrication, and publication workflows.
