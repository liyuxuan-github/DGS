# DGS: Robust and Diverse Watermarks for Diffusion Models

**Accepted at [ACCV 2026](https://accv2026.org/) (18th Asian Conference on Computer Vision)**

**Paper:** Coming soon 

Dynamic Gaussian Shading (DGS) is a training-free watermarking method for diffusion models that addresses generation diversity under a fixed prompt and user watermark.

![embed](fig\embed.png)

![extract](fig\extract.png)

## Getting Started

### Prerequisites
```bash
conda create -n dgs python=3.9
conda activate dgs
pip install -r requirements.txt
```

### Test True Positive Rate and Bit Accuracy

For testing in a lossless situation, you can run,
```bash
python run_dynamic_gaussian_shading.py \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --num 1000
```


To test the performance of Dynamic Gaussian Shading under noise perturbation (e.g., JPEG QF=25), you can run, 
```bash
python run_dynamic_gaussian_shading.py \
      --jpeg_ratio 25 \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --num 1000
```
For more adversarial cases, You can refer to ./script/run.sh

### Calculate CLIP Score

To calculate the CLIP Score, it relies on two pre-trained models, you can run,
```bash
python run_dynamic_gaussian_shading.py \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --num 1000 \
      --reference_model ViT-g-14 \
      --reference_model_pretrain laion2b_s12b_b42k 
```

### Calculate FID

When calculating  FID, we have aligned our settings with [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark) and used the same ground truth dataset. The dataset contains 5000 images from the COCO dataset. You can find the corresponding information such as prompts in 'fid_outputs/coco/meta_data.json'. 
The ground truth dataset can download [here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing).


Then, to calculate FID, you can run,
```bash
python dynamic_gaussian_shading_fid.py \
      --channel_copy 1 \
      --hw_copy 8 \
      --num 5000 
```

### Test Diversity

To calculate diversity, you first need to extract the features of the generated image, you can run,

```bash
python feature_dynamic_gaussian_shading.py \
      --fpr 0.000001 \
      --channel_copy 1 \
      --hw_copy 8 \
      --num 1000 \
      --reference_model ViT-g-14 \
      --reference_model_pretrain laion2b_s12b_b42k \
      --filename dgs
```


Then, to get the EFD, you can run,

```bash
python div.py
python lp.py
```

## Acknowledgement

Our project is implemented base on the following projects. We really appreciate their excellent open-source works!

* [Gaussian Shading]([https://github.com/bsmhmmlf/Gaussian-Shading])

## Citation

If our work has been helpful to you, we would greatly appreciate a citation.

```markdown
Yuxuan Li and Hao Tang.
DGS: Robust and Diverse Watermarks for Diffusion Models.  
ACCV 2026 (accepted; proceedings forthcoming).

The official BibTeX entry will be added once the proceedings are available.
```
