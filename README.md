# Generative AI for Data Augmentation

<br>

## 📌 Overview

This project evaluates whether text-driven synthetic data augmentation improves fine-grained image classification under limited data conditions.

Using the Oxford-IIIT Pet dataset (37 classes), we build a generative pipeline that creates synthetic images from automatically generated and filtered captions, then measure the impact on classification performance.

Experiments are conducted on a 30% stratified training subset to simulate a limited-data scenario.

### Dataset Sizes Used in Experiments

| Dataset configuration | Number of training images |
|---|---|
| Real images only (30% subset) | **1,104** |
| Classical augmentation | **1,104** |
| Synthetic images only | **2,170** |
| Synthetic + classical augmentation | **3,274** |
| Test set | **3,669** |

Classical augmentation applies transformations during training (flips, rotations, color jitter) but does not increase the dataset size, while synthetic augmentation generates new images to expand the dataset.

<br>

## 🧠 Pipeline

1. Caption Generation – BLIP-2

2. Caption Variation – Mistral-7B-Instruct (selected after comparative evaluation)

3. Semantic Filtering – Sentence embeddings + similarity scoring

4. Image Synthesis – Stable Diffusion v1.5 (512×512)

5. Training & Evaluation – ResNet-18 (ImageNet pretrained)

Three configurations are compared:

- Baseline – real images only

- Classical augmentation – flips, rotations, color jitter

- Synthetic + classical augmentation

**Key insights:**

Synthetic + classical augmentation improves performance by ~1.04 percentage points over baseline.

Due to the stochastic nature of deep learning training (weight initialization, data shuffling, and augmentation sampling), performance varied slightly across runs, with improvements occasionally reaching around +2 percentage points.

This demonstrates that structured generative augmentation adds meaningful diversity beyond standard transformations.

<br>

## 🚀 How to Run

This project provides two notebook versions:

1️⃣ **<u>Modularized Version (Recommended)</u>**

This version uses the modular project structure (src/ modules) and reproduces the full pipeline.

Open directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jorj91/Project_Generative_AI_for_Data_Augmentation/blob/main/main.ipynb)

⚠ Recommended: A100 GPU (image generation and LLM stages are memory intensive!)

Run the notebook sequentially. 

Main execution flags:

```
RUN_CAPTIONING = True
RUN_TEXT_VARIATION = True
RUN_IMAGE_GENERATION = True
RUN_TRAINING = True
```

Or view the notebook on GitHub: [main.ipynb](./main.ipynb)


2️⃣ **<u>Submission Version (All Code Integrated)</u>**

This version contains all functions directly inside the notebook, allowing the reviewer to inspect the full pipeline and results without needing to clone modules or run computationally expensive stages again.

Open directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jorj91/Project_Generative_AI_for_Data_Augmentation/blob/main/main_submission.ipynb)

This notebook shows the final executed pipeline with outputs already saved.

Or view the notebook on GitHub: [main_submission.ipynb](./main_submission.ipynb)
