Generative AI for Data Augmentation

📌 Overview

This project evaluates whether text-driven synthetic data augmentation improves fine-grained image classification under limited data conditions.

Using the Oxford-IIIT Pet dataset (37 classes), we build a generative pipeline that creates synthetic images from automatically generated and filtered captions, then measure the impact on classification performance.
Experiments are conducted on a 30% stratified training subset to simulate a limited-data scenario.

🧠 Pipeline

1. Caption Generation – BLIP-2

2. Caption Variation – Mistral-7B-Instruct (selected after comparative evaluation)

3. Semantic Filtering – Sentence embeddings + similarity scoring

4. Image Synthesis – Stable Diffusion v1.5 (512×512)

5. Training & Evaluation – ResNet-18 (ImageNet pretrained)

Three configurations are compared:

- Baseline – real images only

- Classical augmentation – flips, rotations, color jitter

- Synthetic + classical augmentation

Key insight:
Synthetic + classical augmentation improves performance by ~2 percentage points over baseline, demonstrating that structured generative augmentation adds meaningful diversity beyond standard transformations.

🚀 How to Run

Open the notebook directly in Colab and execute sequentially:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/Jorj91/Project_Generative_AI_for_Data_Augmentation/blob/main/main.ipynb)

⚠ Recommended: A100 GPU (image generation and LLM stages are memory intensive!)

Main execution flags:

RUN_CAPTIONING = True

RUN_TEXT_VARIATION = True

RUN_IMAGE_GENERATION = True

RUN_TRAINING = True


... Or view the notebook on GitHub: [main.ipynb](./main.ipynb)