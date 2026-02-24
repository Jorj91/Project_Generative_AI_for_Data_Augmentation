# clone repo

import os

PROJECT_ROOT = "/content/Project_Generative_AI_for_Data_Augmentation"

if not os.path.exists(PROJECT_ROOT):
    !git clone https://github.com/Jorj91/Project_Generative_AI_for_Data_Augmentation.git {PROJECT_ROOT}

%cd {PROJECT_ROOT}

# CONTROLLED VERBOSITY

# Disable HF download progress bars BEFORE importing anything HF-related

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Dependency install
INSTALL_DEPS = True

if INSTALL_DEPS:
    !pip install -r requirements.txt -q

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import importlib

import captioning
importlib.reload(captioning)
from captioning import run_captioning


import training_evaluation
importlib.reload(training_evaluation)
from training_evaluation import run_training

# Setup

import torch
import logging
from torchvision.datasets import OxfordIIITPet
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import random
from transformers.utils import logging as transformers_logging
from huggingface_hub.utils import logging as hf_logging


# Silence transformers & HF logs (keep only errors)
transformers_logging.set_verbosity_error()
hf_logging.set_verbosity_error()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

import gc

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

PROJECT_ROOT # it must be /content/Project_Generative_AI_for_Data_Augmentation

# control flags.
RUN_CAPTIONING = True
RUN_TEXT_VARIATION = True
RUN_IMAGE_GENERATION = True
RUN_TRAINING = True

# dataset loading

dataset_train = OxfordIIITPet(
    root=os.path.join(PROJECT_ROOT, "data", "raw"),
    split="trainval",
    download=True
)

dataset_test = OxfordIIITPet(
    root=os.path.join(PROJECT_ROOT, "data", "raw"),
    split="test",
    download=True
)

print("Train size:", len(dataset_train))
print("Test size:", len(dataset_test))

# extract labels
labels = dataset_train._labels
indices = np.arange(len(dataset_train))

# perform stratified split
train_small_idx, _ = train_test_split(
    indices,
    train_size=0.30,
    stratify=labels,
    random_state=42
)

SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

np.save(os.path.join(SPLIT_DIR, "train_small_indices.npy"), train_small_idx)

train_small_idx

# # run for ALL
# # create subset dataset from training set
dataset_train_small = Subset(dataset_train, train_small_idx)

# # run for 10
# dataset_train_small_10 = Subset(dataset_train, train_small_idx[:10])

CAPTION_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "captions",
    "captions_train_small.json"
)

# it takes 13 minutes with A100 GPU on colab
if RUN_CAPTIONING:

  device = "cuda" if torch.cuda.is_available() else "cpu"

  processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

  model = Blip2ForConditionalGeneration.from_pretrained(
      "Salesforce/blip2-opt-2.7b",
      torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # to reduce GPU memory usage
  )

  model.to(device)
  model.eval()



  captions_dict = run_captioning(
      dataset_train_small=dataset_train_small, # FOR ALL
      # dataset_train_small=dataset_train_small_10, # FOR 10
      model=model,
      processor=processor,
      device=device,
      output_path=CAPTION_PATH
  )

  del model
  del processor
  clear_gpu()
  !nvidia-smi

# check date time of last change of given file
!stat "$CAPTION_PATH"

import text_variation_flan_large
importlib.reload(text_variation_flan_large)
from text_variation_flan_large import run_text_variation as run_flan_large

MAX_ITEMS = None # None = full dataset

CAPTION_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "captions",
    "captions_train_small.json"
)

OUTPUT_PATH_FLAN_L = os.path.join(
    PROJECT_ROOT,
    "data",
    "captions",
    "captions_train_small_flan_large.json"
)

# it takes 42 mins with A100 GPU on colab
run_flan_large(
    caption_file=CAPTION_PATH,
    output_file=OUTPUT_PATH_FLAN_L,
    max_items=None
)

clear_gpu()
!nvidia-smi

from google.colab import files
files.download(OUTPUT_PATH_FLAN_L)

# it takes 24 mins with A100 GPU in colab
from text_variation_flan_xl import run_text_variation as run_flan_xl

OUTPUT_PATH_FLAN_XL = os.path.join(
    PROJECT_ROOT,
    "data",
    "captions",
    "captions_train_small_flan_xl.json"
)

run_flan_xl(
    caption_file=CAPTION_PATH,
    output_file=OUTPUT_PATH_FLAN_XL,
    max_items=None  # None = full dataset
)

clear_gpu()
!nvidia-smi

from google.colab import files
files.download(OUTPUT_PATH_FLAN_XL)

# it takes 1.30h with A100 GPU on colab
from text_variation_mistral import run_text_variation as run_mistral

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CAPTIONS_DIR = os.path.join(DATA_DIR, "captions")
TEXT_VARIATIONS_DIR = os.path.join(DATA_DIR, "text_variations")


os.makedirs(TEXT_VARIATIONS_DIR, exist_ok=True)

CAPTION_FILE = os.path.join(
    CAPTIONS_DIR,
    "captions_train_small.json"
)

TEXT_VARIATION_FILE = os.path.join(
    TEXT_VARIATIONS_DIR,
    "text_variations_train_small.json"
)

if RUN_TEXT_VARIATION:

    run_mistral(
        caption_file = CAPTION_FILE,
        output_file= TEXT_VARIATION_FILE,
        max_items=None
    )

!stat $TEXT_VARIATION_FILE

clear_gpu()
!nvidia-smi

from google.colab import files
files.download(TEXT_VARIATION_FILE)

# # force runtime termination in code
# import os
# os.kill(os.getpid(), 9)

from src.image_generation import (CaptionSelector, SyntheticImageGenerator)
import json

with open(TEXT_VARIATION_FILE, "r") as f:
    text_variations = json.load(f)

selector = CaptionSelector()

selected_data = {}

for idx, data in text_variations.items():

    class_name = data["class_name"]
    original_captions = data["original_captions"]
    generated_captions = data["generated_captions"]

    selected_generated = selector.select_top_captions(
        original_captions,
        generated_captions,
        top_k=2
    )

    selected_data[idx] = {
    "class_name": class_name,
    "original_captions": original_captions,
    "selected_generated_captions": selected_generated
}

selected_data

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

# it takes 1.45h with A100 GPU in Colab
if RUN_IMAGE_GENERATION:
        generator = SyntheticImageGenerator()

        metadata = generator.generate_images(
                selected_data = selected_data,
                output_dir = os.path.join(DATA_DIR, "synthetic/images"),
                checkpoint_file = os.path.join(DATA_DIR, "synthetic/generation_checkpoint.json"),
                final_metadata_file = os.path.join(DATA_DIR, "synthetic/generation_metadata_final.json"),
                batch_size=4
        )

        del generator
        clear_gpu()
        !nvidia-smi

# copy 20 random images generated to showcase in git
import shutil
import random

SHOWCASE_DIR = os.path.join(DATA_DIR, "showcase_images")
os.makedirs(SHOWCASE_DIR, exist_ok = True)

IMAGES_DIR = os.path.join(DATA_DIR, "synthetic/images")

# get all generated PNG images sorted
generated_images = sorted([
    os.path.join(IMAGES_DIR, f)
    for f in os.listdir(IMAGES_DIR)
    if f.endswith("png")
])

# randomly take up to 20
sample_images = random.sample(
    generated_images,
    min(20, len(generated_images))
)

# copy
for img_path in sample_images:
  shutil.copy(img_path, SHOWCASE_DIR)

# it takes 1 min with A100 GPU in colab
if RUN_TRAINING:

    results = run_training(
        PROJECT_ROOT, epochs=5, batch_size=64
    )

print("Baseline Accuracy:", round(results["baseline"]["accuracy"],3))
print("Classical Augmentation Accuracy:", round(results["classical_only"]["accuracy"],3))
print("Synthetic + Classical Augmentation Accuracy:", round(results["synthetic_plus_classical"]["accuracy"],3))

# # capture exact libraries version used in current environemnt in colab
# !pip freeze | grep -E "torch|torchvision|sentence-transformers|transformers|diffusers|accelerate|sentencepiece|scikit-learn|xformers|matplotlib|numpy|pillow|tqdm|bitsandbytes" > requirements_locked.txt


