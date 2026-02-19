import os
import sys
import json


PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from captioning import run_captioning
from text_variation_flan_large import run_text_variation as run_flan_large


# =============================
# CONTROLLED VERBOSITY
# =============================

# Disable HF download progress bars BEFORE importing anything HF-related

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import torch
import logging
import numpy as np
import random
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers.utils import logging as transformers_logging
from huggingface_hub.utils import logging as hf_logging


# MOCK CAPTIONING (DEV MODE)

def mock_captioning(output_path):

    print("DEV MODE: Generating mock captions...")

    captions_dict = {
        "0": {
            "class_name": "Pomeranian",
            "captions": [
                "a pomeranian dog this dog is sitting on the bed",
                "a pomeranian dog my dog is sitting on the bed"
            ]
        },
        "1": {
            "class_name": "Havanese",
            "captions": [
                "a havanese dog is sitting on a tennis court",
                "a havanese dog is sitting on the tennis court"
            ]
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(captions_dict, f, indent=4)

    return captions_dict


# MAIN

if __name__ == "__main__":

    # Silence transformers & HF logs (keep only errors)
    transformers_logging.set_verbosity_error()
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DEV_MODE = device == "cpu"

    '''
    # control flags
    RUN_CAPTIONING = True
    RUN_TEXT_VARIATION = False
    RUN_IMAGE_GENERATION = False
    RUN_TRAINING = False
    '''


    # DATASET

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


    # # run for ALL
    # # create subset dataset from training set
    dataset_train_small = Subset(dataset_train, train_small_idx)

    # run for 10
    dataset_train_small_10 = Subset(dataset_train, train_small_idx[:10])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CAPTIONING

    CAPTION_PATH = os.path.join(PROJECT_ROOT,
                                "data",
                                "captions",
                                "captions_train_small_10.json")

    if not os.path.exists(CAPTION_PATH):
                
                if DEV_MODE:
                    captions_dict = mock_captioning(CAPTION_PATH)
                
                else:
                    print("Running BLIP2 Captioning...")
                     
                    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

                    model = Blip2ForConditionalGeneration.from_pretrained(
                        "Salesforce/blip2-opt-2.7b",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )

                    model.to(device)
                    model.eval()

                    captions_dict = run_captioning(
                        # dataset_train_small=dataset_train_small, # FOR ALL
                        dataset_train_small=dataset_train_small_10, # FOR 10
                        model=model,
                        processor=processor,
                        device=device,
                        output_path=CAPTION_PATH
                    )

    else:
         print("Captions already exist. Skipping caption generation.")

    # TEXT VARIATION
    
    print("\n=== FLAN LARGE TEXT VARIATION===")

    if DEV_MODE:
         print("DEV MODE: ")
    run_flan_large(
        caption_file=CAPTION_PATH,
        max_items=10
    )