# src/captioning.py

"""
Caption Generation Module

This module generates structured captions for each image in the dataset using a large-scale vision-language model (BLIP-2).

Design principles:
- Controlled prompting
- Hallucination mitigation
- Class-name anchoring
- Periodic checkpoint saving
- Deterministic decoding (beam search)
"""


import os
import json
import re
from itertools import groupby
from tqdm import tqdm
import torch

# Prompt Engineering

# Two complementary prompts are used to encourage diversity:
# 1. Focus on posture & environment
# 2. Focus on action & appearance

PROMPTS = [
    "Question: Describe the animal's posture and surroundings. Answer:",
    "Question: Describe what the animal is doing and its appearance. Answer:"
]

# cat breed reference list used to determine whether the class is a cat or dog. Also used later to remove hallucinated breed names from captions.

CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx"
}

# caption generation function

def generate_captions(image, class_name, model, processor, device, all_breeds_lower):

    """
    Generate cleaned and standardized captions for a single image.

    Steps:
    1. Generate caption using controlled prompt
    2. Remove Q/A formatting artifacts
    3. Remove hallucinated breed names
    4. Normalize grammar
    5. Anchor caption to true class name

    Returns:
        List[str]: Two cleaned captions
    """

    captions = []

    # determine animal type (cat or dog)
    animal_type = "cat" if class_name in CAT_BREEDS else "dog"
    class_name_lower = class_name.lower()

    for prompt in PROMPTS:

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad(): # cause we are only generating captions (= doing inference) and not training
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                min_new_tokens=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_beams=3,
                early_stopping=True,
                do_sample=False # deterministic decoding
            )

        caption = processor.decode(output[0], skip_special_tokens=True)

        caption = caption.strip().lower()

        # Remove Q/A formatting artifacts
        if "answer:" in caption:
            caption = caption.split("answer:")[-1].strip()

        # Remove leading articles
        for article in ["the ", "a ", "an "]:
            if caption.startswith(article):
                caption = caption[len(article):]

        # Remove hallucinated breed names
        for breed in all_breeds_lower:
            caption = re.sub(rf"\b{breed}\b", "", caption)

        # Remove leading animal words (redundant prefixes)
        for animal_word in ["cat ", "dog "]:
            if caption.startswith(animal_word):
                caption = caption[len(animal_word):]

        # Remove duplicate spaces
        caption = " ".join(caption.split())

        # If caption becomes empty, fallback to generic description
        if not caption:
            final_caption = f"a {class_name_lower} {animal_type}"
        else:
            final_caption = f"a {class_name_lower} {animal_type} {caption}"

        # remove consecutive duplicate words
        final_caption = " ".join(
            word for word, _ in groupby(final_caption.split()) 
        )

        captions.append(final_caption.strip())

    return captions

# run captioning function

def run_captioning(
    dataset_train_small,
    model,
    processor,
    device,
    output_path,
    save_every=20,
    preview_samples=0
):
    
    """
    Generate captions for the entire reduced training dataset.

    Features:
    - Periodic checkpoint saving
    - Class name anchoring
    - Memory-efficient inference
    - JSON output structure indexed by original dataset index
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    captions_dict = {}
    # Collect all breed names to detect hallucinations
    all_breeds = set(dataset_train_small.dataset.classes)
    all_breeds_lower = {b.lower() for b in all_breeds}

    model.eval()
    torch.cuda.empty_cache()

    for i in tqdm(range(len(dataset_train_small))):

        original_idx = dataset_train_small.indices[i]

        img, label = dataset_train_small[i]
        img = img.convert("RGB")

        class_name = dataset_train_small.dataset.classes[label]

        captions = generate_captions(
            img,
            class_name,
            model,
            processor,
            device,
            all_breeds_lower
        )

        captions_dict[str(original_idx)] = {
            "class_name": class_name,
            "captions": captions
        }

        # Preview first N samples
        if preview_samples > 0 and i < preview_samples:
            print(f"\nSample {i+1}")
            print(f"Index: {original_idx}")
            print(f"Class: {class_name}")
            print("Caption 1:", captions[0])
            print("Caption 2:", captions[1])
            print("-" * 60)
        # periodic chekpoint saving (prevents data loss on runtime crash)
        if i % save_every == 0:
            with open(output_path, "w") as f:
                json.dump(captions_dict, f, indent=2)
    # final save
    with open(output_path, "w") as f:
        json.dump(captions_dict, f, indent=2)

    print("Full caption generation completed.")

    return captions_dict

