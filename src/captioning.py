# src/captioning.py

import os
import json
import re
from itertools import groupby
from tqdm import tqdm
import torch

PROMPTS = [
    "Question: Describe the animal's posture and surroundings. Answer:",
    "Question: Describe what the animal is doing and its appearance. Answer:"
]

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


def generate_captions(image, class_name, model, processor, device, all_breeds_lower):
    captions = []

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
                do_sample=False
            )

        caption = processor.decode(output[0], skip_special_tokens=True)

        caption = caption.strip().lower()

        # Remove Q/A prefix
        if "answer:" in caption:
            caption = caption.split("answer:")[-1].strip()

        # Remove leading articles
        for article in ["the ", "a ", "an "]:
            if caption.startswith(article):
                caption = caption[len(article):]

        # Remove hallucinated breed names
        for breed in all_breeds_lower:
            caption = re.sub(rf"\b{breed}\b", "", caption)

        # Remove leading animal words
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

        final_caption = " ".join(
            word for word, _ in groupby(final_caption.split()) # remove consecutive duplicates
        )

        captions.append(final_caption.strip())

    return captions



def run_captioning(
    dataset_train_small,
    model,
    processor,
    device,
    output_path,
    save_every=20
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    captions_dict = {}

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

        if i % save_every == 0:
            with open(output_path, "w") as f:
                json.dump(captions_dict, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(captions_dict, f, indent=2)

    print("Full caption generation completed.")

    return captions_dict

