import os
import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_flan_xl_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google/flan-t5-xl"

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model, device


def build_prompt(caption):
    return f"Rewrite this caption in three different ways: {caption}"


def generate_variations(prompt, tokenizer, model, device, num_variations=3):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            num_return_sequences=num_variations,
            early_stopping=True
        )

    decoded = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]

    return decoded


def run_text_variation(caption_file, output_file, max_items=None):

    with open(caption_file, "r") as f:
        captions_dict = json.load(f)

    items = list(captions_dict.items())

    if max_items is not None:
        items = items[:max_items]

    tokenizer, model, device = load_flan_xl_model()

    results = {}

    for img, data in tqdm(items):

        captions = list(set(data["captions"]))  # remove duplicates

        results[img] = []

        for caption in captions:

            prompt = build_prompt(caption)
            generated = generate_variations(prompt,tokenizer, model,device)

            results[img].append({
                "Original": caption,
                "Generated": generated})
            
    # final save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)