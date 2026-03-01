"""
Text Variation Module — Mistral-7B-Instruct

This module generates caption paraphrases using a large instruction-tuned causal language model (7B parameters).

Goal:
- Maintain semantic alignment with original caption
- Increase lexical diversity
- Reduce hallucinations
- Produce natural-sounding paraphrases
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# load model

def load_mistral_model():
    """
    Load Mistral-7B-Instruct model with 4-bit quantization (memory startegy)

    Benefits:
    - Significant memory reduction
    - Allows 7B model to run on Colab GPUs
    - Maintains reasonable generation quality
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    model.eval()

    return tokenizer, model, device

# prompt builder

def build_prompt(caption):
    """
    Build instruction-style prompt for Mistral.

    The model follows the instruction format: <s>[INST] ... [/INST]

    We request two alternative rewriteswhile preserving meaning.
    """
    return f"""<s>[INST]
Rewrite the caption in two different ways.
Keep the meaning the same.

Caption: {caption}
[/INST]"""

# generation function

def generate_variations(prompt, tokenizer, model):
    """
    Generate paraphrases using controlled sampling.

    Strategy:
    - Sampling enabled for diversity
    - Moderate temperature
    - Top-p nucleus sampling
    - Extract assistant response after instruction block
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id #use the end-of-sequence token as padding when needed
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response after [/INST] toekn
    response = text.split("[/INST]")[-1].strip()

    # Split numbered lines into clean list
    variations = []

    for line in response.split("\n"):
        line = line.strip()

        if len(line) > 5:
            if line[0].isdigit():
                line = line.split(".", 1)[-1].strip()

            variations.append(line)

    return variations[:2]  # ensure exactly 2 variations

# main pipeline function

def run_text_variation(caption_file, output_file, max_items=None):
    """
    Generate text variations for entire dataset.

    Output structure:
    {
        image_id: {
            "class_name": str,
            "original_captions": [...],
            "generated_captions": [...]
        }
    }
    """
    #load caption file
    with open(caption_file, "r") as f:
        captions_dict = json.load(f)

    if max_items is not None:
        items = list(captions_dict.items())[:max_items]
    else:
        items = captions_dict.items()

    tokenizer, model, device = load_mistral_model()

    text_variations = {}

    for img, data in tqdm(items):

        class_name = data["class_name"]
        captions = list(set(data["captions"]))  # remove duplicates

        all_generated = []

        for caption in captions:

            prompt = build_prompt(caption)

            variations = generate_variations(
                prompt,
                tokenizer,
                model
            )

            all_generated.extend(variations)

        # Remove duplicates across captions
        all_generated = list(set(all_generated))

        text_variations[img] = {
            "class_name": class_name,
            "original_captions": captions,
            "generated_captions": all_generated
        }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(text_variations, f, indent=4)
        
    print(f"Mistral text variations saved to: {output_file}")

    return text_variations