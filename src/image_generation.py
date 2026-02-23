import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# CAPTION SELECTION

class CaptionSelector:

    def __init__(self, embedding_model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)

    def select_top_captions(
            self,
            original_captions, 
            generated_captions, 
            top_k=2
    ):
        if len(generated_captions) <= top_k:
            return generated_captions
        
        # Encode captions
        orig_embeddings = self.model.encode(original_captions)
        gen_embeddings = self.model.encode(generated_captions)

        scores = []

        for i, gen_emb in enumerate(gen_embeddings):

            # Similarity to original captions
            sim_to_orig = cosine_similarity(
                [gen_emb], orig_embeddings
            ).mean()

            # Similarity to other generated captions (diversity)
            other_indices = [j for j in range(len(gen_embeddings)) if j != i]

            if other_indices:
                sim_to_gen = cosine_similarity(
                    [gen_emb],
                    gen_embeddings[other_indices]
                ).mean()
            else:
                sim_to_gen = 0

            # Final weighted score
            score = 0.7 * sim_to_orig - 0.3 * sim_to_gen
            scores.append(score)

        top_indices = np.argsort(scores)[-top_k:]
        selected = [generated_captions[i] for i in top_indices]

        return selected


# STABLE DIFFUSION GENERATOR
    

class SyntheticImageGenerator:

    def __init__(
        self,
        model_name = "runwayml/stable-diffusion-v1-5",
        device = None,
        seed=42
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        self.seed = seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # optimize GPU memory for colab
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                print("xformers not available. Continuing without it.")

            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        self.pipe.safety_checker = None

    def build_prompt(self, caption, class_name):
        prompt = (
            f"A high-resolution realistic photograph of a {class_name}, "
            f"{caption.lower()}"
        )

        return prompt
    
    def generate_images(
            self,
            selected_data,
            output_dir,
            checkpoint_file,
            final_metadata_file=None,
            batch_size=4
    ):
            # start clean
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)  # delete entire image folder

            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)  # delete checkpoint

            if final_metadata_file and os.path.exists(final_metadata_file):
                os.remove(final_metadata_file)  # delete final metadata
                
            os.makedirs(output_dir, exist_ok=True)

            # start fresh metadata
            generation_metadata = {}

            for idx, data in tqdm(selected_data.items()):

                class_name = data["class_name"]
                captions = data["selected_generated_captions"]

                generation_metadata[idx] = []

                # batch loop
                for batch_start in range(0, len(captions), batch_size):
                    
                    batch_captions = captions[batch_start:batch_start + batch_size]

                    prompts = [
                        self.build_prompt(caption, class_name)
                        for caption in batch_captions
                    ]

                    images = self.pipe(
                        prompts,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=self.generator
                    ).images

                    # Save each image from batch

                    for i, image in enumerate(images):

                        global_index = batch_start + i
                        image_filename = f"{idx}_{global_index}.png"
                        image_path = os.path.join(output_dir, image_filename)

                        image.save(image_path)

                        generation_metadata[idx].append({
                            "image_path": image_path,
                            "class_name": class_name,
                            "prompt": prompts[i],
                            "source": "synthetic"
                        })


                    # Save checkpoint at each batch (overwriting)
                        
                    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                    with open(checkpoint_file, "w") as f:
                        json.dump(generation_metadata, f, indent=4)

                    torch.cuda.empty_cache()

            # save final metadata (overwriting)
            if final_metadata_file is not None:
                os.makedirs(os.path.dirname(final_metadata_file), exist_ok=True)
                with open(final_metadata_file, "w") as f:
                    json.dump(generation_metadata, f, indent=4)   
                
            return generation_metadata
                    

            
        