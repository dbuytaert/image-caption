#!/usr/bin/env python3

import time
import argparse
import json
import sys
import re
from pathlib import Path
import urllib.parse
import requests
from io import BytesIO
from PIL import Image
import torch
import ollama
from multiprocessing import Process, Queue
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipForConditionalGeneration,
)

# Model configurations with release dates
MODELS = {
    "vit-gpt2": {
        "architecture": "vit",
        "path": "nlpconnect/vit-gpt2-image-captioning",
        "description": "ViT-GPT2",
        "date": "2021",
        "settings": {
            "min_length": 25,
            "max_length": 75,
            "num_beams": 15,
            "repetition_penalty": 1.4,
            "do_sample": True,
            "temperature": 0.1,
        }
    },
    "git": {
        "architecture": "git",
        "path": "microsoft/git-large-textcaps",
        "description": "Microsoft GIT",
        "date": "2022",
        "settings": {
            "min_length": 25,
            "max_length": 75,
            "num_beams": 15,
            "repetition_penalty": 1.4,
            "do_sample": True,
            "temperature": 0.1,
        }
    },
    "blip": {
        "architecture": "blip",
        "path": "Salesforce/blip-image-captioning-large",
        "description": "BLIP Large",
        "date": "2022",
        "prompt": "Describe the scene in detail:",
        "settings": {
            "min_length": 25,
            "max_length": 75,
            "num_beams": 15,
            "repetition_penalty": 1.4,
            "do_sample": True,
            "temperature": 0.1,
        }
    },
    "blip2-opt": {
        "architecture": "blip2",
        "path": "Salesforce/blip2-opt-2.7b",
        "description": "BLIP-2 with OPT backbone",
        "date": "2023",
        "settings": {
            "min_length": 25,
            "max_length": 75,
            "num_beams": 15,
            "repetition_penalty": 1.4,
            "do_sample": True,
            "temperature": 0.1,
        }
    },
    "blip2-flan": {
        "architecture": "blip2",
        "path": "Salesforce/blip2-flan-t5-xl",
        "description": "BLIP-2 with FLAN-T5 backbone",
        "date": "2023",
        "settings": {
            "min_length": 25,
            "max_length": 75,
            "num_beams": 15,
            "repetition_penalty": 1.4,
            "do_sample": True,
            "temperature": 0.1,
        }
    },
    "minicpm-v": {
        "architecture": "llava",
        "path": "minicpm-v",
        "description": "MiniCPM-V",
        "date": "2024",
        "settings": {
            "prompt": "Describe the image in a single sentence.",
            "temperature": 0.1,
        }
    },
    "llava-13b": {
        "architecture": "llava",
        "path": "llava:13b",
        "description": "Large Language and Vision Assistant (13B)",
        "date": "2024",
        "settings": {
            "prompt": "Complete this sentence: 'This image shows'. Describe the image in a single sentence.",
            "temperature": 0.3,
        }
    },
    "llava-34b": {
        "architecture": "llava",
        "path": "llava:34b",
        "description": "Large Language and Vision Assistant (34B)",
        "date": "2024",
        "settings": {
            "prompt": "Complete this sentence: 'This image shows'. Describe the image in a single sentence.",
            "temperature": 0.1,
        }
    },
    "llama32-vision-11b": {
        "architecture": "llama-vision",
        "path": "llama3.2-vision:11b-instruct-q8_0",
        "description": "Llama 3.2 Vision (11B, Q8)",
        "date": "2024",
        "settings": {
            "prompt": "Complete this sentence: 'This image shows'. Describe the image in a single sentence.",
            "temperature": 0.3,
        }
    },
}

def format_error(error: Exception, context: str = None, done_reason: str = None) -> str:
    """Standardize error message format across all model types"""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"Error in {context}: {str(error)}"
    if done_reason:
        error_msg += f" (done_reason: {done_reason})"
    return error_msg

def debug_log(message: str, data: str = None, debug: bool = False) -> None:
    """Standardize debug output format"""
    if debug:
        print(f"\n--- {message} ---")
        if data:
            print(data)
        print("-" * (len(message) + 8))
        
def clean_caption(caption: str) -> str:
    # First remove any surrounding whitespace and both types of quotes (" and ')
    caption = caption.strip().strip('"\'')
    
    # Extract the first sentence:
    first_sentence = caption.split(".")[0].strip()
    
    # Define patterns for image-related subjects and verbs
    subjects = r"image|photo|photograph|picture|scene"
    verbs = r"shows|showcases|depicts|displays|features|contains|captures|presents"
    
    # Match "This is an image of..." at the start of the sentence
    pattern1 = rf"^This is an? ({subjects}) of\s+"
    first_sentence = re.sub(pattern1, "", first_sentence)
    
    # Match "This image shows..." or "The picture contains..." at the start
    pattern2 = rf"^(?:This|The)\s*(?:{subjects})\s*(?:{verbs})\s+"
    first_sentence = re.sub(pattern2, "", first_sentence)
    
    # Final cleanup: strip spaces and both types of quotes (" and '), then capitalize
    first_sentence = first_sentence.strip().strip('"\'')
    if first_sentence:
        first_sentence = first_sentence[0].upper() + first_sentence[1:]
    
    return first_sentence + "."

def load_image(image_path):
    """Load image from file or URL"""
    try:
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

def generate_huggingface_caption(model_name: str, image_path: str, debug: bool) -> str:
    """Generate caption using Hugging Face model"""
    config = MODELS[model_name]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    img = load_image(image_path)

    try:
        if config["architecture"] == "blip2":
            processor = Blip2Processor.from_pretrained(config["path"])
            model = Blip2ForConditionalGeneration.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, **config["settings"])
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        elif config["architecture"] == "blip":
            processor = AutoProcessor.from_pretrained(config["path"])
            model = BlipForConditionalGeneration.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, **config["settings"])
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        elif config["architecture"] == "vit":
            processor = ViTImageProcessor.from_pretrained(config["path"])
            model = VisionEncoderDecoderModel.from_pretrained(config["path"])
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            inputs = processor(images=img, return_tensors="pt")
            output_ids = model.generate(inputs.pixel_values, **config["settings"])
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elif config["architecture"] == "git":
            processor = AutoProcessor.from_pretrained(config["path"])
            model = AutoModelForCausalLM.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(**inputs, **config["settings"])
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            return f"Error: Unsupported model architecture: {config['architecture']}"
            
        debug_log(f"{model_name} caption", caption, debug)
        return clean_caption(caption)

    except Exception as e:
        return format_error(e, f"Hugging Face model {model_name}")

def generate_ollama_single_run(model_name: str, config: dict, image_path: str, debug: bool) -> str:
    """Execute single run for Ollama models."""
    settings = config["settings"].copy()
    prompt = settings.pop("prompt", None)

    try:
        response = ollama.generate(
            model=config["path"],
            prompt=prompt,
            images=[image_path],
            options=settings,
        )

        if not response.response:
            return format_error("Empty response", done_reason=response.done_reason)

        debug_log(f"Running {model_name}", response.response.strip(), debug)

        return clean_caption(response.response.strip())

    except Exception as e:
        return format_error(e, "Ollama run")

def generate_ollama_caption(model_name: str, image_path: str, debug: bool) -> str:
    """Generate caption using Ollama model."""
    config = MODELS[model_name]
    
    return generate_ollama_single_run(model_name, config, image_path, debug)

def generate_caption(model_name: str, image_path: str, debug: bool) -> str:
    """Generate caption using specified model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if MODELS[model_name]["architecture"] in ["llama-vision", "llava"]:
        result = generate_ollama_caption(model_name, image_path, debug)
    else:
        result = generate_huggingface_caption(model_name, image_path, debug)

    return result

def run_model_in_process(model_name, image_path, queue, debug):
    try:
        start_time = time.time()
        result = generate_caption(model_name, image_path, debug)
        model_result = {
            "caption": result,
            "time": round(time.time() - start_time)
        }
        queue.put({model_name: model_result})
    except Exception as e:
        queue.put({model_name: {"caption": f"ERROR: {str(e)}"}})

def main():
    parser = argparse.ArgumentParser(
        description="Generate image captions using various models"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all available models")
    group.add_argument("image", nargs="?", help="Path to image file or URL")

    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Specific model(s) to use. If not specified, all models will be used.",
    )
    parser.add_argument("--time", action="store_true", help="Include execution time in output")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug output for multi-run models")

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"  {name:20} - {info['description']} ({info['date']})")
        sys.exit(0)

    # If no models specified, run all
    models_to_run = args.model if args.model else MODELS.keys()

    results = {
        "image": args.image,
        "captions": {}
    }

    # Each model runs in its own Process to ensure a clean memory state, optimal
    # process isolation and avoid memory leaks. For now, we run all the models
    # sequentially to prevent out-of-memory errors.
    for model_name in models_to_run:
        # Create a queue for getting results back from each child process
        queue = Queue()
        p = Process(target=run_model_in_process, args=(model_name, args.image, queue, args.debug))
        p.start()

        # Wait for the process to complete before starting next one.
        p.join()

        # Get results from child process and add to results array
        results["captions"].update(queue.get())

    # When time is not requested, flatten the JSON
    if not args.time:
        flattened_results = {
            "image": args.image,
            "captions": {
                model: data["caption"] for model, data in results["captions"].items()
            }
        }
        print(json.dumps(flattened_results, indent=2))
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()