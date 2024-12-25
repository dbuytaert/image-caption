#!/usr/bin/env python3

import time
import argparse
import json
import sys
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

# Settings for all models
SETTINGS = {
    "huggingface": {
        "min_length": 25,
        "max_length": 75,
        "num_beams": 15,
        "repetition_penalty": 1.4,
        "do_sample": True,
        "temperature": 0.1,
    },
    "ollama": {
        "prompt": "Complete this sentence: 'This image shows'. Describe the image in a single sentence under 140 characters.",
        "temperature": 0.1,
    },
}

# Model configurations with release dates
MODELS = {
    "vit-gpt2": {
        "platform": "huggingface",
        "architecture": "vit",
        "path": "nlpconnect/vit-gpt2-image-captioning",
        "description": "ViT-GPT2",
        "date": "2021",
    },
    "git": {
        "platform": "huggingface",
        "architecture": "git",
        "path": "microsoft/git-large-textcaps",
        "description": "Microsoft GIT",
        "date": "2022",
    },
    "blip": {
        "platform": "huggingface",
        "architecture": "blip",
        "path": "Salesforce/blip-image-captioning-large",
        "description": "BLIP Large",
        "date": "2022",
        "prompt": "Describe the scene in detail:",
    },
    "blip2-opt": {
        "platform": "huggingface",
        "architecture": "blip2",
        "path": "Salesforce/blip2-opt-2.7b",
        "description": "BLIP-2 with OPT backbone",
        "date": "2023",
    },
    "blip2-flan": {
        "platform": "huggingface",
        "architecture": "blip2",
        "path": "Salesforce/blip2-flan-t5-xl",
        "description": "BLIP-2 with FLAN-T5 backbone",
        "date": "2023",
    },
    "llama32-vision-11b-q4": {
        "platform": "ollama",
        "architecture": "llama3.2-vision",
        "path": "llama3.2-vision",
        "description": "Llama 3.2 Vision (11B, Q4 mixed)",
        "date": "2024",
    },
    "llama32-vision-11b-q8": {
        "platform": "ollama",
        "architecture": "llama3.2-vision",
        "path": "llama3.2-vision:11b-instruct-q8_0",
        "description": "Llama 3.2 Vision (11B, Q8)",
        "date": "2024",
    },
    # "llama32-vision-11b-fp16": {
    #    "platform": "ollama",
    #    "architecture": "llama3.2-vision",
    #    "path": "llama3.2-vision:11b-instruct-fp16",
    #    "description": "Llama 3.2 Vision (11B, FP16)",
    #    "date": "2024",
    # },
}


def clean_caption(caption: str) -> str:
    """Clean and format caption text"""
    # Split into sentences
    sentences = caption.split(".")

    # Take only the first sentence
    first_sentence = sentences[0].strip()

    # Remove "This image shows" prefix
    prefix = "this image shows"
    if first_sentence.lower().startswith(prefix):
        first_sentence = first_sentence[len(prefix):].strip()
       
    # Remove extra whitespace and newlines
    first_sentence = " ".join(first_sentence.split())

    # Ensure first letter is capitalized
    first_sentence = first_sentence[0].upper() + first_sentence[1:]

    # Ensure it ends with a period
    if not first_sentence.endswith("."):
        first_sentence = first_sentence + "."

    return first_sentence.strip()


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


def generate_huggingface_caption(model_name: str, image_path: str) -> str:
    """Generate caption using Hugging Face model"""
    config = MODELS[model_name]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    settings = SETTINGS.get("huggingface", {})

    img = load_image(image_path)

    # print(f"Running HuggingFace model {config['architecture']} with settings {json.dumps(settings)} ...")

    try:
        if config["architecture"] == "blip":
            processor = AutoProcessor.from_pretrained(config["path"])
            model = BlipForConditionalGeneration.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, **settings)
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return clean_caption(caption)
        elif config["architecture"] == "blip2":
            processor = Blip2Processor.from_pretrained(config["path"])
            model = Blip2ForConditionalGeneration.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, **settings)
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return clean_caption(caption)
        elif config["architecture"] == "vit":
            processor = ViTImageProcessor.from_pretrained(config["path"])
            model = VisionEncoderDecoderModel.from_pretrained(config["path"])
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            inputs = processor(images=img, return_tensors="pt")
            output_ids = model.generate(inputs.pixel_values, **settings)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return clean_caption(caption)
        elif config["architecture"] == "git":
            processor = AutoProcessor.from_pretrained(config["path"])
            model = AutoModelForCausalLM.from_pretrained(config["path"]).to(device)
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(**inputs, **settings)
            caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return clean_caption(caption)
        else:
            print(
                f"Error: Unsupported architecture: {config['architecture']}",
                file=sys.stderr,
            )
    except Exception as e:
        return f"Error: {str(e)}"


def generate_ollama_caption(model_name: str, image_path: str) -> str:
    """Generate caption using Ollama model"""
    config = MODELS[model_name]
    settings = SETTINGS.get("ollama", {})
    prompt = settings.pop("prompt", None)

    # print(f"Running Ollama model {model_name} with settings: {json.dumps(settings)} ...")

    try:
        response = ollama.generate(
            model=config["path"],
            prompt=prompt,
            images=[image_path],
            options=settings,
        )

        # print(f"Ollama response: {json.dumps(response.__dict__, indent=2)}")

        if not response.response:
            return f"Error: Model returned empty response (done_reason: {response.done_reason})"

        return clean_caption(response.response.strip())

    except Exception as e:
        return f"Error: {str(e)}"


def generate_caption(model_name: str, image_path: str) -> str:
    """Generate caption using specified model."""
    result = None

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if MODELS[model_name]["platform"] == "ollama":
        result = generate_ollama_caption(model_name, image_path)
    else:
        result = generate_huggingface_caption(model_name, image_path)

    return result


def run_model_in_process(model_name, image_path, queue):
    try:
        result = generate_caption(model_name, image_path)
        queue.put({model_name: result})
    except Exception as e:
        queue.put({model_name: f"ERROR: {str(e)}"})


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
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"  {name:24} - {info['description']} ({info['date']})")
        sys.exit(0)

    # If no models specified, run all
    models_to_run = args.model if args.model else MODELS.keys()

    # Get filename from path or URL
    image_name = (
        Path(args.image).name
        if not args.image.startswith(("http://", "https://"))
        else Path(urllib.parse.urlparse(args.image).path).name
    )

    results = {"image": args.image, "captions": {}}

    # Each model runs in its own Process to ensure a clean memory state, optimal
    # process isolation and avoid memory leaks. For now, we run all the models
    # sequentially to prevent out-of-memory errors.
    for model_name in models_to_run:
        # Create a queue for getting results back from each child process
        queue = Queue()
        p = Process(target=run_model_in_process, args=(model_name, args.image, queue))
        p.start()

        # Wait for the process to complete before starting next one.
        p.join()

        # Get results from child process and add it results array.
        results["captions"].update(queue.get())

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
