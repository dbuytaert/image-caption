#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import urllib.parse
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import torch
import ollama
from transformers import (
   Blip2Processor, Blip2ForConditionalGeneration,
   AutoProcessor, AutoModelForCausalLM,
   VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
   logging as transformers_logging
)

# Disable transformers progress bars 
transformers_logging.set_verbosity_error()

# Model configurations with release dates
MODELS = {
   'vit': {
       'type': 'vit',
       'path': "nlpconnect/vit-gpt2-image-captioning",
       'description': "ViT-GPT2 model",
       'date': "2021"
   },
   'git': {
       'type': 'git',
       'path': "microsoft/git-large-textcaps",
       'description': "Microsoft GIT model", 
       'date': "2022"
   },
   'opt': {
       'type': 'blip2',
       'path': "Salesforce/blip2-opt-2.7b",
       'description': "BLIP-2 with OPT backbone",
       'date': "2023"
   },
   'flan': {
       'type': 'blip2',
       'path': "Salesforce/blip2-flan-t5-xl",
       'description': "BLIP-2 with FLAN-T5 backbone",
       'date': "2023"
   },
   'llama-3b': {
       'type': 'ollama',
       'path': "llama3.2-vision:latest",
       'description': "Llama 3.2 Vision (3.2B, Q4 mixed)",
       'date': "2024"
   },
   'llama-11b': {
       'type': 'ollama',
       'path': "llama3.2-vision:11b-instruct-q8_0",
       'description': "Llama 3.2 Vision (11B, Q8)",
       'date': "2024"
   }
}

def clean_caption(caption: str) -> str:
   """Clean and format caption text"""
   # Remove extra whitespace, newlines, and trailing/leading spaces
   caption = ' '.join(caption.split())
   
   # Ensure first letter is capitalized
   caption = caption[0].upper() + caption[1:]
   
   # Ensure it ends with a period
   if not caption.endswith('.'):
       caption = caption + '.'
       
   return caption.strip()

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

def generate_hf_caption(model_name: str, image_path: str) -> str:
   """Generate caption using Hugging Face model"""
   config = MODELS[model_name]
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   
   img = load_image(image_path)
   
   # Common generation parameters for longer, better captions
   gen_params = {
       'max_new_tokens': 120, # Allow for longer outputs
       'num_beams': 20,       # Beam search for better quality
       'temperature': 1.0,    # Standard temperature
       'do_sample': True      # Enable sampling for more diverse outputs
   }
   
   # Load model and processor based on type
   if config['type'] == 'blip2':
       processor = Blip2Processor.from_pretrained(config['path'])
       model = Blip2ForConditionalGeneration.from_pretrained(config['path']).to(device)
       inputs = processor(images=img, return_tensors="pt").to(device)
       output_ids = model.generate(**inputs, **gen_params, min_length=20)
       caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
       return clean_caption(caption)
       
   elif config['type'] == 'vit':
       processor = ViTImageProcessor.from_pretrained(config['path'])
       model = VisionEncoderDecoderModel.from_pretrained(config['path'])
       tokenizer = AutoTokenizer.from_pretrained(config['path'])
       inputs = processor(images=img, return_tensors="pt")
       output_ids = model.generate(inputs.pixel_values, **gen_params, min_length=20)
       caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
       return clean_caption(caption)
       
   elif config['type'] == 'git':
       processor = AutoProcessor.from_pretrained(config['path'])
       model = AutoModelForCausalLM.from_pretrained(config['path']).to(device)
       inputs = processor(images=img, return_tensors="pt")
       inputs = {k: v.to(device) for k, v in inputs.items()}
       output_ids = model.generate(**inputs, **gen_params, min_length=20)
       caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
       return clean_caption(caption)

def generate_ollama_caption(model_name: str, image_path: str) -> str:
   """Generate caption using Ollama model"""
   config = MODELS[model_name]
   
   prompt = ("Write a single-sentence description, using no more than 130 characters. Do NOT start with 'This image', 'The image', 'This photo', 'In this image'. Keep simple.")
   
   response = ollama.generate(
       model=config['path'],
       prompt=prompt,
       images=[image_path]
   )
   return clean_caption(response.response.strip())

def generate_caption(model_name: str, image_path: str) -> str:
   """Generate caption using specified model"""
   if model_name not in MODELS:
       raise ValueError(f"Unknown model: {model_name}")
       
   if MODELS[model_name]['type'] == 'ollama':
       return generate_ollama_caption(model_name, image_path)
   else:
       return generate_hf_caption(model_name, image_path)

def main():
   parser = argparse.ArgumentParser(description="Generate image captions using various models")
   group = parser.add_mutually_exclusive_group(required=True)
   group.add_argument("--list", action="store_true", help="List all available models")
   group.add_argument("image", nargs='?', help="Path to image file or URL")
   
   parser.add_argument("--model", nargs='+', choices=list(MODELS.keys()),
                      help="Specific model(s) to use. If not specified, all models will be used.")
   parser.add_argument("--json", action="store_true", help="Output in JSON format")
   args = parser.parse_args()

   if args.list:
       print("Available models:")
       for name, info in MODELS.items():
           print(f"  {name:10} - {info['description']} ({info['date']})")
       sys.exit(0)

   # If no models specified, run all
   models_to_run = args.model if args.model else MODELS.keys()
   
   results = {}
   for model_name in models_to_run:
       try:
           caption = generate_caption(model_name, args.image)
           results[model_name] = {
               "model": MODELS[model_name]['description'],
               "caption": caption
           }
       except Exception as e:
           results[model_name] = {
               "model": MODELS[model_name]['description'],
               "error": str(e)
           }

   if args.json:
       print(json.dumps(results, indent=2))
   else:
       for model_name, result in results.items():
           if "caption" in result:
               print(f"{result['model']} ({MODELS[model_name]['date']}): {result['caption']}")
           else:
               print(f"{result['model']} ({MODELS[model_name]['date']}): ERROR: {result['error']}")

if __name__ == "__main__":
   main()
