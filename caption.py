#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import re
import yaml
from PIL import Image
from pathlib import Path
import time
from collections import defaultdict
import sys
from tempfile import gettempdir

sys.path.append('/opt/homebrew/lib/python3.12/site-packages')

def load_models():
    """Load models from models.yaml and check their status."""
    try:
        # Load config
        config_path = Path(__file__).parent / "models.yaml"
        with open(config_path) as f:
            models = yaml.safe_load(f)
            
        # Get installed models from CLI
        result = subprocess.run(["llm", "models"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running 'llm models': {result.stderr}")
            raise Exception("Failed to get model list")
            
        # Parse output to get installed models
        installed_models = set()
        for line in result.stdout.split('\n'):
            if ':' in line:
                _, model_part = line.split(':', 1)
                # Take the first part before any aliases
                model_id = model_part.split('(')[0].strip().lower()
                if model_id:
                    installed_models.add(model_id)
        
        # print("DEBUG - Installed models:", installed_models)

        # Check each model's status
        for name, config in models.items():
            model_id = config['model'].lower()
            config['installed'] = model_id in installed_models
            config['configured'] = True  # If llm models works, assume configured
            
        return models
        
    except FileNotFoundError:
        print("Error: models.yaml not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing models.yaml: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking model status: {e}")
        return {name: dict(config, installed=False, configured=False) 
                for name, config in models.items()}

def list_models(models):
    """Display models grouped by provider with installation and configuration status."""
    # ANSI color codes
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"
    
    print("\nModel status:")
    
    by_provider = defaultdict(list)
    for name, config in models.items():
        # Better provider detection
        if 'provider' not in config:
            if 'llava' in name.lower():
                config['provider'] = 'ollama'
                
        provider = config.get("provider", "other").upper()
        
        # Simpler status check - just installed or not
        if config['installed']:
            status = ""
            symbol = f"{GREEN}✓{RESET}"
        else:
            status = "not installed"
            symbol = f"{RED}✗{RESET}"
            
        by_provider[provider].append((name, config, status, symbol))
    
    for provider in sorted(by_provider):
        print(f"\n{provider} Models:")
        for name, info, status, symbol in sorted(by_provider[provider]):
            desc_section = f"{info['description']} ({info['deployment']})"
            status_section = f" - {status}" if status else ""
            print(f"  {symbol} {name:15} - {desc_section:35}{status_section}")

def model_is_ready(config):
    """Check if a model is ready to use."""
    return config['installed'] and config['configured']

def clean_caption(caption: str) -> str:
    """Clean caption by extracting first sentence and removing image references."""
    # First remove any surrounding whitespace and quotes
    caption = caption.strip().strip("\"'")

    # Extract the first sentence
    first_sentence = caption.split(".")[0].strip()

    # Define patterns for image-related subjects and verbs
    subjects = r"image|photo|photograph|picture|scene"
    verbs = r"shows|showcases|depicts|displays|features|contains|captures|presents"

    # Match "This is an image of..." at the start
    pattern1 = rf"^This is an? ({subjects}) of\s+"
    first_sentence = re.sub(pattern1, "", first_sentence)

    # Match "This image shows..." or similar at the start
    pattern2 = rf"^(?:This|The)\s*(?:{subjects})\s*(?:{verbs})\s+"
    first_sentence = re.sub(pattern2, "", first_sentence)

    # Final cleanup and capitalize
    first_sentence = first_sentence.strip().strip("\"'")
    if first_sentence:
        first_sentence = first_sentence[0].upper() + first_sentence[1:]

    return first_sentence + "."

def verify_image_path(image_path: str) -> bool:
    """Verify that the image path exists and is accessible."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: image {image_path} not found")
        return False
    if not path.is_file():
        print(f"Error: {image_path} is not a file")
        return False
    return True

def resize_image(image_path: str, debug: bool = False, max_dimension: int = 1024) -> str:
    """Return path to a resized image for LLM processing."""
    path = Path(image_path)
    with Image.open(path) as img:
        # Return original path if image is small enough
        if max(img.size) <= max_dimension:
            return image_path
            
        # Resize while preserving aspect ratio
        img.thumbnail((max_dimension, max_dimension))
        
        # Create temp file path with original extension
        temp_path = Path(gettempdir()) / f"resized-llm-image{path.suffix}"
        
        # Save resized image to temp location
        img.save(temp_path, optimize=True)
        return str(temp_path)
    
def process_image(image_path: str, models_to_use: dict, models_to_run: list,  args: argparse.Namespace) -> dict:
    """Process an image with specified models sequentially."""
    start_time = time.time()
    
    if args.debug:
        print(f"\nRunning caption generation for {len(models_to_run)} models...")
    
    results = {
        "image": image_path,
        "captions": {}
    }
    
    # Resize large images to reduce upload bandwidth to cloud LLMs
    small_image = resize_image(image_path, args.debug)
    
    for model_name in models_to_run:
        model_config = models_to_use[model_name]
        result = run_llm_command(small_image, model_config, args.context, args.debug)
        
        if args.time:
            results["captions"][model_name] = result
        else:
            results["captions"][model_name] = result["caption"]
        
        # Small delay between models to allow resources to be released
        time.sleep(1)
    
    if args.debug:
        total_time = round(time.time() - start_time, 1)
        print("\n" + "="*80)
        print(f"Total execution time: {total_time}s\n")
    
    return results

def run_llm_command(image_path: str, model_config: dict, context: str = None, debug: bool = False) -> dict:
    """Run llm command for a specific model and return the result.
    
    Uses subprocess instead of the Python API for model execution because:
    1. Memory isolation - Each model runs in a separate process, preventing memory leaks 
       from accumulating in the main process
    2. Resource cleanup - Process termination ensures complete cleanup of model resources, 
       especially important with large vision models
    3. Fault isolation - A model crash only affects its own process
    """
    start_time = time.time()
    
    try:
        # Base command
        cmd = ["llm", "-m", model_config["model"]]
        
        # Add attachment for image
        cmd.extend(["-a", str(image_path)])
        
        # Build prompt with context if provided
        prompt = model_config["prompt"]
        if context:
            prompt = f"Consider this context before analyzing the image: {context}\n\n{prompt}"
        
        # Add prompt to command
        cmd.append(prompt)
        
        # Add any model-specific settings
        if "settings" in model_config:
            for key, value in model_config["settings"].items():
                cmd.extend(["-o", key, str(value)])
        
        if debug:
            print("\n" + "="*80)
            print(f"Model: {model_config['model']}")
            print(f"Image: {image_path}")
            print("-"*80)
            
            # Build and show the exact command with settings
            settings_str = ""
            if "settings" in model_config:
                settings_str = " " + " ".join(f"-o {k} {v}" for k, v in model_config["settings"].items())
            print(f"Command:")
            print(f"  llm -m {model_config['model']} -a {image_path}{settings_str}")
            print("-"*80)
            
            print("Prompt details:")
            if context:
                print("  Context provided:")
                print(f"    {context}")
            if "settings" in model_config:
                print("  Settings:")
                for key, value in model_config["settings"].items():
                    print(f"    {key}: {value}")
            print("-"*80)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        execution_time = round(time.time() - start_time, 1)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        
        raw_caption = result.stdout.strip()
        caption = clean_caption(raw_caption)
        
        if debug:
            print(f"Generated caption ({execution_time}s):")
            print(f"  Raw: {raw_caption}")
            print(f"  Clean: {caption}")
            print("="*80)
        
        return {
            "caption": caption,
            "time": execution_time
        }
        
    except subprocess.CalledProcessError as e:
        if debug:
            print(e.stderr, file=sys.stderr)
        return {"caption": e.stderr.strip()}
    except Exception as e:
        error_msg = str(e)
        if debug:
            print(error_msg, file=sys.stderr)
        return {"caption": error_msg}

def main():
    # Load models with status information
    models = load_models()
    
    parser = argparse.ArgumentParser(
        description="Generate image captions using llm CLI with various models"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all available models")
    group.add_argument("image", nargs="?", help="Path to image file or URL")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(models.keys()),
        help="Specific model(s) to use. If not specified, all models will be used."
    )
    parser.add_argument("--time", action="store_true", help="Include execution time in output")
    parser.add_argument("--debug", action="store_true", help="Show debug info (see README.md)")
    parser.add_argument(
        "--context",
        type=str,
        help="Additional context to help generate more accurate captions (e.g., title, location, date)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models(models)
        return
        
    if not verify_image_path(args.image):
        sys.exit(1)
    
    # Determine which models to run
    models_to_run = args.model if args.model else list(models.keys())

    # Create dict of just the models we'll run
    models_to_use = {name: {k:v for k,v in models[name].items() 
                           if k != 'installed'}
                     for name in models_to_run}
    
    # Process image
    results = process_image(args.image, models_to_use, models_to_run, args)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()