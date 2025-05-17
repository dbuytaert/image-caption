#!/usr/bin/env python3

import os
import json
import time
import subprocess
import requests
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configuration
BASE_DIR = Path("/Users/dries/Dropbox/Personal/Website/images")
BASE_URL = "https://dri.es/album/"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
DEFAULT_MODEL = "chatgpt-4o-latest"
DELAY_BETWEEN_REQUESTS = 2  # seconds

def get_image_metadata(image_path):
    """Fetch metadata for an image from the website."""
    album_name = image_path.parent.name
    image_name = image_path.stem
    url = f"{BASE_URL}{album_name}/{image_name}/get"
    headers = {"Authorization": AUTH_TOKEN}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"❌ API error fetching metadata for {image_path}: {e}")
        return None

def generate_alt_text(image_path, model=DEFAULT_MODEL, context=None):
    """Generate alt-text for an image using caption.py.
    
    Args:
        image_path: Path to the image file
        model: AI model to use for generation
        context: Dictionary of contextual information with supported keys:
            - album: Album name the image belongs to
            - title: Image title
            - caption: User-provided caption
            - alt: Existing alt text
            - notes: Additional notes or information
            - Any custom keys will be formatted as "Key: value"
    """
    try:
        cmd = ["./caption.py", str(image_path), "--model", model]
        
        # Build context string from context dictionary
        if context:
            context_parts = []
            for key, value in context.items():
                if value:
                    # Format context based on key
                    if key == "album":
                        context_parts.append(f"Album: {value}")
                    elif key == "title":
                        context_parts.append(f"Title: {value}")
                    elif key == "caption":
                        context_parts.append(f"Caption: {value}")
                    elif key == "alt":
                        context_parts.append(f"Alt-text: {value}")
                    elif key == "notes":
                        context_parts.append(f"Notes: {value}")
                    else:
                        context_parts.append(f"{key.capitalize()}: {value}")
            
            if context_parts:
                context_str = "\n".join(context_parts)
                cmd.extend(["--context", context_str])
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
        output = json.loads(result.stdout)
        alt_text = output.get("captions", {}).get(model, "Alt-text generation failed")
        
        if "Error" in alt_text or "error" in alt_text:
            print(f"❌ Alt-text generation error: {alt_text}")
            return f"Error: {alt_text}"
        
        return alt_text
        
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"❌ Alt-text generation subprocess error: {str(e)}")
        print("\nDebug: Full command output:")
        print("stdout:", result.stdout if 'result' in locals() else "No stdout")
        print("stderr:", result.stderr if 'result' in locals() else "No stderr")
        return f"Error: {e}"

def generate_title(image_path, model=DEFAULT_MODEL):
    """Generate a title for an image using llm command."""
    try:
        result = subprocess.run(
            ["llm", "prompt", "-m", "chatgpt-4o-latest", "-s", "0", f"Generate a short, descriptive title for this image: {image_path.name}"],
            capture_output=True,
            text=True,
            check=True
        )
        title = result.stdout.strip()
        
        if not title:
            print(f"⚠️ No title generated")
            return "Error: Empty title"
            
        return fix_title_case(title)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ LLM error: {e}")
        print(f"   Stderr: {e.stderr}")
        return f"Error: {e}"
    
def format_title(title):
    """Format title using sentence case, proper nouns, and connecting words."""
        
    prompt = f"""Convert this title to sentence case: "{title}"
Tasks:
1. Keep all nouns from the original title; don't add new nouns
2. Capitalize first word and proper nouns (people, places, brands, events)
3. Add connecting words if needed (a, an, the, at, in, on, with, and, or)

Examples:
"xbox party richard andy" -> "Xbox party with Richard and Andy"
"sunset eiffel tower" -> "Sunset at the Eiffel Tower"
"mom dad garden" -> "Mom and Dad in the garden"

Output title only."""

    try:
        result = subprocess.run(
            ["llm", "prompt", "-m", "chatgpt-4o-latest", "-s", "0", prompt],
            capture_output=True,
            text=True,
            check=True
        )

        # Remove surrounding quotes and whitespace
        formatted_title = result.stdout.strip().strip('"\'')
        
        if not formatted_title:
            print(f"❌ Empty output from AI")
            return title
            
        return formatted_title
        
    except subprocess.CalledProcessError as e:
        print(f"❌ LLM error: {e}")
        print(f"   Stderr: {e.stderr}")
        return title

def update_image_metadata(image_path, alt_text=None, title=None):
    """Update the website with new metadata if provided."""
    album_name = image_path.parent.name
    image_name = image_path.stem
    url = f"{BASE_URL}{album_name}/{image_name}/patch"
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    
    payload = {"verified": 0}
    
    if alt_text:
        payload["alt"] = alt_text
    if title:
        payload["title"] = title
    
    try:
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ Error updating {image_name}: {e}")

def process_directory(directory, model=DEFAULT_MODEL, notes=None, force=False):
    """Process all images in a given directory.
    
    Args:
        directory: Subdirectory within BASE_DIR to process
        model: AI model to use for generation
        notes: Additional information to include as context
        force: If True, process all images regardless of verified status
    """
    directory_path = BASE_DIR / directory
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"❌ Directory {directory_path} does not exist.")
        return
    
    image_paths = list(directory_path.glob("*.jpg")) + list(directory_path.glob("*.png")) + list(directory_path.glob("*.gif"))
    
    total_images = len(image_paths)
    print(f"Found {total_images} images to process")
    
    for idx, image_path in enumerate(image_paths, 1):
        time.sleep(DELAY_BETWEEN_REQUESTS)

        album = image_path.parent.name
        image_name = image_path.stem
        print(f"\n[{idx}/{total_images}] 🔗 Image: {BASE_URL}{album}/{image_name}")
        
        metadata = get_image_metadata(image_path)
        if not metadata:
            continue
        
        title = metadata.get("title", "").strip()
        caption = metadata.get("caption", "").strip()
        alt_text = metadata.get("alt", "").strip()
        verified = metadata.get("verified")
        
        if verified == 1 and not force:
            if title:
                print(f"  💠 Skipped verified title: {title}")
            if alt_text:
                print(f"  💠 Skipped verified alt-text: {alt_text}")
            continue

        new_alt_text = None
        new_title = None

        # Build context dictionary
        context = {
            "album": album,
            "title": title,
            "caption": caption,
            "alt": alt_text,
            "notes": notes
        }

        # Generate alt-text with context dictionary
        new_alt_text = generate_alt_text(image_path, model=model, context=context)
        
        if "Error" in new_alt_text:
            print(f"❌ Alt-text generation failed, exiting.")
            return

        # Print existing alt-text if it exists
        if alt_text:
            print(f"  ℹ️ Existing alt-text: {alt_text}")
            
        print(f"  🟢 AI-suggested alt-text: {new_alt_text}")
        
        if title:
            formatted_title = format_title(title)
            if formatted_title != title:
                print(f"  🟢 AI-suggested title: {formatted_title}")
            else:
                print(f"  ℹ️ Keeping existing title: {title}")

        # Only update the photo if there are actual changes
        if new_alt_text or formatted_title != title:
           update_image_metadata(image_path, new_alt_text, formatted_title)
            
def main():
    parser = argparse.ArgumentParser(description="Generate and update image alt-texts and titles.")
    parser.add_argument("directory", help="Directory of images to process (relative to BASE_DIR)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model for alt-text generation")
    parser.add_argument("--context", help="Additional notes to include when generating alt-text")
    parser.add_argument("--force", action="store_true", help="Process all images even if they are verified")
    args = parser.parse_args()
    
    if not AUTH_TOKEN:
        print("❌ Error: AUTH_TOKEN not set. Use export AUTH_TOKEN=your_token or add to .env file")
        return
        
    process_directory(args.directory, args.model, args.context, args.force)

if __name__ == "__main__":
    main()