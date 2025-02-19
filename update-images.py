#!/usr/bin/env python3

import os
import json
import time
import subprocess
import requests
from pathlib import Path
import argparse

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

def generate_alt_text(image_path, model=DEFAULT_MODEL, album=None, title=None, caption=None, additional_context=None):
    """Generate alt-text for an image using caption.py."""
    try:
        cmd = ["./caption.py", str(image_path), "--model", model]
        
        # Build context from available metadata
        context_parts = []
        
        # Use provided album or extract from path
        final_album = album if album else image_path.parent.name.replace("-", " ").title()
        context_parts.append(f"Album: {final_album}")
        
        if title:
            context_parts.append(f"Title: {title}")
        if caption:
            context_parts.append(f"Caption: {caption}")
        if additional_context:
            context_parts.append(f"Additional context: {additional_context}")
            
        if context_parts:
            context = ". ".join(context_parts)
            cmd.extend(["--context", context])
            
        #print("Running command:")
        #print(f"  {' '.join(cmd)}")
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        #print("\nDebug: Raw stdout:")
        #print(result.stdout)
                
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

def fix_title_case(title):
    """Fix title case using sentence case (first word and proper nouns only)."""
    if not title:
        return title
        
    prompt = f"Convert this title to sentence case and don't output anything else: {title}"
    
    try:
        result = subprocess.run(
            ["llm", "prompt", "-m", "chatgpt-4o-latest", "-s", "0", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        fixed_title = result.stdout.strip()
        
        if not fixed_title or len(fixed_title.split()) != len(title.split()):
            print(f"⚠️ Invalid output: {fixed_title}")
            return title
            
        return fixed_title
        
    except subprocess.CalledProcessError as e:
        print(f"❌ LLM error: {e}")
        print(f"   Stderr: {e.stderr}")
        return title

def update_image_metadata(image_path, title, new_alt_text=None, new_title=None):
    """Update the website with new metadata if provided."""
    album_name = image_path.parent.name
    image_name = image_path.stem
    url = f"{BASE_URL}{album_name}/{image_name}/patch"
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    
    payload = {"verified": 0}
    
    if new_alt_text:
        payload["alt"] = new_alt_text
    if new_title:
        payload["title"] = new_title
    
    try:
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ Error updating {image_name}: {e}")

def process_directory(directory, model=DEFAULT_MODEL, additional_context=None):
    """Process all images in a given directory."""
    directory_path = BASE_DIR / directory
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"❌ Directory {directory_path} does not exist.")
        return
    
    image_paths = list(directory_path.glob("*.jpg"))
    total_images = len(image_paths)
    print(f"Found {total_images} images to process")
    
    for idx, image_path in enumerate(image_paths, 1):
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
        
        if verified == 1:
            if title:
                print(f"  💠 Skipped verified title: {title}")
            if alt_text:
                print(f"  💠 Skipped verified alt-text: {alt_text}")
            continue

        new_alt_text = None
        new_title = None

        # Generate alt-text regardless of whether caption exists
        # Use caption as context but don't skip alt-text generation
        new_alt_text = generate_alt_text(
            image_path, 
            model=model,
            album=album,
            title=title,
            caption=caption,
            additional_context=additional_context
        )
        if "Error" in new_alt_text:
            print(f"❌ Alt-text generation failed, exiting.")
            return
        print(f"  🟢 AI-suggested alt-text: {new_alt_text}")
        
        # Show existing caption for reference
        if caption:
            print(f"  ℹ️ Existing caption (for context only): {caption}")

        if title:
            fixed_title = fix_title_case(title)
            if fixed_title != title:
                new_title = fixed_title
                print(f"  🟢 AI-suggested title: {new_title}")
            else:
                print(f"  ℹ️ Keeping existing title: {title}")

        # Only update the photo if there are actual changes
        if new_alt_text or new_title:
            update_image_metadata(image_path, title, new_alt_text, new_title)
        
        time.sleep(DELAY_BETWEEN_REQUESTS)

def main():
    parser = argparse.ArgumentParser(description="Generate and update image alt-texts and titles.")
    parser.add_argument("directory", help="Directory of images to process (relative to BASE_DIR)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model for alt-text generation")
    parser.add_argument("--context", help="Additional context to include when generating alt-text")
    args = parser.parse_args()
    
    process_directory(args.directory, args.model, args.context)

if __name__ == "__main__":
    main()