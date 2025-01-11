
# Installation guide

This project uses various Large Language Models (LLMs) to generate image captions. It provides an easy-to-use script for creating captions and comparing results from different models.

The script supports the following models:

- ViT-GPT2 (2021)
- Microsoft GIT (2022)
- BLIP Large (2022)
- BLIP-2 with OPT backbone (2023)
- BLIP-2 with FLAN-T5 backbone (2023)
- MiniCPM-V (2024)
- LLaVa  (13B) (2024)
- LLaVa (34B) (2024)
- Llama 3.2 Vision (11B, Q8) (2024)

These installation instructions were written for macOS on a MacBook. While the general steps should apply to other platforms, some commands (e.g. those using Homebrew) may require adjustments for Linux or Windows.

## Step 1: Install required tools

Python 3.11 is required for compatibility with PyTorch, the machine learning framework used by Hugging Face. While newer Python versions are available, PyTorch does not support them yet. If you have a newer Python version installed, you will need to install Python 3.11 alongside it.

This project also uses `uv`, a package and dependency manager for Python. `uv` simplifies dependency resolution and creates isolated virtual environments, keeping your system clean and preventing version conflicts.

Let's install **Python 3.11** and **uv** using Homebrew:

```bash
brew install python@3.11
brew install uv
```

## Step 2: Set up the project environment

Clone the Git repository to download the script and all its files:

```bash
git clone https://github.com/dbuytaert/image-caption
cd image-caption
```

Create a virtual environment using `uv`:

```bash
uv venv -p 3.11
```

## Step 3: Install required Python libraries

The script relies on the following Python libraries:

1. **[torch](https://pytorch.org/)**: A deep learning framework for handling neural network operations.
2. **[transformers](https://huggingface.co/docs/transformers/)**: Provides pre-trained models and tools for text and vision processing.
3. **[pillow](https://pillow.readthedocs.io/)**: Handles image processing for vision-language models.
4. **[ollama](https://github.com/ollama/ollama)**: Enables local deployment of large language models with vision support.

These dependencies are defined in `requirements.txt`. Install them in your vertual environment using the following command:

```bash
uv pip install -r requirements.txt
```

## Step 4: Install and start the Ollama service

While we interact with Ollama through its Python API, the Ollama service needs to be running locally. Install and start it with:

```bash
brew install ollama
```

```bash
ollama serve
```

Keep this service running in a separate terminal while using the captioning script.

Before using a specific LLM model, you need to pull it. For example, to pull the llava:13b model:

```bash
ollama pull llava:13b
ollama pull llava:34b
ollama pull llama3.2-vision:11b-instruct-q8_0
```

## Step 5: Running the script

Make the script executable:

```bash
chmod +x caption
```

Run the script:

```bash
# Run all models on the provided image
./caption image.jpg

# Print a list of all available models
./caption --list

# Run a single model
./caption image.jpg --model git

# Run a single model and capture executing time
./caption image.jpg --model git --time

# Run multiple models
./caption image.jpg --model blip2-flan llama32-vision-11b

# Run all models on multiple files
find . -name "*.jpg" -exec ./caption {} \;

# Run all models on multiple files and combine the output into a single JSON file
find . -name "*.jpg" -exec ./caption {} \; | jq -s '{"results": .}' > captions.json

# Same as above but capture execution time
find . -name "*.jpg" -exec ./caption {} --time \; | jq -s '{"results": .}' > captions.json
```

**Beware:** The first time you run the script, it will download the model data from Hugging Face and additional models from Ollama. This initial download is very large and may take some time depending on your internet connection. Subsequent runs will use the cached models and be much faster.

Example output:

```bash
./caption --list

Available models:
  vit-gpt2             - ViT-GPT2 (2021)
  git                  - Microsoft GIT (2022)
  blip                 - BLIP Large (2022)
  blip2-opt            - BLIP-2 with OPT backbone (2023)
  blip2-flan           - BLIP-2 with FLAN-T5 backbone (2023)
  minicpm-v            - MiniCPM-V (2024)
  llava-13b            - Large Language and Vision Assistant (13B) (2024)
  llava-34b            - Large Language and Vision Assistant (34B) (2024)
  llama32-vision-11b   - Llama 3.2 Vision (11B, Q8) (2024)
```

```bash
./caption test-images/image-1.jpg
{
  "image": "test-images/image-1.jpg",
  "captions": {
    "vit-gpt2": "A city at night with skyscrapers and a traffic light on the side of the street in front of a tall building.",
    "git": "A busy city street is lit up at night, with the word qroi on the right side of the sign.",
    "blip": "This is an aerial view of a busy city street at night with lots of people walking and cars on the side of the road.",
    "blip2-opt": "An aerial view of a busy city street at night.",
    "blip2-flan": "An aerial view of a busy street in tokyo, japanese city at night with large billboards.",
    "minicpm-v": "A bustling cityscape at night with illuminated billboards and advertisements, including one for Michael Kors.",
    "llava-13b": "A bustling nighttime scene from Tokyo's famous Shibuya Crossing, characterized by its bright lights and dense crowds of people moving through the intersection.",
    "llava-34b": "A bustling city street at night, filled with illuminated buildings and numerous pedestrians.",
    "llama32-vision-11b": "A bustling city street at night, with towering skyscrapers and neon lights illuminating the scene."
  }
}
```

From here, you can use the captions, compare them for consistency, combine them to create more accurate descriptions, or process them with another language model for translation or improvement. Get creative!

## Adding new models

Adding new models is straightforward. Open `caption.py` and look for the `MODELS` sections near the top of the script. You can easily add additional Hugging Face or Ollama models. 

If you have success with different or newer models, I'd love to hear from you! You can reach me at [dries@buytaert.net](mailto:dries@buytaert.net) or contribute by opening a ticket or pull request on GitHub.

## Managing disk space

The model data is stored at the following locations:

- **Hugging Face models**: Stored in `~/.cache/huggingface/`.
- **Ollama models**: Stored in `~/.ollama/models/`.

If you need to free up disk space later:

- For Hugging Face models: Use `transformers-cli cache clean` or manually delete `~/.cache/huggingface/`.
- For Ollama models: Remove specific models with `ollama rm [model-name]`.

## Testing

The project includes unit tests for the caption clean-up functionality. To run the tests:

```bash
# Create and activate a virtual environment
uv venv --python=python3.11
source .venv/bin/activate

# Install the dependencies in the activated environment
uv pip install -r requirements.txt

# Run the tests
python3 -m unittest test_caption.py -v
```
