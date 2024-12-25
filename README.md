
# Installation guide

This project generates image captions using various vision-language models. It offers a simple script to create descriptive captions for images, enabling comparison across different models.

It combines two popular approaches for running vision-language models: Hugging Face Transformers and Ollama. Hugging Face provides a wide variety of machine learning models with standardized interfaces through its "transformers" library. Ollama enables optimized local deployment of large language models with vision capabilities. By supporting both platforms, we can compare different model architectures and deployment approaches.

These instructions were written for macOS (Sequoia) on a MacBook. While the general steps should apply to other platforms, some commands (e.g. those using Homebrew) may require adjustments for Linux or Windows.

## Step 1: Install required tools

Python 3.11 is required for compatibility with PyTorch. While newer Python versions are available, PyTorch does not fully support them yet. If you already have a newer Python version installed, you will need to install Python 3.11 alongside it.

This project also uses `uv`, a fast Python package and dependency manager written in Rust. `uv` simplifies dependency resolution and creates isolated virtual environments, keeping your system Python installation clean and preventing version conflicts.

Let's install **Python 3.11** and **uv** using Homebrew:

```bash
brew install python@3.11
brew install uv
```

## Step 2: Set up the project environment

Clone the repository:

```bash
git clone https://github.com/dbuytaert/image-caption
cd image-caption
```

Create a virtual environment:

```bash
uv venv -p 3.11
```

## Step 3: Install required Python libraries

The script relies on the following Python libraries:

1. **[torch](https://pytorch.org/)**: A deep learning framework for handling neural network operations.
2. **[transformers](https://huggingface.co/docs/transformers/)**: Provides pre-trained models and tools for text and vision processing.
3. **[pillow](https://pillow.readthedocs.io/)**: Handles image processing for vision-language models.
4. **[ollama](https://github.com/ollama/ollama)**: Enables local deployment of large language models with vision support.

These dependencies are listed in the `requirements.txt` file. Install them using the following command:

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

## Step 5: Running the script

Make the script executable:

```bash
chmod +x caption
```

Run the script using the local virtual environment:

```bash
# Run all models on the provided image
./caption image.jpg

# Print a list of all available models
./caption --list

# Run a single model
./caption image.jpg --model git

# Run multiple models
./caption image.jpg --model blip2-flan llama32-vision-11b-q4

# Run all models on multiple files
find . -name "*.jpg" -exec ./caption {} \;

# Run all models on multiple files and combine the output into a single JSON file:
find . -name "*.jpg" -exec ./caption {} \; | jq -s '{"results": .}' > captions.json
```

**Note:** The first time you run the script, it will download the model data from Hugging Face and additional models from Ollama. This initial download is very large and may take some time depending on your internet connection. Subsequent runs will use the cached models and be much faster.

Example output:

```bash
./caption --list

Available models:
  vit-gpt2                 - ViT-GPT2 (2021)
  git                      - Microsoft GIT (2022)
  blip                     - BLIP Large (2022)
  blip2-opt                - BLIP-2 with OPT backbone (2023)
  blip2-flan               - BLIP-2 with FLAN-T5 backbone (2023)
  llama32-vision-11b-q4    - Llama 3.2 Vision (11B, Q4 mixed) (2024)
  llama32-vision-11b-q8    - Llama 3.2 Vision (11B, Q8) (2024)
```

```bash
./caption test-images/image-1.jpg
{
  "image": "test-images/image-1.jpg",
  "captions": {
    "vit-gpt2": "A candle is lit on a wooden table in front of a fire place with candles and other items on top of it.",
    "git": "Two candles are lit next to each other on a table, one of them is lit up and the other is lit up.",
    "blip2-opt": "A candle sits on top of a wooden table.",
    "blip2-flan": "A candle sits on a wooden table next to a backgammon board and a glass of wine.",
    "blip": "There is a lit candle sitting on top of a wooden table next to a game board and a glass of wine on the table.",
    "llama32-vision-11b-q4": "The image depicts a dimly lit room with a wooden table, featuring a backgammon board and two candles.",
    "llama32-vision-11b-q8": "This photograph captures a cozy, dimly lit room with a wooden table as its central focus."
  }
}
```

The captions can be compared for consistency, programmatically combined to create a more accurate description, or processed by another language model for translation or improvement.

## Adding new models

Adding new models is straightforward. Open `caption.py` and look for the `MODELS` and `SETTINGS` sections near the top of the script. You can easily add additional Hugging Face or Ollama models in these sections. 

If you have success with different or newer models, I'd love to hear from you! You can reach me at [dries@buytaert.net](mailto:dries@buytaert.net) or contribute by opening a ticket or pull request on GitHub.

## Managing disk space

The model data is stored at the following locations:

- **Hugging Face models**: Stored in `~/.cache/huggingface/`.
- **Ollama models**: Stored in `~/.ollama/models/`. You can pre-download models by running:

If you need to free up disk space later:

- For Hugging Face models: Use `transformers-cli cache clean` or manually delete `~/.cache/huggingface/`.
- For Ollama models: Remove specific models with:

```bash
ollama rm [model-name]
```
