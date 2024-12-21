
# Installation guide

This project generates image captions using various vision-language models. It offers a simple script to create descriptive captions for images, enabling comparison across different models.

It combines two popular approaches for running vision-language models: Hugging Face Transformers and Ollama. Hugging Face provides a wide variety of machine learning models with standardized interfaces through its "transformers" library. Ollama enables optimized local deployment of large language models with vision capabilities. By supporting both platforms, we can compare different model architectures and deployment approaches.

## Step 1: Install required tools

First, install **Python 3.11** and **uv** using Homebrew:

```bash
brew install python@3.11
brew install uv
```

Python 3.11 is not the latest version of Python, but it is required for compatibility with PyTorch and some other dependencies used in this project.

**uv** is a fast Python package installer and resolver, written in Rust. It creates virtual environments much faster than traditional tools like pip or venv.

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

Install the following Python libraries needed for the project:

1. [torch](https://pytorch.org/): A deep learning framework for handling neural network operations.
2. [transformers](https://huggingface.co/docs/transformers/): Provides pre-trained models and tools for text and vision processing.
3. [pillow](https://pillow.readthedocs.io/): Handles image processing for vision-language models.
4. [ollama](https://github.com/ollama/ollama): Enables local deployment of large language models with vision support.

Install the dependencies using uv:

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
chmod +x caption.py
```

Run the script using the local virtual environment:

```bash
# Run _all_ models on the provided image (default)
./caption image.jpg

# Print a list of all available models
./caption --list

# Run a single model
./caption image.jpg --model git

# Run multiple models
./caption image.jpg --model flan llama-3b

# Get the output as JSON
./caption image.jpg --json
```

**Note:** The first time you run the script, it will download the model data from Hugging Face and additional models from Ollama. This initial download may take some time depending on your internet connection. Subsequent runs will use the cached models and be much faster.

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
