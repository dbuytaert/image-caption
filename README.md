# Image caption generator

    uv venv 
    source .venv/bin/activate 
    uv pip install -r requirements.txt
    ./update-images.py album-name --context "Brief description" --force

Generate image captions using LLMs through Simon Willison's `llm` CLI tool.

## Quickstart

```bash
# One-time setup
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set API keys
llm keys set openai
export AUTH_TOKEN=your_api_token  # or add to .env file

# Run it
./update-images.py album-name --context "Brief description of the photos"
```

This processes all images in the album directory, generates alt-text using the default model (GPT-5), and updates titles. Example:

```bash
./update-images.py chamonix-2026 --context "Ski trip in Chamonix with close friends"
```

Add `--force` to re-process images that were already verified.

## Prerequisites

1. Python 3.x
2. [uv](https://docs.astral.sh/uv/) for package management
3. Ollama (for local models):
   ```bash
   brew install ollama
   ```

## Installation

1. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Install LLM plugins:
   ```bash
   # Local models via Ollama
   uv pip install llm-ollama

   # Anthropic Claude models
   uv pip install llm-anthropic

   # Mistral models
   uv pip install llm-mistral
   ```

4. Pull required local models:
   ```bash
   ollama pull llava:13b
   ollama pull llava:34b
   ollama pull llava-llama3
   ollama pull llama3.2-vision:11b-instruct-q8_0
   ollama pull minicpm-v
   ollama pull qwen2.5vl:7b
   ollama pull qwen2.5vl:32b
   ollama pull gemma3:12b
   ollama pull gemma3:27b
   ollama pull mistral-small3.1:24b
   ollama pull llama4:16x17b
   ```

## Upgrading

```bash
uv pip install -U llm
uv pip install -U llm-ollama llm-anthropic llm-mistral
```

## API keys

```bash
# OpenAI (for GPT-5, GPT-4o)
llm keys set openai

# Anthropic (for Claude)
llm keys set anthropic

# Mistral (for Pixtral models)
llm keys set mistral
```

## Supported models

The `models.yaml` file configures model-specific parameters like prompts, temperature and token limits. You can add any vision model supported by `llm` by adding its configuration to `models.yaml`.

### Cloud models

- GPT-5 (OpenAI) - gpt-5
- GPT-4o (OpenAI) - chatgpt-4o-latest
- Claude Sonnet 4.6 (Anthropic) - anthropic/claude-sonnet-4-6
- Pixtral 12B (Mistral) - mistral/pixtral-12b-latest
- Pixtral Large (Mistral) - mistral/pixtral-large-latest

### Local models (via Ollama)

- LLaVA 13B - llava:13b
- LLaVA 34B - llava:34b
- LLaVA Llama3 - llava-llama3
- Llama 3.2 Vision (11B) - llama3.2-vision:11b-instruct-q8_0
- Llama 4 (16x17B) - llama4:16x17b
- MiniCPM-V - minicpm-v
- Qwen 2.5-VL (7B) - qwen2.5vl:7b
- Qwen 2.5-VL (32B) - qwen2.5vl:32b
- Gemma 3 (12B) - gemma3:12b
- Gemma 3 (27B) - gemma3:27b
- Mistral Small 3.1 (24B) - mistral-small3.1:24b

## Usage

### Individual image captions (caption.py)

List available models:
```bash
./caption.py --list
```

Generate captions using all models:
```bash
./caption.py path/to/image.jpg
```

Use specific models:
```bash
./caption.py path/to/image.jpg --model gpt-5 pixtral-12b
```

Add context to improve caption accuracy:
```bash
./caption.py path/to/image.jpg --context "Photo taken at DrupalCon Barcelona 2024"
```

Additional options:
```bash
--context  # Add contextual information to improve caption accuracy
--time     # Include execution time in output
--debug    # Show detailed debug information
```

### Batch image updates (update-images.py)

Set up authentication token:
```bash
export AUTH_TOKEN=your_api_token

# Or create a .env file with:
# AUTH_TOKEN=your_api_token
```

Process an entire directory of images:
```bash
./update-images.py album-name
```

Use a specific model:
```bash
./update-images.py album-name --model claude-sonnet-4-6
```

Add contextual information for better alt-text:
```bash
./update-images.py album-name --context "Event: DrupalCon Barcelona 2024"
```

Force update even for verified images:
```bash
./update-images.py album-name --force
```
