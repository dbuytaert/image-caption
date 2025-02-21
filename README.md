# Image caption generator

This Python script generates image captions using different large language models through Simon Willison's `llm` CLI tool.

# Prerequisites

1. Python 3.x
2. Ollama (for local models):
   ```bash
   brew install ollama
   ```

# Installation steps

1. Install uv:
   ```bash
   pip install -U uv
   ```

2. Create a virtual environment:
   ```bash
   uv venv 
   ```

3. Install llm and verify path:
   ```bash
   uv pip install llm pyyaml requests   # Install packages
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

4. Install LLM plugins:
   ```bash
   # Local models via Ollama
   uv pip install llm-ollama

   # Anthropic Claude models
   uv pip install llm-anthropic

   # Mistral models
   uv pip install llm-mistral
   ```

5. Pull required local models:
   ```bash
   ollama pull llava:13b
   ollama pull llava:34b
   ollama pull llava-llama3
   ollama pull llama3.2-vision:11b-instruct-q8_0
   ollama pull minicpm-v
   ```

# Upgrading

To upgrade llm and its plugins:
```bash
uv pip install -U llm
uv pip install -U llm-ollama llm-anthropic llm-mistral
```

### Configure API keys

Set up API keys for cloud-based models:

```bash
# OpenAI (for GPT-4 Vision)
llm keys set openai

# Anthropic (for Claude)
llm keys set anthropic

# Mistral (for Pixtral models)
llm keys set mistral
```

## Supported models

This tool supports all vision and multi-modal models available through the `llm` CLI tool. The `models.yaml` file configures model-specific parameters like prompts, temperature and token limits. While several models are pre-configured, you can add any model supported by `llm` by adding its configuration to `models.yaml`. 

### Cloud models

- Claude 3 Sonnet (Anthropic) - anthropic/claude-3-sonnet-20240229
- GPT-4 Vision (OpenAI) - chatgpt-4o-latest
- Pixtral 12B (Mistral) - mistral/pixtral-12b-latest
- Pixtral Large (Mistral) - mistral/pixtral-large-latest

### Local models (via Ollama)

- LLaVA 13B - llava:13b
- LLaVA 34B - llava:34b
- LLaVA Llama3 - llava-llama3
- Llama 3.2 Vision (11B) - llama3.2-vision:11b-instruct-q8_0
- MiniCPM-V - minicpm-v


## Usage

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
./caption.py path/to/image.jpg --model chatgpt-4o-latest pixtral-12b
```

Add context to improve caption accuracy:
```bash
./caption.py path/to/image.jpg --context "Photo taken at DrupalCon Barcelona 2024"
./caption.py path/to/image.jpg --context "Location: Isle of Skye, Scotland"
```

Additional options:
```bash
--context  # Add contextual information to improve caption accuracy
--time     # Include execution time in output
--debug    # Show detailed debug information
```

## Output format

Standard output:
```json
{
  "image": "path/to/image.jpg",
  "captions": {
    "model-name": "Generated caption.",
    "another-model": "Another caption."
  }
}
```

With timing information (`--time` flag):
```json
{
  "image": "path/to/image.jpg",
  "captions": {
    "model-name": {
      "caption": "Generated caption.",
      "time": 2
    }
  }
}
```