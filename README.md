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

2. Create and activate virtual environment:
   ```bash
   uv venv                     # Create virtual environment
   source .venv/bin/activate   # Activate it (Unix/macOS)
   ```

3. Install llm and verify path:
   ```bash
   uv pip install llm pyyaml   # Install packages
   which llm                   # Should show path in .venv/bin/llm
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

### Cloud models
- Claude 3 Sonnet (Anthropic)
- GPT-4 Vision (OpenAI)
- Pixtral 12B (Mistral)
- Pixtral Large (Mistral)

### Local models (via Ollama)
- LLaVA 13B and 34B
- LLaVA Llama3
- Llama 3.2 Vision (11B)
- MiniCPM-V

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