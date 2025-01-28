# Python Chat

A terminal-based chatbot using OpenRouter API with LangChain integration. Features a hybrid memory system, multiple AI model support, and token usage tracking.

## Features

- Multiple AI model support through OpenRouter API
- Hybrid memory system using vector store for context-aware conversations
- Token usage and cost tracking
- Encrypted API key storage
- Beautiful terminal UI using Rich
- Docker support for easy deployment with pre-downloaded model

## Quick Start with Docker

The Docker image comes with the embedding model pre-installed:

```bash
# Pull the image
docker pull ghcr.io/entropy-warrior/pythonchat:latest

# Run the container
docker run -it \
  -v $(pwd)/storage/config:/app/storage/config \
  -v $(pwd)/storage/history:/app/storage/history \
  ghcr.io/entropy-warrior/pythonchat:latest
```

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Entropy-Warrior/pythonchat.git
cd pythonchat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the embedding model (not needed if using Docker):
```bash
python download_model.py
```

4. Run the application:
```bash
python pythonchat.py
```

## Environment Variables

- `MODELS_DIR`: Directory for storing ML models (default: 'storage/models')
- `CONFIG_DIR`: Directory for configuration files (default: 'storage/config')
- `HISTORY_DIR`: Directory for chat history (default: 'storage/history')

## Commands

- `exit`, `quit`: Exit the application
- `switch model`: Change the AI model
- `save`: Save current chat context
- `load`: Load saved chat context
- `clear`: Clear current chat context
- `prompt <new prompt>`: Set a new system prompt
- `debug on/off`: Toggle debug mode

## Docker Build

To build the Docker image locally (includes downloading the model):

```bash
docker build -t pythonchat .
docker run -it pythonchat
```

## License

MIT License 