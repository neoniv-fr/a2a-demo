# Requirements
You need uv and python version >=3.12, you can check with this command : 
echo 'import sys; print(sys.version)' | uv run -
Claude code (need NodeJS 18+)
npm install -g @anthropic-ai/claude-code


# Virtual Environment

uv venv .venv
source .venv/bin/activate
uv add git+https://github.com/djsamseng/A2A#subdirectory=samples/python --branch prefixPythonPackage
uv add click
uv add langchain langchain-ollama langgraph
uv add langchain-mcp-adapters

# Run
source .venv/bin/activate (one time)
uv run my-project --ollama-host http://127.0.0.1:11434 --ollama-model qwq
uv run google-a2a-cli --agent http://localhost:10002
