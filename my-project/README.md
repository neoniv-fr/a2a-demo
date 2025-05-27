# Requirements
- You need uv and python version >=3.12, you can check with this command : `echo 'import sys; print(sys.version)' | uv run -`
- Claude code (need NodeJS 18+) : `npm install -g @anthropic-ai/claude-code`


# Virtual Environment

```uv venv .venv
source .venv/bin/activate #(to connect to the environment)
uv add git+https://github.com/google/A2A#subdirectory=samples/python
uv add click
uv add langchain langchain-ollama langgraph
uv add langchain-mcp-adapters
```


# Brest Agent
Pour utiliser le Brest Agent, il faut importer le serveur mcp : 
```
git clone https://github.com/Nijal-AI/Brest-mcp-server.git
```
Ensuite, dans "my-project/src/brestagent/brestagent.py", il faut modifier la ligne 75 avec son chemin absolue vers le fichier pour lancer le serveur.

# Run
```
cd src
source .venv/bin/activate (one time)
uv run brestagent
```


# Host Agent
Pour lancer un Host Agent
```
git clone https://github.com/google/A2A.git # Une seule fois

cd A2A/demo/ui
uv run main.py
```
Ensuite allez sur localhost:12000 et allez dans Agents et connectez vos agents

Vous pouvez également utiliser les agents du repo A2A : 

```
cd A2A/samples/python/agents/<agent au choix>
uv run .
```