import logging

import click
#from dotenv import load_dotenv
# import google_a2a
from common.types import AgentSkill, AgentCapabilities, AgentCard
from common.server import A2AServer
from my_project.codeagent import CodeAgent
from my_project.task_manager import MyAgentTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10002)
@click.option("--ollama-host", default="http://127.0.0.1:11434")
@click.option("--ollama-model", default=None)
def main(host, port, ollama_host, ollama_model):
  skill = AgentSkill(
    id="my-project-code-skill",
    name="Code Tool",
    description="Generate or modify code, use bash commands",
    tags=["bash", "git", "code"],
    examples=["Can you solve this error ?", "what's 2x3"],
    inputModes=["text"],
    outputModes=["text"],
  )
  capabilities = AgentCapabilities(
    streaming=True, pushNotifications=True
  )
  agent_card = AgentCard(
    name="Code Agent",
    description="This agent can generate and modify code. He can also use bash to use git or other.",
    url=f"http://{host}:{port}/",
    version="0.1.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=capabilities,
    skills=[skill]
  )
  logging.info(agent_card)

  task_manager = MyAgentTaskManager(
    ollama_host=ollama_host,
    ollama_model=ollama_model,
    agent=CodeAgent()
  )
  server = A2AServer(
    agent_card=agent_card,
    task_manager=task_manager,
    host=host,
    port=port,
  )
  server.start()

if __name__ == "__main__":
  main()
