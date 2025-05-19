import logging

import click
from common.types import AgentSkill, AgentCapabilities, AgentCard
from common.server import A2AServer
from gitagent import GitAgent
from task_manager import MyAgentTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10101)
def main(host, port):
  skill = AgentSkill(
    id="git_skill",
    name="Git Tool",
    description="Interact with git",
    tags=["git", "push", "commit", "status"],
    examples=["Can you commit this change ?"],
    inputModes=["text"],
    outputModes=["text"],
  )
  capabilities = AgentCapabilities(
    streaming=True, pushNotifications=True
  )
  agent_card = AgentCard(
    name="Git Agent",
    description="This agent can interact with a git repository, with commit, push, clone, status, pull...",
    url=f"http://{host}:{port}/",
    version="0.1.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=capabilities,
    skills=[skill]
  )
  logging.info(agent_card)

  task_manager = MyAgentTaskManager(
    agent=GitAgent()
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
