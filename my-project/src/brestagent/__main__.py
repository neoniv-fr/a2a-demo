import logging

import click
from common.types import AgentSkill, AgentCapabilities, AgentCard
from common.server import A2AServer
from brestagent import BrestExpertAgent
from task_manager import MyAgentTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10030)
def main(host, port):
  skill = AgentSkill(
    id="brest_skill",
    name="Brest tool",
    description="Obtenir des informations sur Brest",
    tags=["Brest", "culture", "histoire", "lieu", "événement"],
    examples=["Où se trouve la gare ?"],
    inputModes=["text"],
    outputModes=["text"],
  )
  capabilities = AgentCapabilities(
    streaming=True, pushNotifications=True
  )
  agent_card = AgentCard(
    name="Brest Expert Agent",
    description="Cet agent peut répondre à des questions sur la ville de Brest",
    url=f"http://{host}:{port}/",
    version="0.1.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=capabilities,
    skills=[skill]
  )
  logging.info(agent_card)

  task_manager = MyAgentTaskManager(
    agent=BrestExpertAgent()
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
