import logging

import click
from common.types import AgentSkill, AgentCapabilities, AgentCard
from common.server import A2AServer
from codeagent import CodeAgent
from task_manager import MyAgentTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10100)
def main(host, port):
  skill = AgentSkill(
    id="my-project-code-skill",
    name="Code Tool",
    description="Generate or modify code",
    tags=["bash", "code"],
    examples=["Can you solve this error ?"],
    inputModes=["text"],
    outputModes=["text"],
  )
  capabilities = AgentCapabilities(
    streaming=True, pushNotifications=True
  )
  agent_card = AgentCard(
    name="Code Agent",
    description="This agent can generate and modify code.",
    url=f"http://{host}:{port}/",
    version="0.1.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=capabilities,
    skills=[skill]
  )
  logging.info(agent_card)

  task_manager = MyAgentTaskManager(
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
