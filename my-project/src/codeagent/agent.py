from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
import asyncio

async def get_tools_mcp():
  server_params = StdioServerParameters(
    command="claude",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["mcp", "serve"],
  )
  async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await load_mcp_tools(session)
        # print("\n\nTOOLS MCP CLAUDE CODE", tools, "/n/n")

  return tools
  
def create_ollama_agent(ollama_base_url: str, ollama_model: str):
  ollama_chat_llm = ChatOllama(
    base_url=ollama_base_url,
    model=ollama_model,
    temperature=0.2
  )

  tools=asyncio.run(get_tools_mcp())
  agent = create_react_agent(ollama_chat_llm, tools)
  return agent

async def run_ollama(ollama_agent: CompiledGraph, prompt: str):
  agent_response = await ollama_agent.ainvoke(
    {"messages": prompt }
  )
  message = agent_response["messages"][-1].content
  return str(message)
