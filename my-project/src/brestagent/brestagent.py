import logging

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.tools import tool
# from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters  # type: ignore
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent  # type: ignore
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class BrestExpertAgent:
    """Currency Conversion Agent Example."""

    SYSTEM_INSTRUCTION = (
        "Vous êtes un assistant expert de la ville de Brest. "
        "Votre rôle est de répondre aux questions sur Brest, sa culture, son histoire, ses lieux, ses événements, "
        "et d'aider les utilisateurs à obtenir des informations pertinentes sur la ville. "
        "Si la question ne concerne pas Brest ou ses environs, indiquez poliment que vous ne pouvez répondre qu'aux sujets liés à Brest."
    )

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as completed if the request is complete'
        'Select status as input_required if the input is a question to the user'
        'Set response status to error if the input indicates an error'
    )

    def __init__(self):
        # self.model = AzureChatOpenAI(azure_deployment="gpt-4o-mini")
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

    async def create_graph(self):
        """
        server_params = StdioServerParameters(
                command="python",
                # Make sure to update to the full absolute path to your math_server.py file
                args=["/Users/neoniv/Documents/tutoA2A/Brest-mcp-server/src/server.py"],
            )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                # Get tools
                tools = await load_mcp_tools(session)
        """
        client = MultiServerMCPClient(
            {
                "brest": {
                    "command": "python",
                    "args": ["/Users/neoniv/Documents/tutoA2A/Brest-mcp-server/src/server.py"],
                    "url": "http://localhost:3001",
                    "transport": "stdio",
                    # "transport": "streamable_http",
                },
            }
        )
        tools = await client.get_tools()
        # model = AzureChatOpenAI(azure_deployment="gpt-4o-mini")  # Azure deployment name
        model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
        # Create and run the agent
        self.agent = create_react_agent(model, tools, checkpointer=memory,prompt=self.SYSTEM_INSTRUCTION,response_format=ResponseFormat,)
            
    async def stream(
        self, query: str, sessionId: str
    ) -> AsyncIterable[dict[str, Any]]:
        inputs: dict[str, Any] = {'messages': [('user', query)]}
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}
        for item in self.agent.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': str(message),
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': str(message),
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.agent.get_state(config)

        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status in {'input_required', 'error'}:
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
