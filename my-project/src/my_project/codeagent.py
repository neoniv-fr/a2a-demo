from collections.abc import AsyncIterable
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

import asyncio

logger = logging.getLogger(__name__)


memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CodeAgent:
    SYSTEM_INSTRUCTION = (
        'You are a software engineering assistant that helps with coding tasks.'
        "You can read, write and modify with Read and Write tool to solve a bug, adding a feature..."
        "You can use your tools to use git if the user asks"
        'Set response status to input_required if the user needs to provide more information.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        # self.tools = asyncio.run(self.get_tools_mcp())
        self.tools = asyncio.run(self.get_tools_mcp())
        # print("\n\nTOOLS MCP CLAUDE CODE", self.tools, "/n/n")
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )
        logger.info(self.graph.get_graph())
    
    async def get_tools_mcp(self):
        server_params = StdioServerParameters(
            command="claude",
            # Make sure to update to the full absolute path to your math_server.py file
            args=["mcp", "serve"],
            # cwd="/Users/neoniv/Documents/tutoA2A/my-project/src/my_project",
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Get tools
                tools = await load_mcp_tools(session)
                # print("\n\nTOOLS MCP CLAUDE CODE", tools, "/n/n")

        return tools

    async def ainvoke(self, query, sessionId) -> str:
        config = {'configurable': {'thread_id': sessionId}}
        self.graph.ainvoke({'messages': [('user', query)]}, config)
        return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': sessionId}}

        client = MultiServerMCPClient(
            {
                "math": {
                    "command": "claude",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["mcp", "serve"],
                    "transport": "stdio",
                },
            }
        )



        self.graph = create_react_agent(
            self.model,
            tools=await client.get_tools(),
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the exchange rates...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the exchange rates..',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if (
                structured_response.status == 'input_required'
                or structured_response.status == 'error'
            ):
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

    
    def invoke(self, query: str, sessionId: str) -> dict[str, Any]:
        """Synchronous invocation of the MCP agent."""
        raise NotImplementedError(
            'Synchronous invocation is not supported by this agent. Use the streaming endpoint (tasks/sendSubscribe) instead.'
    )