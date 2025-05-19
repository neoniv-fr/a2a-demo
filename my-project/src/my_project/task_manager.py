import asyncio
import logging
import traceback
from typing import AsyncIterable, Union

# import google_a2a
from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
  Artifact,
  InternalError,
  InvalidParamsError,
  JSONRPCResponse,
  Message,
  PushNotificationConfig,
  SendTaskRequest,
  SendTaskResponse,
  SendTaskStreamingRequest,
  SendTaskStreamingResponse,
  Task,
  TaskArtifactUpdateEvent,
  TaskIdParams,
  TaskSendParams,
  TaskState,
  TaskStatus,
  TaskStatusUpdateEvent,
  TextPart,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from my_project.agent import create_ollama_agent, run_ollama, get_tools_mcp
from my_project.codeagent import CodeAgent

logger = logging.getLogger(__name__)

class MyAgentTaskManager(InMemoryTaskManager):
  def __init__(
    self,
    ollama_host: str,
    ollama_model: Union[None, str],
    agent: CodeAgent
  ):
    super().__init__()
    """
    if ollama_model is not None:
      self.ollama_agent = create_ollama_agent(
        ollama_base_url=ollama_host,
        ollama_model=ollama_model
      )
    else:
      model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
      tools=asyncio.run(get_tools_mcp())
      # print("\n\n",tools,"\n\n")
      self.ollama_agent=create_react_agent(model,tools)
    """
    self.agent=agent

  async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
      task_send_params: TaskSendParams = request.params
      query = self._get_user_query(task_send_params)

      try:
          async for item in self.agent.stream(
              query, task_send_params.sessionId
          ):
              is_task_complete = item['is_task_complete']
              require_user_input = item['require_user_input']
              artifact = None
              message = None
              parts = [{'type': 'text', 'text': item['content']}]
              end_stream = False

              if not is_task_complete and not require_user_input:
                  task_state = TaskState.WORKING
                  message = Message(role='agent', parts=parts)
              elif require_user_input:
                  task_state = TaskState.INPUT_REQUIRED
                  message = Message(role='agent', parts=parts)
                  end_stream = True
              else:
                  task_state = TaskState.COMPLETED
                  artifact = Artifact(parts=parts, index=0, append=False)
                  end_stream = True

              task_status = TaskStatus(state=task_state, message=message)
              latest_task = await self.update_store(
                  task_send_params.id,
                  task_status,
                  None if artifact is None else [artifact],
              )
              await self.send_task_notification(latest_task)

              if artifact:
                  task_artifact_update_event = TaskArtifactUpdateEvent(
                      id=task_send_params.id, artifact=artifact
                  )
                  await self.enqueue_events_for_sse(
                      task_send_params.id, task_artifact_update_event
                  )

              task_update_event = TaskStatusUpdateEvent(
                  id=task_send_params.id, status=task_status, final=end_stream
              )
              await self.enqueue_events_for_sse(
                  task_send_params.id, task_update_event
              )

      except Exception as e:
          logger.error(f'An error occurred while streaming the response: {e}')
          await self.enqueue_events_for_sse(
              task_send_params.id,
              InternalError(
                  message=f'An error occurred while streaming the response: {e}'
              ),
          )

  def _validate_request(
      self, request: SendTaskRequest | SendTaskStreamingRequest
  ) -> JSONRPCResponse | None:
      task_send_params: TaskSendParams = request.params
      if not utils.are_modalities_compatible(
          task_send_params.acceptedOutputModes,
          CodeAgent.SUPPORTED_CONTENT_TYPES,
      ):
          logger.warning(
              'Unsupported output mode. Received %s, Support %s',
              task_send_params.acceptedOutputModes,
              CodeAgent.SUPPORTED_CONTENT_TYPES,
          )
          return utils.new_incompatible_types_error(request.id)

      if (
          task_send_params.pushNotification
          and not task_send_params.pushNotification.url
      ):
          logger.warning('Push notification URL is missing')
          return JSONRPCResponse(
              id=request.id,
              error=InvalidParamsError(
                  message='Push notification URL is missing'
              ),
          )

      return None

  async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
      """Handles the 'send task' request."""
      validation_error = self._validate_request(request)
      if validation_error:
          return SendTaskResponse(id=request.id, error=validation_error.error)

      if request.params.pushNotification:
          if not await self.set_push_notification_info(
              request.params.id, request.params.pushNotification
          ):
              return SendTaskResponse(
                  id=request.id,
                  error=InvalidParamsError(
                      message='Push notification URL is invalid'
                  ),
              )

      await self.upsert_task(request.params)
      task = await self.update_store(
          request.params.id, TaskStatus(state=TaskState.WORKING), None
      )
      await self.send_task_notification(task)

      task_send_params: TaskSendParams = request.params
      query = self._get_user_query(task_send_params)
      try:
          agent_response = await self.agent.ainvoke(
              query, task_send_params.sessionId
          )
      except Exception as e:
          logger.error(f'Error invoking agent: {e}')
          raise ValueError(f'Error invoking agent: {e}')
      return await self._process_agent_response(request, agent_response)

  async def on_send_task_subscribe(
      self, request: SendTaskStreamingRequest
  ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
      try:
          error = self._validate_request(request)
          if error:
              return error

          await self.upsert_task(request.params)

          if request.params.pushNotification:
              if not await self.set_push_notification_info(
                  request.params.id, request.params.pushNotification
              ):
                  return JSONRPCResponse(
                      id=request.id,
                      error=InvalidParamsError(
                          message='Push notification URL is invalid'
                      ),
                  )

          task_send_params: TaskSendParams = request.params
          sse_event_queue = await self.setup_sse_consumer(
              task_send_params.id, False
          )

          asyncio.create_task(self._run_streaming_agent(request))

          return self.dequeue_events_for_sse(
              request.id, task_send_params.id, sse_event_queue
          )
      except Exception as e:
          logger.error(f'Error in SSE stream: {e}')
          print(traceback.format_exc())
          return JSONRPCResponse(
              id=request.id,
              error=InternalError(
                  message='An error occurred while streaming the response'
              ),
          )

  async def _process_agent_response(
      self, request: SendTaskRequest, agent_response: dict
  ) -> SendTaskResponse:
      """Processes the agent's response and updates the task store."""
      task_send_params: TaskSendParams = request.params
      task_id = task_send_params.id
      history_length = task_send_params.historyLength
      task_status = None

      parts = [{'type': 'text', 'text': agent_response['content']}]
      artifact = None
      if agent_response['require_user_input']:
          task_status = TaskStatus(
              state=TaskState.INPUT_REQUIRED,
              message=Message(role='agent', parts=parts),
          )
      else:
          task_status = TaskStatus(state=TaskState.COMPLETED)
          artifact = Artifact(parts=parts)
      task = await self.update_store(
          task_id, task_status, None if artifact is None else [artifact]
      )
      task_result = self.append_task_history(task, history_length)
      await self.send_task_notification(task)
      return SendTaskResponse(id=request.id, result=task_result)

  def _get_user_query(self, task_send_params: TaskSendParams) -> str:
      part = task_send_params.message.parts[0]
      if not isinstance(part, TextPart):
          raise ValueError('Only text parts are supported')
      return part.text

  async def send_task_notification(self, task: Task):
      if not await self.has_push_notification_info(task.id):
          logger.info(f'No push notification info found for task {task.id}')
          return
      push_info = await self.get_push_notification_info(task.id)

      logger.info(f'Notifying for task {task.id} => {task.status.state}')
      await self.notification_sender_auth.send_push_notification(
          push_info.url, data=task.model_dump(exclude_none=True)
      )

  async def on_resubscribe_to_task(
      self, request
  ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
      task_id_params: TaskIdParams = request.params
      try:
          sse_event_queue = await self.setup_sse_consumer(
              task_id_params.id, True
          )
          return self.dequeue_events_for_sse(
              request.id, task_id_params.id, sse_event_queue
          )
      except Exception as e:
          logger.error(f'Error while reconnecting to SSE stream: {e}')
          return JSONRPCResponse(
              id=request.id,
              error=InternalError(
                  message=f'An error occurred while reconnecting to stream: {e}'
              ),
          )

  async def set_push_notification_info(
      self, task_id: str, push_notification_config: PushNotificationConfig
  ):
      # Verify the ownership of notification URL by issuing a challenge request.
      is_verified = (
          await self.notification_sender_auth.verify_push_notification_url(
              push_notification_config.url
          )
      )
      if not is_verified:
          return False

      await super().set_push_notification_info(
          task_id, push_notification_config
      )
      return True
  """
  async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
    # Upsert a task stored by InMemoryTaskManager
    await self.upsert_task(request.params)

    task_id = request.params.id
    # Our custom logic that simply marks the task as complete
    # and returns the echo text
    received_text = request.params.message.parts[0].text
    response_text = f"on_send_task received: {received_text}"
    if self.ollama_agent is not None:
      response_text = await run_ollama(ollama_agent=self.ollama_agent, prompt=received_text)

    task = await self._update_task(
      task_id=task_id,
      task_state=TaskState.COMPLETED,
      response_text=response_text
    )

    # Send the response
    return SendTaskResponse(id=request.id, result=task)

  async def _stream_3_messages(self, request: SendTaskStreamingRequest):
    task_id = request.params.id
    received_text = request.params.message.parts[0].text

    text_messages = ["one", "two", "three"]
    for text in text_messages:
      parts = [
        {
          "type": "text",
          "text": f"{received_text}: {text}",
        }
      ]
      message = Message(role="agent", parts=parts)
      task_state = TaskState.WORKING
      task_status = TaskStatus(
        state=task_state,
        message=message
      )
      task_update_event = TaskStatusUpdateEvent(
        id=request.params.id,
        status=task_status,
        final=False,
      )
      await self.enqueue_events_for_sse(
        request.params.id,
        task_update_event
      )

    ask_message = Message(
      role="agent",
      parts=[
        {
          "type": "text",
          "text": "Would you like more messages? (Y/N)"
        }
      ]
    )
    task_update_event = TaskStatusUpdateEvent(
      id=request.params.id,
      status=TaskStatus(
        state=TaskState.INPUT_REQUIRED,
        message=ask_message
      ),
      final=True,
    )
    await self.enqueue_events_for_sse(
      request.params.id,
      task_update_event
    )

  async def on_send_task_subscribe(
    self,
    request: SendTaskStreamingRequest
  ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
    task_id = request.params.id
    is_new_task = task_id in self.tasks
    # Upsert a task stored by InMemoryTaskManager
    await self.upsert_task(request.params)

    received_text = request.params.message.parts[0].text
    sse_event_queue = await self.setup_sse_consumer(task_id=task_id)
    if not is_new_task and received_text == "N":
      task_update_event = TaskStatusUpdateEvent(
        id=request.params.id,
        status=TaskStatus(
          state=TaskState.COMPLETED,
          message=Message(
            role="agent",
            parts=[
              {
                "type": "text",
                "text": "All done!"
              }
            ]
          )
        ),
        final=True,
      )
      await self.enqueue_events_for_sse(
        request.params.id,
        task_update_event,
      )
    else:
      asyncio.create_task(self._stream_3_messages(request))

    return self.dequeue_events_for_sse(
      request_id=request.id,
      task_id=task_id,
      sse_event_queue=sse_event_queue,
    )
  """

  async def _update_task(
    self,
    task_id: str,
    task_state: TaskState,
    response_text: str,
  ) -> Task:
    task = self.tasks[task_id]
    agent_response_parts = [
      {
        "type": "text",
        "text": response_text,
      }
    ]
    task.status = TaskStatus(
      state=task_state,
      message=Message(
        role="agent",
        parts=agent_response_parts,
      )
    )
    task.artifacts = [
      Artifact(
        parts=agent_response_parts,
      )
    ]
    return task
