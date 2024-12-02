from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models import Message, Role


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    role: str
    goal: str
    backstory: str
    model: str = Field(default="gpt-4-1106-preview")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class TaskAgent:
    """Wrapper for pydantic-ai agents with task-specific functionality."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = PydanticAgent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self._conversation_history: list[Message] = []

    async def execute(
        self,
        task_description: str,
        tools: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a task with the given tools and context."""
        # Format tools and context into a prompt
        tools_str = (
            "\nAvailable tools:\n"
            + "\n".join(f"- {name}: {result}" for name, result in (tools or {}).items())
            if tools
            else ""
        )

        context_str = (
            "\nContext:\n"
            + "\n".join(f"- {key}: {value}" for key, value in (context or {}).items())
            if context
            else ""
        )

        # Create the full prompt
        prompt = f"{task_description}{tools_str}{context_str}"

        # Add the prompt to conversation history
        self._conversation_history.append(Message(role=Role.USER, content=prompt))

        # Get response from the agent
        response = await self.agent.achat(
            messages=self._conversation_history, stream=False
        )

        # Add response to history
        self._conversation_history.append(
            Message(role=Role.ASSISTANT, content=response)
        )

        return response

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._conversation_history = []
