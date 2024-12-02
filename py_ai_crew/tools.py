import asyncio
from typing import Callable, TypeVar, ParamSpec, Optional, Any, Dict
from functools import wraps
import inspect

P = ParamSpec("P")
R = TypeVar("R")


class Tool:
    """Wrapper class for tool functions with metadata."""

    def __init__(
        self,
        func: Callable[P, R],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.is_async = asyncio.iscoroutinefunction(func)

        # Preserve function signature for better IDE support
        self.__signature__ = inspect.signature(func)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the tool function."""
        if self.is_async:
            return await self.func(*args, **kwargs)
        return self.func(*args, **kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """Return tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "is_async": self.is_async,
            "signature": str(self.__signature__),
        }


def register_tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> Callable[[Callable[P, R]], Tool]:
    """Decorator to register functions as task-compatible tools with metadata."""

    def decorator(func: Callable[P, R]) -> Tool:
        return Tool(func, name=name, description=description)

    return decorator
