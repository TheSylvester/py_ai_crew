# py_ai_crew

A hierarchical task execution framework for AI agents with goal-based validation and quality control.

## Installation

You can install the package using pip:

```bash
# Install basic package
pip install py_ai_crew

# Install with development dependencies
pip install py_ai_crew[dev]
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/TheSylvester/py_ai_crew.git
cd py_ai_crew

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## Core Features

### Hierarchical Task Management

- **Manager-Based Oversight**: Intelligent task delegation and quality control through manager agents
- **Dynamic Assessment**: Real-time task evaluation and execution path modification
- **Quality Validation**: Built-in validation cycles with feedback loops
- **Flexible Delegation**: Smart task routing based on agent capabilities and context

### Goal-Based Execution

- **Structured Goals**: Define and track specific execution goals
- **Validation Functions**: Custom validation logic for goal achievement
- **Progress Tracking**: Monitor goal status and attempts
- **Multi-Goal Coordination**: Handle multiple interdependent goals
- **Early Termination**: Stop execution when goals are unachievable

### Advanced Flow Control

- **Process Types**:
  - Sequential execution with dependency management
  - Hierarchical execution with manager oversight
- **Conditional Logic**:
  - Dynamic branching based on results
  - Retry mechanisms with state tracking
  - Custom flow conditions
- **Execution Modes**:
  - Synchronous and asynchronous task execution
  - Parallel task processing
  - Cached results for efficiency

### State and Context Management

- **Execution State**:
  - Task status tracking
  - Run metrics collection
  - Error handling and recovery
- **Context Handling**:
  - Inter-task dependency management
  - Shared context updates
  - Memory management across tasks
- **Caching**:
  - Result caching with smart invalidation
  - Performance optimization

## Architecture

### Core Components

#### Flow (`flow.py`)

```python
class Flow:
    """Manages sequences of tasks with conditional branching, loops, and goal-based execution."""
    # Handles task orchestration, execution flow, and process management
```

#### Task (`models.py`)

```python
class Task:
    """A task to be executed by an agent."""
    # Defines task structure, execution logic, and output handling
```

#### Goals (`goals.py`)

```python
class Goal:
    """Base class for defining execution goals."""
    # Manages goal definition, validation, and tracking
```

#### TaskExecutor (`executor.py`)

```python
class TaskExecutor:
    """Handles task execution with built-in evaluation capabilities."""
    # Manages individual task execution and metrics tracking
```

### Integration with pydantic-ai

The framework extends pydantic-ai's capabilities while maintaining clean separation:

- Uses pydantic-ai's Agent for core execution
- Extends RunResult for enhanced output handling
- Leverages dependency injection system
- Adds higher-level orchestration features


## Usage Examples

### Basic Task Flow

```python
from py_ai_crew import Flow, Task, Goal

# Define tasks
research_task = Task(
    description="Research AI developments",
    agent=research_agent,
    async_execution=True
)

write_task = Task(
    description="Write summary report",
    agent=writer_agent
)

# Create flow
flow = Flow(
    tasks=[research_task, write_task],
    memory=True,
    process_type=ProcessType.HIERARCHICAL
)

# Execute
results = await flow.kickoff()
```

### Goal-Based Execution

```python
# Define goal
quality_goal = Goal(
    name="content_quality",
    description="Ensure high-quality output",
    validation_fn=lambda result, _: result.data.quality_score >= 0.9
)

# Create goal set
goals = GoalSet(require_all=True)
goals.add_goal("quality", quality_goal)

# Configure flow with goals
flow = Flow(
    tasks=[write_task, review_task],
    goals=goals,
    max_retries=3
)
```

### Advanced Features

#### Conditional Branching

```python
# Define condition
revision_condition = FlowCondition(
    task_id="review",
    condition=lambda result: result.data.needs_revision,
    next_task_id="revise"
)

# Add to flow
flow = Flow(
    tasks=[write_task, review_task, revise_task],
    conditions=[revision_condition]
)
```

#### Custom Callbacks

```python
async def log_completion(task: Task, result: Any):
    print(f"Task completed: {task.description}")
    print(f"Result: {result.data}")

flow = Flow(
    tasks=[task],
    callbacks={"on_complete": log_completion}
)
```

## Best Practices

1. **Task Design**

   - Keep tasks atomic and focused
   - Use appropriate async/sync execution
   - Provide clear descriptions and expected outputs

2. **Goal Definition**

   - Define specific, measurable goals
   - Use appropriate validation functions
   - Set realistic max_attempts

3. **Flow Configuration**

   - Choose appropriate process type
   - Enable memory for dependent tasks
   - Configure caching based on needs

4. **Error Handling**
   - Implement task-specific error handling
   - Use appropriate retry strategies
   - Monitor goal progress

## Examples

Check out the [examples](examples/) directory for more detailed examples:

- [Blog Generation](examples/basic/blog_generation.py): A complete example of hierarchical task execution for AI blog post generation
- More examples coming soon...

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to this project.

## Author

Sylvester Wong (sylvester@thesylvester.ca)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
