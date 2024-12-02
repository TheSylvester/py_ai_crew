"""Example of using py_ai_crew for AI blog post generation."""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent

from py_ai_crew import Task, Flow, FlowCondition, ProcessType, Goal, GoalSet


class ContentQuality(str, Enum):
    """Quality levels for content."""

    EXCELLENT = "excellent"  # 0.9 - 1.0
    GOOD = "good"  # 0.8 - 0.9
    FAIR = "fair"  # 0.7 - 0.8
    POOR = "poor"  # < 0.7


class ResearchResult(BaseModel):
    """Structured output for research task."""

    findings: list[str] = Field(description="List of key AI developments")
    impact_scores: list[int] = Field(
        description="Impact scores (1-10) for each finding"
    )
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "findings": [
                        "GPT-4 Turbo released",
                        "New breakthrough in quantum ML",
                    ],
                    "impact_scores": [9, 8],
                    "confidence": 0.95,
                }
            ]
        }
    )


class BlogPost(BaseModel):
    """Structured output for blog post task."""

    title: str = Field(description="Engaging title for the blog post")
    content: str = Field(description="Well-structured blog post content")
    keywords: list[str] = Field(description="SEO keywords for the post")
    target_audience: str = Field(description="Target audience for the post")
    estimated_quality: float = Field(
        description="Writer's estimated quality score (0-1)", ge=0, le=1
    )


class QualityCheck(BaseModel):
    """Structured output for quality check task."""

    quality_score: float = Field(description="Overall quality score (0-1)", ge=0, le=1)
    issues: list[str] = Field(description="List of identified issues")
    needs_revision: bool = Field(description="Whether the post needs revision")
    quality_level: ContentQuality = Field(description="Qualitative quality assessment")
    improvement_suggestions: list[str] = Field(
        description="Specific suggestions for improvement"
    )
    strengths: list[str] = Field(description="Content strengths")


# Initialize agents with structured outputs
research_agent = Agent(
    model="gpt-4-1106-preview",
    result_type=ResearchResult,
    system_prompt=(
        "You are an AI research analyst. Analyze the latest developments in AI technology. "
        "Focus on major breakthroughs, new models, and industry trends. "
        "Provide structured output with findings, impact scores, and confidence level. "
        "Only report findings you are highly confident about."
    ),
)

writer_agent = Agent(
    model="gpt-4-1106-preview",
    result_type=BlogPost,
    system_prompt=(
        "You are a tech writer specializing in AI. Create engaging blog posts about AI technology. "
        "Focus on clarity, accuracy, and maintaining reader interest. "
        "Estimate the quality of your work and revise until you're confident it's excellent. "
        "Consider readability, engagement, and technical accuracy in your quality estimation."
    ),
)

quality_agent = Agent(
    model="gpt-4-1106-preview",
    result_type=QualityCheck,
    system_prompt=(
        "You are a content quality analyst with extremely high standards. "
        "Review blog posts for accuracy, clarity, and engagement. "
        "Be thorough in your analysis and specific in your feedback. "
        "Focus on actionable improvements and highlight both issues and strengths."
    ),
)


# Define tasks with async execution where appropriate
research_task = Task(
    description=(
        "Research the latest AI developments in 2024. Focus on major breakthroughs, "
        "new models, and industry trends. Only include findings with high confidence. "
        "Provide impact assessment for each finding."
    ),
    agent=research_agent,
    async_execution=True,  # Research can be done asynchronously
)

write_task = Task(
    description=(
        "Write an engaging blog post about the latest AI advancements. Use the research "
        "findings to create a compelling narrative. Focus on clarity and engagement. "
        "Estimate the quality of your work and aim for excellence."
    ),
    agent=writer_agent,
    output_file="blog-posts/latest_ai_advancements.md",
)

quality_task = Task(
    description=(
        "Review the blog post for quality, accuracy, and engagement. Be thorough "
        "and specific in your feedback. Provide actionable improvements and highlight "
        "both issues and strengths."
    ),
    agent=quality_agent,
)


# Define goals
research_confidence_goal = Goal(
    name="research_confidence",
    description="Ensure research findings have high confidence",
    validation_fn=lambda result, _: (
        isinstance(result.data, ResearchResult) and result.data.confidence >= 0.8
    ),
)

content_quality_goal = Goal(
    name="content_quality",
    description="Ensure blog post meets quality standards",
    validation_fn=lambda result, _: (
        isinstance(result.data, QualityCheck)
        and result.data.quality_score >= 0.9
        and result.data.quality_level == ContentQuality.EXCELLENT
        and not result.data.needs_revision
    ),
)

# Create goal set
goals = GoalSet(require_all=True)
goals.add_goal("research", research_confidence_goal)
goals.add_goal("quality", content_quality_goal)


# Define flow conditions for revision loop
def needs_revision(result: Any) -> bool:
    """Check if content needs revision based on quality check."""
    if not isinstance(result.data, QualityCheck):
        return False
    return result.data.needs_revision


revision_condition = FlowCondition(
    task_id="2",  # Quality check task
    condition=needs_revision,
    next_task_id="1",  # Return to writing task
)


async def log_task_completion(task: Task, result: Any) -> None:
    """Log task completion with metrics and goal progress."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Task completed: {task.description}")
    print(f"Status: {task.status}")

    if isinstance(result.data, ResearchResult):
        print("\nResearch Findings:")
        for finding, impact in zip(result.data.findings, result.data.impact_scores):
            print(f"- {finding} (Impact: {impact}/10)")
        print(f"Confidence: {result.data.confidence:.2f}")

    elif isinstance(result.data, BlogPost):
        print(f"\nBlog Post: {result.data.title}")
        print(f"Target Audience: {result.data.target_audience}")
        print(f"Keywords: {', '.join(result.data.keywords)}")
        print(f"Writer's Quality Estimate: {result.data.estimated_quality:.2f}")
        print(f"\nPreview: {result.data.content[:200]}...")

    elif isinstance(result.data, QualityCheck):
        print(f"\nQuality Assessment:")
        print(f"Score: {result.data.quality_score:.2f}")
        print(f"Level: {result.data.quality_level}")
        print("\nStrengths:")
        for strength in result.data.strengths:
            print(f"✓ {strength}")
        if result.data.issues:
            print("\nIssues Found:")
            for issue in result.data.issues:
                print(f"! {issue}")
        if result.data.improvement_suggestions:
            print("\nImprovement Suggestions:")
            for suggestion in result.data.improvement_suggestions:
                print(f"→ {suggestion}")


async def main():
    """Run the example flow with hierarchical process."""
    # Create and configure flow with hierarchical process
    flow = Flow(
        tasks=[research_task, write_task, quality_task],
        memory=True,  # Enable context sharing between tasks
        cache=True,  # Enable result caching
        callbacks={"on_task_complete": log_task_completion},
        conditions=[revision_condition],
        goals=goals,
        process_type=ProcessType.HIERARCHICAL,  # Use hierarchical process
        max_retries=3,
    )

    try:
        # Execute flow
        results = await flow.kickoff()

        # Check final goal status
        failed_goals = goals.get_failed_goals()
        if failed_goals:
            print("\nFailed to achieve goals:", failed_goals)
        else:
            print("\nAll goals achieved successfully!")

    except Exception as e:
        print(f"Flow execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
