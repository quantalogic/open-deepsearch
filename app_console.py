#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "quantalogic",
# ]
# ///

import os

from quantalogic import Agent
from quantalogic.console_print_events import console_print_events
from quantalogic.console_print_token import console_print_token
from quantalogic.tools import (
    DuckDuckGoSearchTool,
    SerpApiSearchTool,
    ReadFileTool,
    ReplaceInFileTool,
    WriteFileTool,
    ReadHTMLTool,
    LLMTool,
    ListDirectoryTool
)


MAX_ITERATIONS = 10
MODEL_NAME = "openrouter/openai/gpt-4o-mini"

# Verify API key is set - required for authentication with DeepSeek's API
# This preemptive check prevents runtime failures and ensures secure API access
# We validate credentials early to maintain system reliability
if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


tools = [
# DuckDuckGoSearchTool(),
    SerpApiSearchTool(),
    ReadFileTool(),
    WriteFileTool(),
    ReplaceInFileTool(),
    ReadHTMLTool(),
    ListDirectoryTool(),
    LLMTool(name="report_writer", model_name=MODEL_NAME,on_token=console_print_token)
]


def ask_for_user_validation(question: str) -> bool:
    """Prompt the user for validation using Rich."""

    print(question)
    return True


# Initialize agent with DeepSeek model and Python tool
agent = Agent(
    model_name=MODEL_NAME, tools=tools, ask_for_user_validation=ask_for_user_validation
)

# Configure comprehensive event monitoring system
# This system is crucial for:
# - Real-time debugging and issue diagnosis
# - Performance analysis and optimization
# - Maintaining audit trails of agent activities
# The specific events tracked were chosen to provide maximum observability
agent.event_emitter.on(
    [
        "task_complete",
        "task_think_start",
        "task_think_end",
        "tool_execution_start",
        "tool_execution_end",
        "error_max_iterations_reached",
        "memory_full",
        "memory_compacted",
        "memory_summary",
    ],
    console_print_events,
)

# Register stream_chunk event listener with string instead of list
agent.event_emitter.on("stream_chunk", console_print_token)


output_directory = "./results"

subject_to_search = input("Enter a subject to search: ")


task_prompt = f"""
MISSION: Execute comprehensive multi-source research analysis on this subject: {subject_to_search}

YOU MUST COMPLETE THE SEARCH IN LESS THAN {MAX_ITERATIONS} ITERATIONS.

- Language: Primary English, include significant non-English sources if relevant

1. SEARCH About the subject 

- step1: Use a search tool to find information related to the subject
- step2: Once you find some result from search, choose which one to read to get a better understanding of the subject
- step3: Repeat the search if needed, until you have a clear understanding of the subject -> got to step1

2. ANALYSIS / SYNTHESIS REQUIREMENTS:

- Cross-reference findings
- Highlight consensus vs. controversy
- Quantify confidence levels for major claims
- Identify knowledge gaps
- Note emerging trends
- Compare geographical/cultural perspectives

3. FINAL REPORT GENERATION:

Write a final report in {output_directory}/:

## Executive Summary

- Key findings and implications
- Confidence assessment
- Critical knowledge gaps

## Methodology
- Search strategy
- Source selection criteria
- Analysis framework
- Limitations

## Findings
- Major themes
- Supporting evidence
- Contrasting views
- Statistical analysis
- Trend analysis

## Source Analysis
- Credibility assessment
- Bias evaluation
- Methodology review

## Recommendations
- Research gaps to address
- Suggested follow-up studies
- Practical applications

## Citations
- Full bibliography
- Citation metrics
- Source credibility scores

## Minimum length of the final report: at least 2000 words

Format all content using GitHub-flavored markdown with proper heading hierarchy, code blocks, tables, and emphasis formatting.
"""

# Execute a precision mathematics task to demonstrate:
# - The system's ability to handle complex computations
# - Seamless integration with PythonTool
# - Real-time monitoring capabilities for debugging
# This serves as both a functional test and capability demonstration
result = agent.solve_task(task_prompt, streaming=True, max_iterations=MAX_ITERATIONS)
print(result)
