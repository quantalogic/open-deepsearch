#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "quantalogic",
#     "streamlit",
# ]
# ///

import os
import sys
import json
import streamlit as st

# ----------------------------------------------------------------------
# IMPORTANT: Must be the very first Streamlit command!
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Search AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from quantalogic import Agent
from quantalogic.tools import (
    SerpApiSearchTool,
    ReadFileTool,
    WriteFileTool,
    ReplaceInFileTool,
    ReadHTMLTool,
    ListDirectoryTool,
    LLMTool,
)

# ==============================================================================
# Constants & Environment Check
# ==============================================================================

MAX_ITERATIONS = 10
MODEL_NAME = "openrouter/openai/gpt-4o-mini"
OUTPUT_DIRECTORY = "./results"

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("The environment variable OPENROUTER_API_KEY is not set!")
    st.stop()  # Halt execution if key is missing

# ==============================================================================
# Session State Initialization (for logs)
# ==============================================================================

if "token_log" not in st.session_state:
    st.session_state.token_log = ""
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# ==============================================================================
# Placeholders for real-time output panels
# ==============================================================================

token_placeholder = st.empty()
event_placeholder = st.empty()

# ==============================================================================
# Custom Callback Functions for Streaming Output & Events
# ==============================================================================

def streamlit_print_token(event: str, data: any = None):
    """
    Callback to update the streaming tokens panel.
    Appends each token received and re-renders the content.
    """
    if data:
        st.session_state.token_log += str(data)
        token_placeholder.markdown(
            f"""
            <div style="
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
                white-space: pre-wrap;
            ">
            {st.session_state.token_log}
            </div>
            """,
            unsafe_allow_html=True,
        )

def streamlit_print_events(event: str, data: any = None):
    """
    Callback to update the event log panel.
    Each new event is appended as a formatted HTML block.
    """
    event_entry = (
        f"Event: {event}\n"
        f"Data: {json.dumps(data, indent=2) if data else 'No data'}"
    )
    st.session_state.event_log.append(event_entry)
    combined = "<br>".join(
        [
            f"""<pre style="
                    background-color: #e8f4fd;
                    padding: 5px;
                    border-radius: 3px;
                    font-family: monospace;
                ">{entry}</pre>"""
            for entry in st.session_state.event_log
        ]
    )
    event_placeholder.markdown(combined, unsafe_allow_html=True)

def ask_for_user_validation(question: str) -> bool:
    """
    A simple user validation function.
    In this Streamlit app we automatically confirm but also show the question.
    """
    st.info(question)
    return True

# ==============================================================================
# Agent & Tools Setup
# ==============================================================================

tools = [
    SerpApiSearchTool(),
    ReadFileTool(),
    WriteFileTool(),
    ReplaceInFileTool(),
    ReadHTMLTool(),
    ListDirectoryTool(),
    LLMTool(name="report_writer", model_name=MODEL_NAME, on_token=streamlit_print_token),
]

agent = Agent(
    model_name=MODEL_NAME,
    tools=tools,
    ask_for_user_validation=ask_for_user_validation,
)

# Register our event listeners for various agent events
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
    streamlit_print_events,
)
agent.event_emitter.on("stream_chunk", streamlit_print_token)

# ==============================================================================
# Streamlit UI Layout
# ==============================================================================

def main():
    """
    Main function to run the Streamlit Deep Search application.
    This function renders the UI for entering the search subject,
    and displays live streaming output and events.
    """
    st.title("🔍 Deep Search Application")
    st.markdown(
        """
This application uses advanced AI tools to execute a comprehensive multi-source
research analysis on any topic you choose. The AI streams its output and events
in real-time below.
"""
    )

    subject_to_search = st.text_input("Enter a subject to search:", placeholder="e.g., Renewable Energy Trends")
    start_search = st.button("Start Search")

    if start_search and subject_to_search:
        # Clear previous logs when a new search starts
        st.session_state.token_log = ""
        st.session_state.event_log = []
        token_placeholder.empty()
        event_placeholder.empty()

        # Build the task prompt for deep search
        task_prompt = f"""
MISSION: Execute comprehensive multi-source research analysis on this subject: {subject_to_search}

YOU MUST COMPLETE THE SEARCH IN LESS THAN {MAX_ITERATIONS} ITERATIONS.

- Language: Primary English, include significant non-English sources if relevant

1. SEARCH About the subject 

- step1: Use a search tool to find information related to the subject
- step2: Once you find some result from search, choose which one to read to get a better understanding of the subject
- step3: Repeat the search if needed, until you have a clear understanding of the subject -> go to step1

2. ANALYSIS / SYNTHESIS REQUIREMENTS:

- Cross-reference findings
- Highlight consensus vs. controversy
- Quantify confidence levels for major claims
- Identify knowledge gaps
- Note emerging trends
- Compare geographical/cultural perspectives

3. FINAL REPORT GENERATION:

Write a final report in {OUTPUT_DIRECTORY}/:

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
        st.markdown("### Initiating Deep Search...")
        # Execute the deep search and stream the results
        with st.spinner("Processing deep search..."):
            result = agent.solve_task(task_prompt, streaming=True, max_iterations=MAX_ITERATIONS)
        st.markdown("### Final Report")
        st.markdown(result)

# ==============================================================================
# Embedded Streamlit Launcher (while preserving uv run)
# ==============================================================================
if __name__ == "__main__":
    # When using uv run, embed the streamlit command while checking an environment flag
    # to avoid re-launch loops.
    if os.environ.get("STREAMLIT_EMBEDDED") != "1":
        os.environ["STREAMLIT_EMBEDDED"] = "1"
        sys.argv = ["streamlit", "run", sys.argv[0]]
        from streamlit.web import cli as stcli
        sys.exit(stcli.main())
    else:
        # Already embedded → run the main application
        main()