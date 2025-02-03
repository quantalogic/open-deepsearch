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
# Set page config as the very first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Search AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Optional: inject custom CSS for an improved UX
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall app font */
    .stApp { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    /* Input field styling */
    .stTextInput>div>div>input {
        padding: 0.5rem;
        font-size: 1rem;
    }
    /* Add spacing for containers */
    .container { margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
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
# Global placeholders for live logs (will be set inside main())
# ==============================================================================
token_placeholder = None
event_placeholder = None

# ==============================================================================
# Custom Callback Functions for Streaming Output & Events
# ==============================================================================

def streamlit_print_token(event: str, data: any = None):
    """
    When the agent streams tokens, update the Live Output panel.
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
                max-height: 300px;
                overflow-y: auto;
            ">
            {st.session_state.token_log}
            </div>
            """,
            unsafe_allow_html=True,
        )

def streamlit_print_events(event: str, data: any = None):
    """
    When the agent emits an event, update the Event Log panel.
    """
    event_entry = (
        f"<strong>Event:</strong> {event}<br>"
        f"<strong>Data:</strong> {json.dumps(data, indent=2) if data else 'No data'}"
    )
    st.session_state.event_log.append(event_entry)
    combined = "<br><br>".join(
        [
            f"""<div style="
                    background-color: #e8f4fd;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: monospace;
                    white-space: pre-wrap;
                    max-height: 300px;
                    overflow-y: auto;
                ">{entry}</div>"""
            for entry in st.session_state.event_log
        ]
    )
    event_placeholder.markdown(combined, unsafe_allow_html=True)

def ask_for_user_validation(question: str) -> bool:
    """
    This function displays the validation question and auto-confirms.
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

# Register event listeners for various agent events
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
    global token_placeholder, event_placeholder

    st.title("🔍 Deep Search Application")
    st.subheader("Comprehensive Multi-Source Research Analysis")
    st.markdown(
        """
Welcome to the Deep Search application. Enter a subject below to begin an in-depth, multi-source research analysis.
        """
    )

    # ------------------------------------------------------------------
    # Search Input Form
    # ------------------------------------------------------------------
    with st.container():
        subject_to_search = st.text_input("Search Subject", placeholder="e.g., Renewable Energy Trends")
        start_search = st.button("Start Search")

    st.markdown("---")
    
    # ------------------------------------------------------------------
    # Tabs for output organization
    # ------------------------------------------------------------------
    tabs = st.tabs(["Live Output", "Event Log", "Final Report"])
    token_placeholder = tabs[0].empty()
    event_placeholder = tabs[1].empty()

    if start_search and subject_to_search:
        # Reset the logs
        st.session_state.token_log = ""
        st.session_state.event_log = []
        token_placeholder.empty()
        event_placeholder.empty()

        # Build the deep search task prompt
        task_prompt = f"""
MISSION: Execute comprehensive multi-source research analysis on this subject: {subject_to_search}

YOU MUST COMPLETE THE SEARCH IN LESS THAN {MAX_ITERATIONS} ITERATIONS.

- Language: Primary English, include significant non-English sources if relevant

1. SEARCH About the subject 
   - step1: Use a search tool to find information related to the subject
   - step2: Once you find a search result, choose one to get a better understanding
   - step3: Repeat if necessary until you understand the subject fully

2. ANALYSIS / SYNTHESIS REQUIREMENTS:
   - Cross-reference findings
   - Highlight consensus versus controversy
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

        with st.spinner("Processing Deep Search..."):
            result = agent.solve_task(task_prompt, streaming=True, max_iterations=MAX_ITERATIONS)
        tabs[2].markdown("### Final Report")
        tabs[2].markdown(result)

# ==============================================================================
# Embedded Streamlit Launcher (while preserving uv run)
# ==============================================================================

if __name__ == "__main__":
    if os.environ.get("STREAMLIT_EMBEDDED") != "1":
        # Embed the streamlit command to generate proper ScriptRunContext with uv run.
        os.environ["STREAMLIT_EMBEDDED"] = "1"
        sys.argv = ["streamlit", "run", sys.argv[0]]
        from streamlit.web import cli as stcli
        sys.exit(stcli.main())
    else:
        main()