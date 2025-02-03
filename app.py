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
# Set page config as the very first Streamlit command
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Search AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Inject custom CSS for improved UX and styling for collapsible tree view
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
    /* Container spacing */
    .container { margin-bottom: 20px; }
    /* Styling the collapsible tree view */
    details {
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 5px 10px;
      margin-bottom: 5px;
      background-color: #f9f9f9;
    }
    details summary {
      cursor: pointer;
      font-weight: bold;
      margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Import Agent and Tools
# ----------------------------------------------------------------------
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
# Global placeholders for live logs (set later inside main())
# ==============================================================================
token_placeholder = None
event_placeholder = None

# ==============================================================================
# Utility function: generate collapsible tree HTML recursively
# ==============================================================================
def get_tree_html(data, indent=0):
    """Return HTML for displaying data as a collapsible tree view using <details>."""
    html = ""
    spacing = indent * 20  # adjust spacing with indent (in pixels)
    if isinstance(data, dict):
        for key, value in data.items():
            html += f"<details style='margin-left:{spacing}px' open>"
            html += f"<summary>{key}</summary>"
            html += get_tree_html(value, indent + 1)
            html += "</details>"
    elif isinstance(data, list):
        for index, item in enumerate(data):
            html += f"<details style='margin-left:{spacing}px' open>"
            html += f"<summary>[{index}]</summary>"
            html += get_tree_html(item, indent + 1)
            html += "</details>"
    else:
        # If the data is a basic type, just render it
        html += f"<div style='margin-left:{spacing}px'>{data}</div>"
    return html

# ==============================================================================
# Custom Callback Functions for Streaming Output & Events
# ==============================================================================

def streamlit_print_token(event: str, data: any = None):
    """
    Callback to update the Live Output panel.
    Each token received is appended and re-rendered.
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
    Callback to update the Event Log panel.
    Renders the event data as a collapsible tree view.
    """
    html = f"<div style='border: 1px solid #ccc; margin-bottom: 10px; padding:10px;'>"
    html += f"<div><strong>Event:</strong> {event}</div>"
    if data:
        tree_html = get_tree_html(data, indent=1)
        html += tree_html
    html += "</div>"
    st.session_state.event_log.append(html)
    combined = "<br>".join(st.session_state.event_log)
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
    LLMTool(
        name="report_writer", 
        model_name=MODEL_NAME, 
        on_token=streamlit_print_token
    ),
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
# Main Application Layout
# ==============================================================================

def main():
    global token_placeholder, event_placeholder

    # Sidebar with instructions
    st.sidebar.title("Instructions")
    st.sidebar.info(
        """
Enter a subject to perform a deep multi-source research analysis.
Watch as the AI streams tokens, and view detailed event logs in a collapsible tree view.
        """
    )

    # Main header and description
    st.title("🔍 Deep Search Application")
    st.subheader("Comprehensive Multi-Source Research Analysis")
    st.markdown(
        """
Welcome! Use the input below to start a deep search. The app will display the live AI output, detailed event logs,
and the final report in separate tabs.
        """
    )

    # Container for search input
    with st.container():
        subject_to_search = st.text_input("Search Subject", placeholder="e.g., Renewable Energy Trends")
        start_search = st.button("Start Search")

    st.markdown("---")
    
    # Tabs organized for Live Output, Event Log, and Final Report
    tabs = st.tabs(["Live Output", "Event Log", "Final Report"])
    token_placeholder = tabs[0].empty()
    event_placeholder = tabs[1].empty()

    if start_search and subject_to_search:
        # Clear previous logs when a new search starts
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
        os.environ["STREAMLIT_EMBEDDED"] = "1"
        sys.argv = ["streamlit", "run", sys.argv[0]]
        from streamlit.web import cli as stcli
        sys.exit(stcli.main())
    else:
        main()