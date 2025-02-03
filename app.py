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
import time
import streamlit as st

# ----------------------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Quantalogic Deep Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Custom CSS for Styling
# ----------------------------------------------------------------------
st.markdown("""
    <style>
    /* General application styling */
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
    
    /* Collapsible tree for event log */
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
    
    /* Flex container for search input and button */
    .search-container {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 20px;
    }
    .search-container .sbox { flex: 3; }
    .search-container .btn { flex: 1; }
    </style>
    """, unsafe_allow_html=True)

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
# Global Constants and Environment Check
# ==============================================================================
MAX_ITERATIONS = 10
MODEL_NAME = "openrouter/openai/gpt-4o-mini"
OUTPUT_DIRECTORY = "./results"

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("The environment variable OPENROUTER_API_KEY is not set!")
    st.stop()

# ==============================================================================
# Session State Initialization for Logs
# ==============================================================================
if "token_log" not in st.session_state:
    st.session_state.token_log = ""
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# Global placeholders for displaying logs in UI
token_placeholder = None
event_placeholder = None

# ==============================================================================
# Utility: Generate a Unique Report Filename
# ==============================================================================
def get_next_report_filename() -> str:
    """
    Returns a unique filename in the format report_001.md inside OUTPUT_DIRECTORY.
    """
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    files = [f for f in os.listdir(OUTPUT_DIRECTORY) if f.startswith("report_") and f.endswith(".md")]
    max_num = 0
    for f in files:
        try:
            num = int(f[len("report_") : -3])
            max_num = max(max_num, num)
        except Exception:
            continue
    new_num = max_num + 1
    return f"report_{new_num:03d}.md"

# ==============================================================================
# Utility: Poll for Final Report File Existence
# ==============================================================================
def wait_for_final_report(report_path: str, wait_seconds: int = 5) -> bool:
    """
    Polls for the existence of the final report file for up to wait_seconds seconds.
    """
    start_time = time.time()
    while time.time() - start_time < wait_seconds:
        if os.path.exists(report_path):
            return True
        time.sleep(0.5)
    return False

# ==============================================================================
# Utility: Convert Data to an HTML Collapsible Tree View
# ==============================================================================
def get_tree_html(data, indent: int = 0) -> str:
    """
    Recursively converts data (dictionary, list, or basic types) into an HTML collapsible tree.
    """
    html = ""
    spacing = indent * 20  # pixels
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
        html += f"<div style='margin-left:{spacing}px'>{data}</div>"
    return html

# ==============================================================================
# Callback: Print Streaming Tokens in Live Output Panel
# ==============================================================================
def streamlit_print_token(event: str, data: any = None):
    """
    Update the live output panel with streaming tokens.
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
                overflow-y: auto;">
            {st.session_state.token_log}
            </div>
            """, unsafe_allow_html=True
        )

# ==============================================================================
# Callback: Log and Display Events in a Collapsible Tree View
# ==============================================================================
def streamlit_print_events(event: str, data: any = None):
    """
    Display event details in the event log pane using collapsible HTML.
    """
    html = f"<div style='border: 1px solid #ccc; margin-bottom: 10px; padding:10px;'>"
    html += f"<div><strong>Event:</strong> {event}</div>"
    if data:
        tree_html = get_tree_html(data, indent=1)
        html += tree_html
    html += "</div>"
    st.session_state.event_log.append(html)
    combined = "<h3>Event Log</h3>" + "<br>".join(st.session_state.event_log)
    event_placeholder.markdown(combined, unsafe_allow_html=True)

# ==============================================================================
# Callback: Ask for User Validation (auto-confirmation in this demo)
# ==============================================================================
def ask_for_user_validation(question: str) -> bool:
    st.info(question)
    return True

# ==============================================================================
# Agent and Tools Setup
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

# Register agent event listeners for detailed logging.
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

    # Sidebar Instructions
    st.sidebar.title("Instructions")
    st.sidebar.info(
        """
Enter a subject for deep multi-source research analysis.
The agent will stream output and detailed event logs in real time.
After the task completes, a final report — including an Executive Summary and a Final Report Summary — 
will be generated and displayed in the Final Report tab.
        """
    )

    # Header and Description on the Main Page
    st.title("🔍 Quantalogic Deep Search")
    st.subheader("Comprehensive Multi-Source Research Analysis")
    st.markdown(
        """
Enter a subject below and click the "Start Search" button.
You will see live output and event logs while the task is processing.
Once complete, you will be alerted and the final report will appear in the Final Report tab.
        """
    )

    # Search Container using a Flex Layout
    with st.container():
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        subject_to_search = st.text_input(
            "",
            key="search_input",
            placeholder="e.g., Renewable Energy Trends",
            label_visibility="collapsed"
        )
        start_search = st.button("Start Search")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for Live Output, Event Log, and Final Report
    tabs = st.tabs(["Live Output", "Event Log", "Final Report"])
    token_placeholder = tabs[0].empty()
    with tabs[1]:
        st.markdown("<h3>Event Log</h3>", unsafe_allow_html=True)
    event_placeholder = tabs[1].empty()

    # Execute deep search task if search initiated and subject is provided.
    if start_search and subject_to_search.strip():
        # Generate a unique filename and define path for the final report.
        final_report_filename = get_next_report_filename()
        final_report_path = os.path.join(OUTPUT_DIRECTORY, final_report_filename)

        # Clear previous logs from session state and UI.
        st.session_state.token_log = ""
        st.session_state.event_log = []
        token_placeholder.empty()
        event_placeholder.empty()

        # Build the deep search task prompt with clear instructions for generating
        # an Executive Summary and a Final Report Summary.
        task_prompt = f"""
MISSION: Execute comprehensive multi-source research analysis on the subject: {subject_to_search}

YOU MUST COMPLETE THE SEARCH IN LESS THAN {MAX_ITERATIONS} ITERATIONS.

1. SEARCH:
   - Utilize search tools to locate relevant information on the given subject.
   - Analyze the most promising results for deeper insights.
   - Iterate as needed until a comprehensive understanding is achieved.

2. ANALYSIS / SYNTHESIS:
   - Cross-reference the findings.
   - Identify consensus versus controversy.
   - Quantify the confidence level for major claims.
   - Highlight critical knowledge gaps and emerging trends.
   - Present geographical and cultural perspectives.

3. FINAL REPORT GENERATION:
   Write a final report in {final_report_path} containing at least 2000 words and including the following sections:
   
   ## Executive Summary
   - Summarize key findings, implications, and confidence assessments.
   
   ## Methodology
   - Describe the search strategy, source selection criteria, analytical framework, and limitations.
   
   ## Findings
   - Present major themes, supporting evidence, contrasting views, statistical and trend analyses.
   
   ## Source Analysis
   - Assess source credibility, biases, and methodological reviews.
   
   ## Recommendations
   - Identify research gaps, propose follow-up studies, and discuss practical applications.
   
   ## Citations
   - Provide a complete bibliography, citation metrics, and source credibility scores.
   
   ## Final Report Summary
   - Conclude with a succinct summary of the entire report, encapsulating overall findings and insights.
   
All content must be formatted as GitHub-flavored markdown with proper headings, code blocks, tables, and emphasis.
        """

        # Execute the deep search task with streaming output.
        with st.spinner("Processing Deep Search..."):
            _ = agent.solve_task(
                task_prompt, streaming=True, max_iterations=MAX_ITERATIONS
            )

        # Alert the user that the task is complete.
        st.success("Task complete! The final report has been generated.")
        st.balloons()

        # Poll for the final report file for up to 5 seconds.
        if wait_for_final_report(final_report_path, wait_seconds=5):
            try:
                with open(final_report_path, "r", encoding="utf-8") as file:
                    report_content = file.read()
                tabs[2].markdown(report_content)
            except Exception as e:
                st.error(f"Error reading the final report file: {e}")
        else:
            tabs[2].markdown("Final report file was not found after waiting. Please check the Event Log for more details.")

# ==============================================================================
# Streamlit Application Launcher
# ==============================================================================
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_EMBEDDED") != "1":
        os.environ["STREAMLIT_EMBEDDED"] = "1"
        sys.argv = ["streamlit", "run", sys.argv[0]]
        from streamlit.web import cli as stcli
        sys.exit(stcli.main())
    else:
        main()