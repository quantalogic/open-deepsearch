from typing import Any, Dict, Optional
import streamlit as st
import uuid
from pathlib import Path
from datetime import datetime
import html

# Import quantalogic components (assumed installed)
from quantalogic import Agent
from quantalogic.tools import DuckDuckGoSearchTool, SerpApiSearchTool, LLMTool, ReadFileTool, WriteFileTool
from quantalogic.console_print_events import console_print_events  # legacy, not used here
from quantalogic.console_print_token import console_print_token  # legacy, not used here

MODEL_NAME = "openrouter/openai/gpt-4o-mini"

global agent

# =============================================================================
# EVENT BOX FORMATTING (Console‐Style with Unicode Borders)
# =============================================================================
def flatten_event_data(data: Any, indent: int = 0) -> list[str]:
    """
    Recursively flatten event data (dicts and lists) into a list of formatted lines.
    """
    lines = []
    spacer = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            key_str = f"◈ {key} → "
            if isinstance(value, (dict, list)):
                lines.append(spacer + key_str)
                # Recursively flatten nested data with extra indent
                lines.extend(flatten_event_data(value, indent=indent + 4))
            else:
                # Preserve multiline strings as separate lines
                val_str = str(value)
                for subline in val_str.splitlines():
                    lines.append(spacer + key_str + subline)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(spacer + "•")
                lines.extend(flatten_event_data(item, indent=indent + 4))
            else:
                for subline in str(item).splitlines():
                    lines.append(spacer + "• " + subline)
    else:
        for subline in str(data).splitlines():
            lines.append(spacer + subline)
    return lines

def format_event_box(event: str, data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a formatted string with Unicode box borders representing the event.
    
    The output mimics a rich console display like:
    
    ╔══ ... 🎯 task_think_start ... ═╗
    ║ ◈ iteration → 1              ║
    ║ ◈ total_tokens → 1286         ║
    ║ ...                          ║
    ╚══ ... Items: 6 ... ═╝
    """
    # Prepare header text
    header = f"🎯 {event}"
    
    # Prepare content lines from event data.
    if data:
        content_lines = flatten_event_data(data, indent=0)
    else:
        content_lines = ["ⓘ No event data"]

    # Determine width of box: consider header and every content line.
    all_lines = [header] + content_lines
    content_width = max(len(line) for line in all_lines)
    
    # Total width of the box, adding padding on either side (2 spaces)
    box_width = content_width + 4

    # Build the top border: center header in the border.
    header_text = f" {header} "
    border_total = box_width - 2 - len(header_text)
    left_border = "═" * (border_total // 2)
    right_border = "═" * (border_total - len(left_border))
    top_border = f"╔{left_border}{header_text}{right_border}╗"

    # Build content lines: each content line padded to content_width.
    body_lines = []
    for line in content_lines:
        padded_line = line.ljust(content_width)
        body_lines.append(f"║  {padded_line}  ║")

    # Build bottom border with item count.
    item_count = len(data) if data and isinstance(data, dict) else 0
    items_text = f" Items: {item_count} "
    border_total_bottom = box_width - 2 - len(items_text)
    left_border_bottom = "═" * (border_total_bottom // 2)
    right_border_bottom = "═" * (border_total_bottom - len(left_border_bottom))
    bottom_border = f"╚{left_border_bottom}{items_text}{right_border_bottom}╝"

    # Combine all parts.
    box_lines = [top_border] + body_lines + [bottom_border]
    return "\n".join(box_lines)

# =============================================================================
# EVENT LOG DISPLAYS
# =============================================================================
def update_event_logs():
    """
    Update the full event log in the main panel and the minimal log in the sidebar.
    """
    full_log = "\n\n".join(st.session_state.get("event_log", []))
    minimal_log = "\n".join(st.session_state.get("minimal_event_log", []))
    
    # Update the main panel display
    if "main_event_log_container" in st.session_state:
        st.session_state["main_event_log_container"].code(full_log, language="")
    # Update the sidebar minimal display
    if "sidebar_minimal_event_log_container" in st.session_state:
        st.session_state["sidebar_minimal_event_log_container"].code(minimal_log, language="")

# =============================================================================
# EVENT LOGGING (Integrates our new formatted and minimal event logs)
# =============================================================================
def log_event(message: str, event_type: str = "info", extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Append an event to both the full event log (for the main panel) and the minimal version (for the sidebar).
    """
    # Combine provided message with extra_data into a single dict
    event_data = {"message": message}
    if extra_data:
        event_data.update(extra_data)
    
    # Generate formatted event box text (full details)
    box_text = format_event_box(f"{message} ({event_type})", event_data)
    
    if "event_log" not in st.session_state:
        st.session_state["event_log"] = []
    st.session_state["event_log"].append(box_text)
    
    # Create a minimal one-line version with a timestamp.
    minimal_text = f"{datetime.now().strftime('%H:%M:%S')} - {message} ({event_type})"
    if "minimal_event_log" not in st.session_state:
        st.session_state["minimal_event_log"] = []
    st.session_state["minimal_event_log"].append(minimal_text)
    
    # Update both displays.
    update_event_logs()

# =============================================================================
# AGENT CALLBACKS (Event tracking and streaming)
# =============================================================================
def handle_stream_chunk(event: str, data: Optional[str] = None) -> None:
    """
    Callback to handle streaming token chunks.
    Appends tokens to a session buffer and displays them as code.
    """
    if event == "stream_chunk" and data:
        if "stream_response" not in st.session_state:
            st.session_state["stream_response"] = ""
            st.session_state["stream_container"] = st.empty()
        st.session_state["stream_response"] += data
        st.session_state["stream_container"].code(st.session_state["stream_response"], language="python")

def track_events(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Callback to track agent events and display them in the formatted event box.
    """
    if event == "task_think_start":
        st.info("🤔 Agent is thinking...", icon="⏳")
        log_event("task_think_start", "thinking", data)
    elif event == "tool_execution_start":
        tool_name = data.get("tool_name", "Unknown tool") if data else "Unknown tool"
        st.info(f"🔧 Tool execution started: {tool_name}", icon="⚙️")
        log_event("tool_execution_start", "info", data)
    elif event == "task_think_end":
        st.success("✅ Agent finished thinking")
        log_event("task_think_end", "success", data)
    elif event == "tool_execution_end":
        tool_name = data.get("tool_name", "Unknown tool") if data else "Unknown tool"
        st.success(f"🔧 Tool execution ended: {tool_name}")
        log_event("tool_execution_end", "success", data)
    elif event == "error_max_iterations_reached":
        st.error("❌ Maximum iterations reached. Exiting.")
        log_event("error_max_iterations_reached", "error", data)

# =============================================================================
# AGENT INITIALIZATION
# =============================================================================
def initialize_agent(results_dir: str, model_name: str = MODEL_NAME):
    """
    Initialize the QuantaLogic agent with the configured tools.
    Streaming tokens are processed via 'handle_stream_chunk' and events via 'track_events'.
    """
    agent = Agent(
        model_name=model_name,
        tools=[
            DuckDuckGoSearchTool(),
            SerpApiSearchTool(),
            ReadFileTool(),
            WriteFileTool(),
        ]
    )
    # Register event listeners.
    agent.event_emitter.on(
        event=[
            "task_complete",
            "task_think_start",
            "task_think_end",
            "tool_execution_start",
            "tool_execution_end",
            "error_max_iterations_reached",
            "memory_full",
            "memory_compacted",
            "memory_summary"
        ],
        listener=track_events
    )

    agent.event_emitter.on(
        event=[
            "task_complete",
            "task_think_start",
            "task_think_end",
            "tool_execution_start",
            "tool_execution_end",
            "error_max_iterations_reached",
            "memory_full",
            "memory_compacted",
            "memory_summary"
        ],
        listener=console_print_events
    )
    

    #agent.event_emitter.on(event=["stream_chunk"], listener=handle_stream_chunk)
    #agent.event_emitter.on(event=["stream_chunk"], listener=console_print_token)
    return agent

# =============================================================================
# DEEP SEARCH FUNCTIONALITY
# =============================================================================
def deep_search(query: str):
    """
    Execute deep search with iterative research.
    The agent events and streaming are tracked and shown.
    Results are saved to a directory.
    """
    task_id = str(uuid.uuid4())
    results_dir = Path(f"deepsearch-results/{task_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    st.info("🚀 Initiating deep search operation...")
    log_event("deep_search_start", "info", {"query": query})
    
    agent = initialize_agent(str(results_dir))
    # (Re-)attach event listeners for current task.
    agent.event_emitter.on(
        ["task_think_start", "tool_execution_start", "task_think_end", "tool_execution_end", "error_max_iterations_reached"],
        track_events,
    )
    agent.event_emitter.on(["stream_chunk"], handle_stream_chunk)

    research_task = f"""
Context: Results will be saved to directory {results_dir}

System Prompt: You are a Deep Search Agent. Perform comprehensive research on the given topic.
- Use multiple search tools to gather information
- Validate information from at least 3 sources
- Save raw search results to {results_dir}
- Generate intermediate analysis files
- Produce a final report with citations

Research Topic: {query}

Requirements:
1. Perform 3 iterations of web search using different query formulations
2. Save raw search results as search_[iteration].json in {results_dir}
3. Create summary_[iteration].md after each search in {results_dir}
4. Cross-validate facts across sources
5. Generate final report in {results_dir}/report.md with:
   - Executive summary
   - Key findings
   - Source citations
   - Recommendations for further research
"""
    with st.status("🔍 Conducting research...", expanded=True) as status:
        try:
            result = agent.solve_task(research_task)
            status.update(label="🎉 Research completed!", state="complete")
            log_event("research_completed", "success", {"results_dir": str(results_dir)})
            return result, results_dir
        except Exception as e:
            status.update(label="❌ Research failed", state="error")
            st.error(f"An error occurred during research: {e}")
            log_event("research_failed", "error", {"error": str(e)})
            return str(e), results_dir

# =============================================================================
# DISPLAY SEARCH RESULTS
# =============================================================================
def display_results(result: str, results_dir: Path):
    """
    Display deep search results and research artifacts.
    """
    st.success("🎉 Deep Search Completed!")
    log_event("display_results", "success", {"results_dir": str(results_dir)})
    
    with st.expander("📄 Final Report", expanded=True):
        st.markdown(result)
    
    st.subheader("📁 Research Artifacts")
    cols = st.columns(3)
    cols[0].metric("Total Files", len(list(results_dir.glob("*"))))
    cols[1].metric("Search Iterations", len(list(results_dir.glob("search_*.json"))))
    cols[2].metric("Analysis Files", len(list(results_dir.glob("summary_*.md"))))
    
    with st.expander("📂 View Research Files"):
        for file in results_dir.glob("*"):
            st.write(f"📄 {file.name}")
            with open(file, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=file.name,
                    mime="text/plain" if file.suffix == ".md" else "application/json",
                    key=f"dl_{file.name}",
                )

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.title("🔍 Deep Search Agent")
    st.markdown("Powered by QuantaLogic Research Framework")

    # Create containers for full and minimal event logs if not already present.
    if "main_event_log_container" not in st.session_state:
        st.session_state["main_event_log_container"] = st.empty()
    if "sidebar_minimal_event_log_container" not in st.session_state:
        st.session_state["sidebar_minimal_event_log_container"] = st.sidebar.empty()

    # Research query form.
    with st.form(key="search_form"):
        query = st.text_area("Research Query", height=150, placeholder="Enter your research topic or question...")
        submitted = st.form_submit_button("Start Research")
        
        if submitted and query.strip():
            # Clear any previous streaming output.
            if "stream_response" in st.session_state:
                del st.session_state["stream_response"]
            if "stream_container" in st.session_state:
                st.session_state["stream_container"].empty()
                del st.session_state["stream_container"]

            with st.spinner("Initializing research environment..."):
                result, results_dir = deep_search(query)
                display_results(result, results_dir)

    # Display full event log in an expander in the main panel.
    with st.expander("📝 Full Event Log", expanded=True):
        # This will automatically reflect the latest log from session_state.
        update_event_logs()

if __name__ == '__main__':
    main()