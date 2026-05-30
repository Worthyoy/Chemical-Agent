"""Compatibility entry point for the LangGraph workflow."""

from pathlib import Path
import sys

EXTRACT_DIR = Path(__file__).resolve().parent
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from langgraph_workflow.pipeline import main


if __name__ == "__main__":
    main()
