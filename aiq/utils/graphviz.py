"""
Utility functions for generating graph visualizations using graphviz.
"""

import logging
import graphviz
from typing import Any

logger = logging.getLogger(__name__)


def save_workflow_visualization(
    app: Any, output_path: str = "mediation_workflow"
) -> None:
    """Generate and save a visualization of the workflow graph.

    Args:
        app: The compiled workflow application
        output_path: The path where the visualization should be saved (without extension)
    """
    try:
        dot = graphviz.Digraph(comment="Case Generation Workflow")
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="#4A90E2",
            fontcolor="white",
            fontname="Arial",
        )
        dot.attr("edge", color="#666666", penwidth="1.5")

        # Define node colors for different types of nodes
        node_colors = {
            "initial": "#4A90E2",  # Blue
            "basic_case_information_extraction": "#50C878",  # Emerald Green
            "document_extraction": "#FFA500",  # Orange
            "document_generation": "#9370DB",  # Medium Purple
            "END": "#FF6B6B",  # Coral Red
        }

        # Add nodes with custom colors
        for node in app.get_graph().nodes:
            color = node_colors.get(
                node, "#4A90E2"
            )  # Default to blue if node type not specified
            dot.node(node, node, fillcolor=color)

        # Add edges
        for edge in app.get_graph().edges:
            dot.edge(edge[0], edge[1])

        # Save the graph
        dot.render(output_path, format="png", cleanup=True)
        logger.info(f"Saved workflow visualization to {output_path}.png")
    except ImportError:
        logger.warning(
            "graphviz not installed. Skipping workflow visualization. Install with: pip install graphviz"
        )
    except Exception as e:
        logger.error(f"Failed to save workflow visualization: {str(e)}")
