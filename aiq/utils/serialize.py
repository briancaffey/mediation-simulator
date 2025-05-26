"""
Utility functions for serializing Pydantic models and other objects.
"""

from typing import Any, Dict, List, Union
from datetime import datetime
from langchain_core.messages import BaseMessage


def serialize_pydantic(obj: Any) -> Union[Dict, List, str, Any]:
    """Serialize a Pydantic model or nested structure of Pydantic models to a dictionary.

    Args:
        obj: The object to serialize. Can be a Pydantic model, dict, list, or primitive type.

    Returns:
        The serialized object, with Pydantic models converted to dictionaries.
    """
    if isinstance(obj, BaseMessage):
        # Handle LangChain messages
        serialized = {
            "type": obj.__class__.__name__,
            "content": str(obj.content),
        }
        # Only include additional_kwargs if it's not empty
        if obj.additional_kwargs:
            serialized["additional_kwargs"] = obj.additional_kwargs
        return serialized
    elif hasattr(obj, "model_dump"):
        # Handle special cases for MediationPhase and Party - return just the name string
        if hasattr(obj, "name"):  # For MediationPhase and Party
            return obj.name
        # Handle MediationEvent specially
        elif hasattr(obj, "event_id"):  # For MediationEvent
            return {
                "event_id": str(obj.event_id),  # Ensure UUID is converted to string
                "timestamp": (
                    obj.timestamp.isoformat()
                    if isinstance(obj.timestamp, datetime)
                    else str(obj.timestamp)
                ),
                "mediation_phase": (
                    obj.mediation_phase.name
                    if hasattr(obj.mediation_phase, "name")
                    else str(obj.mediation_phase)
                ),
                "speaker": (
                    obj.speaker.name
                    if hasattr(obj.speaker, "name")
                    else str(obj.speaker)
                ),
                "content": str(obj.content),
                "summary": str(obj.summary),
                "token_count": int(obj.token_count),
            }
        # For other Pydantic models, use model_dump
        data = obj.model_dump()
        return data
    elif isinstance(obj, dict):
        return {k: serialize_pydantic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_pydantic(item) for item in obj]
    return obj
