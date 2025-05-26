from pydantic import BaseModel
from typing import Literal


class Party(BaseModel):  # Using BaseModel to allow future extension if needed
    name: Literal[
        "MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY", "CLERK_SYSTEM"
    ]

    def __str__(self):
        return self.name

    def __hash__(self):  # Make it hashable for dict keys if needed
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Party):
            return self.name == other.name
        return False


class MediationPhase(BaseModel):
    name: Literal[
        "OPENING_STATEMENTS",
        "JOINT_DISCUSSION_INFO_GATHERING",
        "CAUCUSES",
        "NEGOTIATION_BARGAINING",
        "CONCLUSION_CLOSING_STATEMENTS",
        "ENDED",  # Terminal phase
    ]

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, MediationPhase):
            return self.name == other.name
        return False