from typing import Literal

# Party types as string literals
PartyType = Literal["MEDIATOR", "REQUESTING_PARTY", "RESPONDING_PARTY", "CLERK_SYSTEM"]

# Mediation phase types as string literals
MediationPhaseType = Literal[
    "OPENING_STATEMENTS",
    "JOINT_DISCUSSION_INFO_GATHERING",
    "CAUCUSES",
    "NEGOTIATION_BARGAINING",
    "CONCLUSION_CLOSING_STATEMENTS",
    "ENDED",  # Terminal phase
]