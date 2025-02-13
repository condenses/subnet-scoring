from pydantic import BaseModel, field_validator
from typing import List


class Message(BaseModel):
    role: str
    content: str


class ScoringRequest(BaseModel):
    original_messages: List[Message]
    compressed_messages: List[Message]

    @field_validator("original_messages", "compressed_messages")
    @classmethod
    def validate_messages(cls, messages: List[Message]) -> List[Message]:
        if not messages:
            raise ValueError("Message list cannot be empty")

        if len(messages) % 2 != 0:
            raise ValueError("Message list must have an even number of messages")

        expected_pattern = ["user", "assistant"] * (len(messages) // 2)
        actual_roles = [msg.role for msg in messages]

        if actual_roles != expected_pattern:
            raise ValueError(
                "Messages must alternate between 'user' and 'assistant' roles"
            )

        return messages


class ScoringResponse(BaseModel):
    score: float
