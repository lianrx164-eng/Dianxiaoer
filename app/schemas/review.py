from pydantic import BaseModel, Field


class ApproveRequest(BaseModel):
    approved_answer: str = Field(min_length=1)
