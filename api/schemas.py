"""Request bodies. Price requests reuse Phase 3's PriceRequest unchanged."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from models.price.predictor import PriceRequest


class ForecastRequest(PriceRequest):
    # Restated from PriceRequest so the rule is visible here: inf/NaN are 422s, not 500s.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    property_id: str | None = Field(default=None, max_length=64)


class ListingCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=5000)
    asking_price_aed: float = Field(gt=0)
    area_id: int
    building_name: str | None = Field(default=None, max_length=200)
    project_name: str | None = Field(default=None, max_length=200)
    property_kind: Literal["apartment", "hotel_apartment", "townhouse", "villa"]
    status: Literal["ready", "off_plan"]
    size_sqm: float = Field(gt=0)
    size_basis: Literal["built_up", "plot"] = "built_up"
    bedrooms: int | None = Field(default=None, ge=0, le=8)
    agent_id: int | None = None
    posted_at: date | None = None
    photo_ids: list[int] = Field(default_factory=list, max_length=10)
