from pydantic import BaseModel
from typing import List, Optional, Literal


#For Map data extraction
class Borehole(BaseModel):
    Name: str
    Number: int
    Elevation_level: float
class Borehole_data(BaseModel):
    metadata: List[Borehole]


#For drilllog data extraction
class Metadata(BaseModel):
    PROJECT_NAME: str
    HOLE_NO: str
    Elevation_level: float
    LOCATION: str
    GROUND_WATER_LEVEL: Optional[float]
    DATE: str
    DRILLER: str
class Sample(BaseModel):
    Sample_number: str
    Depth: float
    Hits: str
    TCR: Optional[float] = None # TCR% is optional, as it may not be present in all samples, only present in core samples
    RQD: Optional[float] = None # RQD% is optional, as it may not be present in all samples,  only present in core samples
    # Method: Literal['자연시료', '관입시험기에 의한 시료', '코아시료', '흐트러진시료']
    Method: str
class Soil(BaseModel):
    range: str
    soil_name: str
    soil_color: str
    observation: str
class Extracted_data(BaseModel):
    metadata: Metadata
    sample_data: List[Sample]
    soil_data: List[Soil]