from typing import Any, List, Optional
from pydantic import BaseModel


class AcqsOut(BaseModel):
    acqfunctions : Optional[List[Any]]

class InputExperiment(BaseModel):
    name: Optional[str]
    n_ins: int
    input_names: List[str]
    input_mms: Optional[List[List[float]]]
    n_objs: int
    objective_names: List[str]
    objective_mms: Optional[List[bool]]
    n_cons: int
    kernel: str
    acq_id: int
    acqfunct_hps : Optional[Any]

class OutputExperiment(BaseModel):
    id: Optional[int]
    name: Optional[str]
    n_ins: int
    input_names: List[str]
    input_mms: Optional[List[List[float]]]
    n_objs: int
    objective_names: List[str]
    objective_mms: Optional[List[bool]]
    n_cons: int
    kernel: Optional[str]
    acq_id: Optional[int]
    acqfunct_hps : Optional[Any]
    X: Optional[List]
    Y: Optional[List]
    next_x: Optional[List]

class Sample(BaseModel):
    x: Optional[List]
    y: Optional[List]

class XSample(BaseModel):
    next_x: Optional[List]


class OutputSamples(BaseModel):
    X: Optional[List]
    Y: Optional[List]
