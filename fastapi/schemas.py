from typing import List, Optional
from pydantic import BaseModel

import numpy

class InputExperiment(BaseModel):
    name: Optional[str]
    n_ins: int
    input_names: List[str]
    n_objs: int
    objective_names: List[str]
    n_cons: int
    kernel: str
    acq_funct: str

class OutputExperiment(BaseModel):
    id: Optional[int]
    name: Optional[str]
    n_ins: int
    input_names: List[str]
    n_objs: int
    objective_names: List[str]
    n_cons: int
    kernel: Optional[str]
    acq_funct: Optional[str]
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
