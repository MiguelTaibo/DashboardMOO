import pandas as pd
from pydantic.types import Json
import requests

backend = "http://localhost:8000/"

def startExperiment(state):
    data_dict = {
        "n_ins": state.n_ins, 
        "input_names": state.input_names, 
        "n_objs": state.n_objs,
        "objective_names": state.objective_names,
        "n_cons": state.n_cons,
        "kernel": state.kernel,
        "acq_funct": state.acq_funct
    }
    if state.name!="":
        data_dict['name']=state.name
    response = requests.post(backend+'startexp', json=data_dict)

    if response.status_code!=200:
        return
    state.selected = True
    response = response.json()
    for key in response:
        state[key]=response[key]
    return

def loadExperiment(state):
    state.selected = True
    response = requests.get(backend+"loadexp/"+state.name)

    if response.status_code!=200:
        return
    response = response.json()
    for key in response:
        state[key]=response[key]

    return

def drawRandomSample(state):
    response = requests.get(backend+"getsample/random/"+str(state.id))
    
    if response.status_code!=200:
        return
    response = response.json()
    for key in response:
        state[key]=response[key]
    return state

def sendSample(state):
    response = requests.post(backend+"setsample/"+str(state.id), json={"x": state.next_x, "y": state.next_y})
    if response.status_code!=200:
        return
    state.next_x = [None for _ in range(state.n_ins)]
    state.next_y = [None for _ in range(state.n_objs)]
    response = response.json()
    for key in response:
        state[key]=response[key]

    return