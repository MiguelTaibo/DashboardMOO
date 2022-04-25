import requests
import os

from frontutils import filterdict

backend = os.getenv('backend', "http://localhost:8000/api/")

def getAcqfunctions(state):
    response = requests.get(backend+'acqfunctions', verify=False)

    if response.status_code!=200:
        state.input_error = True
        state.input_error_text = response.json()['detail']
        return
    response = response.json()
    for key in response:
        state[key]=response[key]
    return

def startExperiment(state):
    data_dict = {
        "n_ins": state.n_ins, 
        "input_names": state.input_names, 
        "input_mms": state.input_mms,
        "n_objs": state.n_objs,
        "objective_names": state.objective_names,
        "objective_mms": [o=="Maximize" for o in state.objective_mms],
        "n_cons": state.n_cons,
        "kernel": state.kernel,
        "acq_id": filterdict(state.acqfunctions,"name",state.acqfunct)[0]["id"],
        "acqfunct_hps": state.acqfunct_hps
    }
    if state.name!="":
        data_dict['name']=state.name
    elif not state.input_error:
        state.input_error = True
        state.input_error_text = "If you do not set a name for your experiment you will be unable to recap it, if you resubmit the experiment it will work without setting a name."
        return
    state.input_error = False
    
    response = requests.post(backend+'startexp', json=data_dict)

    if response.status_code!=200:
        state.input_error = True
        state.input_error_text = response.json()['detail']
        return
    state.selected = True
    response = response.json()
    for key in response:
        state[key]=response[key]
    return

def loadExperiment(state):
    response = requests.get(backend+"loadexp/"+state.name)
    state.input_error = False

    if response.status_code!=200:
        state.input_error = True
        state.input_error_text = response.json()['detail']
        return
    state.selected = True
    response = response.json()
    for key in response:
        state[key]=response[key]

    state.acqfunct =  filterdict(state.acqfunctions,"id",state.acq_id)[0]["name"]

    return

def drawNextSample(state):

    response = requests.get(backend+"getsample/next/"+str(state.id))
    
    if response.status_code!=200:
        return
    response = response.json()
    for key in response:
        state[key]=response[key]
    return state

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
    state.next_x = [0. for _ in range(state.n_ins)]
    state.next_y = [None for _ in range(state.n_objs)]
    response = response.json()
    for key in response:
        state[key]=response[key]

    return