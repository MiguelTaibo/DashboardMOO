import sys, os
cur_path = os.getcwd()
sys.path.append(cur_path+"/MOOEasyTool")

from fastapi import FastAPI
from sqlalchemy.orm import Session
from fastapi import Depends

from db_session import get_db
from dbModels import Test
from schemas import AcqsOut, InputExperiment, OutputExperiment, OutputSamples, Sample, XSample
from cruds import getAcqfunctions, getNextSample, setSample, startTest, loadTest, getRandomSample

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/acqfunctions", response_model=AcqsOut)
async def get_acqfunctions(db: Session = Depends(get_db)):
    """
    Return all acq functions with their id, name, help, and hyperparameters (name, default, and help)
    """
    return getAcqfunctions(db)

@app.post("/startexp", response_model=OutputExperiment)
async def start_test(experiment: InputExperiment, db: Session = Depends(get_db)):
    """
    Acaba de definir un test y lo guarda en la bbdd
    """
    return startTest(experiment, db)

@app.get("/loadexp/{name}", response_model=OutputExperiment)
async def load_test(name: str, db: Session = Depends(get_db)):
    """
    Busca en la bd el test segun el nombre y lo carga
    """
    return loadTest(name, db)

@app.get("/getsample/random/{testid}", response_model=XSample)
async def get_random_sample(testid:int,  db: Session = Depends(get_db)):
    """
    Get a random value for the next input variables to try
    """
    return getRandomSample(testid, db)

@app.get("/getsample/next/{testid}", response_model=XSample)
async def get_next_sample(testid:int,  db: Session = Depends(get_db)):
    """
    Get next value for the next input variables to try using 
    experiments hyperparameters
    """
    return getNextSample(testid, db)

@app.post("/setsample/{testid}", response_model=OutputSamples)
async def set_sample(testid:int,sample: Sample, db: Session = Depends(get_db)):
    """
    Set a new sample in the experiment
    """
    return setSample(testid, sample, db)
