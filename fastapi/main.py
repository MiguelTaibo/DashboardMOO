
from fastapi import File
from fastapi import FastAPI
from sqlalchemy.orm import Session
from fastapi import Depends

from db_session import get_db
from models import Test
from schemas import InputExperiment, OutputExperiment, OutputSamples, Sample, XSample
from cruds import setSample, startTest, loadTest, getRandomSample

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

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

@app.post("/setsample/{testid}", response_model=OutputSamples)
async def set_sample(testid:int,sample: Sample, db: Session = Depends(get_db)):
    """
    Set a new sample in the experiment
    """
    return setSample(testid, sample, db)
