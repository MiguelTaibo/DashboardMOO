from sqlalchemy.orm import Session

from fastapi import HTTPException
import numpy as np
import os

from schemas import InputExperiment, OutputExperiment, OutputSamples, Sample
from models import *

def startTest(experiment: InputExperiment, db: Session):

    if experiment.n_ins!=len(experiment.input_names):
        raise HTTPException(status_code=404, detail="Input names has different length that number of inputs")
    if experiment.n_objs!=len(experiment.objective_names):
        raise HTTPException(status_code=404, detail="Objectives names has different length that number of objectives")

    kernel = db.query(Kernel).filter(Kernel.name==experiment.kernel).first()
    acq = db.query(AcqFunction).filter(AcqFunction.name ==experiment.acq_funct).first()

    new_exp = Test(n_ins = experiment.n_ins,
        n_objs = experiment.n_objs,
        n_cons = experiment.n_cons,
        kernel_id = kernel.id,
        acq_id = acq.id)
    if hasattr(experiment,'name'):
        new_exp.name = experiment.name

    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)

    for name in experiment.input_names:
        input_db = Input(name=name, test_id=new_exp.id)
        db.add(input_db)
    for name in experiment.objective_names:
        output_db = Output(name=name, test_id=new_exp.id)
        db.add(output_db)

    db.commit()

    return OutputExperiment(
        name = name,
        n_ins = new_exp.n_ins,
        input_names = experiment.input_names,
        n_objs = new_exp.n_objs,
        objective_names = experiment.objective_names,
        n_cons = new_exp.n_cons,
        Kernel = new_exp.kernel_id,
        acq_funct = new_exp.acq_id
    )

def loadTest(name:str, db: Session):

    test = db.query(Test).filter(Test.name==name).first()
    if test is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    inputs_db = db.query(Input).filter(Input.test_id==test.id).all()
    outputs_db = db.query(Output).filter(Output.test_id==test.id).all()
    
    try:
        X = np.load("experiments/"+str(test.id)+"X.npy")
    except:
        X = None

    try:
        Y = np.load("experiments/"+str(test.id)+"Y.npy")
    except:
        Y = None

    try:
        next_x = np.load("experiments/"+str(test.id)+"Xnext.npy")
    except:
        next_x = None

    return OutputExperiment(
        id = test.id,
        name = name,
        n_ins = test.n_ins,
        input_names = [el.name for el in inputs_db],
        n_objs = test.n_objs,
        objective_names = [el.name for el in outputs_db],
        n_cons = test.n_cons,
        X = None if X is None else X.tolist(),
        Y = None if Y is None else Y.tolist(),
        next_x = None if next_x is None else next_x.tolist()
    )

def getRandomSample(testid:int,  db: Session):
    test = db.query(Test).filter(Test.id==testid).first()
    
    test_file = "experiments/"+str(testid)+"Xnext.npy"
    x = np.random.uniform(0,1,test.n_ins)
    np.save(test_file, x)

    return {"next_x": x.tolist()}

def getNextSample(testid:int,  db: Session):
    test = db.query(Test).filter(Test.id==testid).first()
    
    test_file = "experiments/"+str(testid)+"Xnext.npy"
    x = np.random.uniform(0,1,test.n_ins)
    np.save(test_file, x)

    return {"next_x": x.tolist()}


def setSample(testid:int,sample: Sample, db: Session):
    
    try:
        X = np.load("experiments/"+str(testid)+"X.npy")
        X = np.append(X, np.array([sample.x]), axis=0)
    except:
        X = np.array(np.array([sample.x]))
    np.save("experiments/"+str(testid)+"X.npy", X)

    try:
        Y = np.load("experiments/"+str(testid)+"Y.npy")
        Y = np.append(Y, np.array([sample.y]), axis=0)
    except:
        Y = np.array(np.array([sample.y]))
    np.save("experiments/"+str(testid)+"Y.npy", Y)

    os.remove("experiments/"+str(testid)+"Xnext.npy")

    return OutputSamples(X=X.tolist(), Y=Y.tolist())
    