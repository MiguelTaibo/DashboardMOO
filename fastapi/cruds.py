from sqlalchemy.orm import Session
from MOOEasyTool.models.GaussianProcess import GaussianProcess
from MOOEasyTool.acquisition_functions.MESMO import mesmo_acq
from MOOEasyTool.acquisition_functions.PESMO import pesmo_acq
from MOOEasyTool.acquisition_functions.MES import mes_acq, basic_mes_acq
import gpflow

from fastapi import HTTPException
import numpy as np
import os
from MOOEasyTool.models.GPProblem import GPProblem

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from schemas import InputExperiment, OutputExperiment, OutputSamples, Sample
from dbModels import *

def getAcqfunctions(db: Session):
    from MOOEasyTool.acquisition_functions.MES import mes_acq_hp, basic_mes_acq_hp
    from MOOEasyTool.acquisition_functions.MESMO import mesmo_acq_hp
    from MOOEasyTool.acquisition_functions.UseMO import usemo_acq_hp

    return {"acqfunctions": [mes_acq_hp, basic_mes_acq_hp, mesmo_acq_hp, usemo_acq_hp]}

def startTest(experiment: InputExperiment, db: Session):
    if hasattr(experiment,'name') and experiment.name is not None and db.query(Test).filter(Test.name==experiment.name).first() is not None:
        raise HTTPException(status_code=404, detail="Experiment with name '"+experiment.name+"' already exits.")

    if experiment.n_ins!=len(experiment.input_names):
        raise HTTPException(status_code=404, detail="Input names has different length that number of inputs.")
    if experiment.n_objs!=len(experiment.objective_names):
        raise HTTPException(status_code=404, detail="Objectives names has different length that number of objectives.")

    new_exp = Test(
        n_ins = experiment.n_ins,
        n_objs = experiment.n_objs,
        n_cons = experiment.n_cons,
        kernel_id = 1,
        acq_id = experiment.acq_id
    )
    
    if hasattr(experiment,'name'):
        new_exp.name = experiment.name

    ## ACQ hyperparameters
    for k,v in experiment.acqfunct_hps.items():
        if k=="M":
            new_exp.acq_M = v
        if k=="N":
            new_exp.acq_N = v


    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)

    for name, mm in zip(experiment.input_names, experiment.input_mms):
        input_db = Input(name=name, test_id=new_exp.id, lowerBound=mm[0], upperBound=mm[1])
        db.add(input_db)
    for name, mm in zip(experiment.objective_names, experiment.objective_mms):
        output_db = Output(name=name, test_id=new_exp.id, maximize=mm)
        db.add(output_db)

    db.commit()

    return OutputExperiment(
        id = new_exp.id,
        name = new_exp.name,
        n_ins = new_exp.n_ins,
        input_names = experiment.input_names,
        input_mms = experiment.input_mms,
        n_objs = new_exp.n_objs,
        objective_names = experiment.objective_names,
        objective_mms = experiment.objective_mms,
        n_cons = new_exp.n_cons,
        Kernel = new_exp.kernel_id,
        acq_id = new_exp.acq_id,
        acqfunct_hps = experiment.acqfunct_hps
    )

def loadTest(name:str, db: Session):

    test = db.query(Test).filter(Test.name==name).first()
    if test is None:
        raise HTTPException(status_code=404, detail="Experiment '"+name+"' not found")

    ## acq hyperparameters
    acqfunct_hps = dict()
    if test.acq_N is not None:
        acqfunct_hps['N'] = test.acq_N
    if test.acq_M is not None:
        acqfunct_hps['M'] = test.acq_M

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

    lowerBounds,upperBounds = [],[]
    for i in inputs_db:
        lowerBounds.append(i.lowerBound)
        upperBounds.append(i.upperBound)
    GP = GaussianProcess(test.n_objs, test.n_cons, test.n_ins, lowerBounds, upperBounds, X = X, Y = Y, noise_variance=2e-6)
    
    GP.updateGP()
    GP.optimizeKernel()

    problem = GPProblem(GP=GP)
    res = minimize(problem,
            NSGA2(),
            save_history=True,
            verbose=False)
    

    metrics = db.query(Evaluation.ns, Evaluation.adh, Evaluation.agd).filter(Evaluation.test_id==test.id).all()

    ns = [m[0] for m in metrics]
    adh = [m[1] for m in metrics]
    agd = [m[2] for m in metrics]

    return OutputExperiment(
        id = test.id,
        name = name,
        n_ins = test.n_ins,
        input_names = [el.name for el in inputs_db],
        input_mms = [[el.lowerBound, el.upperBound] for el in inputs_db],
        n_objs = test.n_objs,
        objective_names = [el.name for el in outputs_db],
        objective_mms = [el.maximize for el in outputs_db],

        n_cons = test.n_cons,
        acq_id = test.acq_id,
        acqfunct_hps = acqfunct_hps,
        X = None if X is None else X.tolist(),
        Y = None if Y is None else Y.tolist(),
        next_x = None if next_x is None else next_x.tolist(),

        ns = ns,
        agd = agd,
        adh = adh,

        pareto_front = res.F.tolist(), 
        pareto_set = res.X.tolist()
    )

def getRandomSample(testid:int,  db: Session):
    test = db.query(Test).filter(Test.id==testid).first()
    
    test_file = "experiments/"+str(testid)+"Xnext.npy"
    x = np.random.uniform(0,1,test.n_ins)
    np.save(test_file, x)

    return {"next_x": x.tolist()}

def getNextSample(testid:int,  db: Session):
    test = db.query(Test).filter(Test.id==testid).first()
    
    inputs = db.query(Input).filter(Input.test_id==test.id).all()
    lowerBounds,upperBounds = [],[]
    for i in inputs:
        lowerBounds.append(i.lowerBound)
        upperBounds.append(i.upperBound)
    try:
        X = np.load("experiments/"+str(testid)+"X.npy")
    except:
        X = np.array([])

    outputs = db.query(Output).filter(Output.test_id==test.id).all()
    try:
        Y = np.load("experiments/"+str(testid)+"Y.npy")
    except:
        Y = np.array(np.array([]))
    for idx,o in enumerate(outputs):
        if (o.maximize):
            Y[:,idx] = - Y[:,idx]

    GP = GaussianProcess(test.n_objs, test.n_cons, test.n_ins, lowerBounds, upperBounds, X = X, Y = Y, noise_variance=2e-6)
    GP.updateGP()
    GP.optimizeKernel()

    x_best, _ = id_to_acqfunct(test.acq_id)(GP, N=test.acq_N, M=test.acq_M)

    test_file = "experiments/"+str(testid)+"Xnext.npy"
    np.save(test_file, x_best)

    return {"next_x": x_best.tolist()}

def setSample(testid:int,sample: Sample, db: Session):
    
    test = db.query(Test).filter(Test.id==testid).first()
    
    inputs = db.query(Input).filter(Input.test_id==test.id).all()
    lowerBounds,upperBounds = [],[]
    for i in inputs:
        lowerBounds.append(i.lowerBound)
        upperBounds.append(i.upperBound)

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

    try:
        os.remove("experiments/"+str(testid)+"Xnext.npy")
    except:
        pass
    
    GP = GaussianProcess(test.n_objs, test.n_cons, test.n_ins, lowerBounds, upperBounds, X = X, Y = Y, noise_variance=2e-6)
    GP.updateGP()
    GP.optimizeKernel()

    problem = GPProblem(GP=GP)
    res = minimize(problem,
            NSGA2(),
            save_history=True,
            verbose=False)
    

    adh, agd = GP.evaluateNoRealPareto()

    new_evaluation = Evaluation(
        test_id = test.id,
        ns = GP.X.shape[0],
        agd = agd,
        adh = adh
    )

    db.add(new_evaluation)
    db.commit()
    db.refresh(new_evaluation)

    metrics = db.query(Evaluation.ns, Evaluation.adh, Evaluation.agd).filter(Evaluation.test_id==test.id).all()

    ns = [m[0] for m in metrics]
    adh = [m[1] for m in metrics]
    agd = [m[2] for m in metrics]
    
    return OutputSamples(
        X=X.tolist(), Y=Y.tolist(), 
        ns=ns, adh=adh, agd=agd, 
        pareto_front = res.F.tolist(), pareto_set = res.X.tolist()
    )

###### Auxiliar functions

def id_to_acqfunct(id):
    if (id==1):
        print("basic mes acq")
        return basic_mes_acq
    if (id==2):
        print("mes acq")
        return mes_acq
    if (id==3):
        print("mesmo acq")
        return mesmo_acq
    if (id==4):
        print("mesmo acq")
        return pesmo_acq
