"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gpflow
import pandas as pd
import sobol_seq

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


from gpflow.utilities import print_summary
from frontutils import get_pareto_undominated_by


class frontGP(object):
    def __init__(self, O:int, C:int, d:int, lowerBounds: float, upperBounds: float, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.multiGPR : MultiGPR = None

    def addSample(self, x, y, save=False, filename=None):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)
        if save and filename is not None:
            self.writeSample(filename, x,y)

    def updateGP(self):
        self.multiGPR = MultiGPR(X = self.X, Y = self.Y, noise_variance = self.noise_variance)

    def optimizeKernel(self):
        self.multiGPR.optimizeKernel()

    ## Visualization methods
    def plot(self):

        fig, axs = plt.subplots(nrows = self.O, ncols=self.d, figsize=(10,5))

        if self.d >1:
            for j in range(self.d):
                grid = np.ones((10_000,self.d))
                for k in range(self.d):
                    grid[:,k]=grid[:,k]*(self.upperBounds[k]+self.lowerBounds[k])/2
                xx = np.linspace(self.lowerBounds[j], self.upperBounds[j], 10_000).reshape(10_000, 1)
                grid[:,j]=xx[:,0]
                mean, var = self.multiGPR.predict_y(grid)

                if self.O==1:
                    axs[j].plot(self.X[:,j], self.Y[:,0], 'kx', mew=2)
                    axs[j].plot(grid[:,j], mean[:,0], 'C0', lw=2)
                    axs[j].fill_between(grid[:,j],
                                    mean[:,0] - 2*np.sqrt(var[:,0]),
                                    mean[:,0] + 2*np.sqrt(var[:,0]),
                                    color='C0', alpha=0.2)
                else:
                    for i in range(self.O):
                        axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                        axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                        axs[i, j].fill_between(grid[:,j],
                                        mean[:,i] - 2*np.sqrt(var[:,i]),
                                        mean[:,i] + 2*np.sqrt(var[:,i]),
                                        color='C0', alpha=0.2)

        else:
            xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 10_000).reshape(10_000, 1)
            mean, var = self.multiGPR.predict_y(xx)
            if self.O==1:
                axs.plot(self.X, self.Y[:,0], 'kx', mew=2)
                axs.plot(xx[:,0], mean[:,0], 'C0', lw=2)
                axs.fill_between(xx[:,0],
                                mean[:,0] - 2*np.sqrt(var[:,0]),
                                mean[:,0] + 2*np.sqrt(var[:,0]),
                                color='C0', alpha=0.2)
            else:
                for i in range(self.O):
                    axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                    axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                    axs[i].fill_between(xx[:,0],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)

        return fig, axs

    def plotParetos(self, state):
        problem = GPProblem(self)
        res = minimize(problem,
                NSGA2(),
                save_history=True,
                verbose=False)
        
        pareto_front = res.F
        pareto_set = getSetfromFront(res.X, res.F, pareto_front)

        for idx, mm in enumerate(state.objective_mms):
            if mm:
                pareto_front[:,idx]=-pareto_front[:,idx]

        best_known_pareto_front = get_pareto_undominated_by(self.Y)
        best_known_pareto_set = getSetfromFront(self.X, self.Y, best_known_pareto_front)

        fig1, axs1 = plt.subplots(figsize=(8,8))
        if self.d>1:
            axs1.plot(pareto_set[:,state.input_names.index(state.setx)],pareto_set[:,state.input_names.index(state.sety)], 'bx', markersize=3, label=r"Estimated Pareto Set")
            axs1.plot(best_known_pareto_set[:,state.input_names.index(state.setx)], best_known_pareto_set[:,state.input_names.index(state.sety)], 'gx', markersize=10, label=r"Best Known Pareto Set")
            axs1.set_ylabel(state.sety, fontsize=14)
            axs1.set_xlabel(state.setx, fontsize=14)
        else:
            axs1.plot(pareto_set[:,0], [0 for _ in pareto_set[:,0]],'bx',  markersize=3, label=r"Estimated Pareto Set")
            axs1.plot(best_known_pareto_set[:,0], [0 for _ in best_known_pareto_set[:,0]],'gx', markersize=10, label=r"Best Known Pareto Set")
            axs1.set_xlabel(state.input_names[0], fontsize=14)
            axs1.set_yticks(ticks = [])
        axs1.legend(fontsize=14)

        fig2, axs2 = plt.subplots(figsize=(8,8))
        axs2.plot(pareto_front[:,state.objective_names.index(state.frontx)], pareto_front[:,state.objective_names.index(state.fronty)], 'xb', markersize=3, label=r"Estimated Pareto Front")
        axs2.plot(best_known_pareto_front[:,state.objective_names.index(state.frontx)],best_known_pareto_front[:,state.objective_names.index(state.fronty)], 'xg', markersize=10, label=r"Best Known Pareto Front")
        axs2.set_xlabel(state.frontx, fontsize=14)
        axs2.set_ylabel(state.fronty, fontsize=14)
        axs2.legend(fontsize=14)

        return fig1,fig2
    
    def plotMetrics(self, state):
        fig, axs = plt.subplots(figsize=(8,8))
        axs.plot(state.ns, state.agd, label=r"$AGD_1(\mathcal{Y}_E^*, \mathcal{Y}_{BK}^*)$")
        axs.plot(state.ns, state.adh, label=r"$d_{ADH}(\mathcal{Y}_E^*, \mathcal{Y}_{BK}^*)$")
        axs.set_ylabel("Log Metrics",fontsize=14)
        axs.set_xlabel("Algorithm Iteration",fontsize=14)
        axs.legend(fontsize=14)
        return fig

    def dlParetos(self, state):
        problem = GPProblem(self)
        res = minimize(problem,
                NSGA2(),
                save_history=True,
                verbose=False)
        
        pareto_front = res.F
        pareto_set = res.X

        df =pd.DataFrame(data = np.append(pareto_set,pareto_front,axis=1),columns=state.input_names+state.objective_names)
        return df.to_csv()

    def dlParetoBestKnown(self, state):
        pareto_front = get_pareto_undominated_by(self.Y)
        pareto_set = getSetfromFront(self.X, self.Y, pareto_front)
        df =pd.DataFrame(data = np.append(pareto_set,pareto_front,axis=1),columns=state.input_names+state.objective_names)
        return df.to_csv()

class MultiGPR(object):
    def __init__(self, X = None, Y = None, noise_variance=0.01):
        self.GPRs = [
            gpflow.models.GPR(
                [X, Y[:,i:i+1]],
                kernel = gpflow.kernels.SquaredExponential(), 
                mean_function = gpflow.mean_functions.Constant(),
                noise_variance = noise_variance
            )
            for i in range(Y.shape[-1]) 
        ]
        self.opt = gpflow.optimizers.Scipy()

    def optimizeKernel(self):
        for GPR in self.GPRs:
            self.opt.minimize(
                GPR.training_loss, 
                variables=GPR.trainable_variables)

    def predict_y(self, xx):

        mean_vars = tf.concat([GPR.predict_y(xx) for GPR in self.GPRs], axis=-1)
        mean = mean_vars[0]
        var = mean_vars[1]
        return mean, var

    def predict_f_samples(self, xx, n_samples):
        presamples = [GPR.predict_f_samples(xx, n_samples) for GPR in self.GPRs]
        samples = tf.concat(presamples[:], axis=-1)
        return samples        

    def printGPRs(self):
        for GPR in self.GPRs:
            print_summary(GPR)



def getSetfromFront(xvalues, yvalues, front):
    res = None
    for y in front:
        x = xvalues[np.where(np.all(yvalues==y,axis=1))[0]]
        if res is None:
            res = np.array(x)
        else:
            res = np.append(res,x, axis=0)

    return res

class GPProblem(Problem):
    def __init__(self, GP):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array(GP.lowerBounds), xu=np.array(GP.upperBounds))
        self.multiGPR = GP.multiGPR

    def _evaluate(self, x, out, *args, **kwargs):
        mean, _ = self.multiGPR.predict_y(np.array([[x]]))
        out["F"] = np.column_stack(mean[0])