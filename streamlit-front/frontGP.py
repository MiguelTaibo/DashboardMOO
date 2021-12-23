"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import io

import gpflow
import pandas as pd
import sobol_seq

from frontutils import get_pareto_undominated_by


class frontGP(object):
    def __init__(self, O:int, C:int, d:int, lowerBounds: float, upperBounds: float, kernel, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.opt = gpflow.optimizers.Scipy()
        self.GPR = None

    def addSample(self, x, y, save=False, filename=None):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)
        if save and filename is not None:
            self.writeSample(filename, x,y)

    def updateGPR(self):
        self.GPR = gpflow.models.GPR(
            [self.X, self.Y],
            kernel = self.kernel, 
            noise_variance = self.noise_variance)

    def optimizeKernel(self):
        self.opt.minimize(
            self.GPR.training_loss, 
            variables=self.GPR.trainable_variables)

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
                mean, var = self.GPR.predict_y(grid)

                (a_min,b_min) = (1e10, -1e10)
                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)

                    (a,b) = axs[i, j].get_ylim()
                    a_min = min(a,a_min)
                    b_min = max(b,b_min)

                for i in range(self.O):
                    axs[i, j].set_ylim(a_min, b_min)
        else:
            xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 10_000).reshape(10_000, 1)
            mean, var = self.GPR.predict_y(xx)
            for i in range(self.O):
                axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                axs[i].fill_between(xx[:,0],
                                mean[:,i] - 2*np.sqrt(var[:,i]),
                                mean[:,i] + 2*np.sqrt(var[:,i]),
                                color='C0', alpha=0.2)

        return fig, axs

    def plotParetos(self, state):
        grid = sobol_seq.i4_sobol_generate(self.d,10_000)
        mean, _ = self.GPR.predict_y(grid)
        mean = mean.numpy()

        for idx, mm in enumerate(state.objective_mms):
            if mm:
                mean[:,idx]=-mean[:,idx]

        pareto_front = get_pareto_undominated_by(mean)
        pareto_set = getSetfromFront(grid, mean, pareto_front)

        for idx, mm in enumerate(state.objective_mms):
            if mm:
                pareto_front[:,idx]=-pareto_front[:,idx]


        fig1, axs1 = plt.subplots(figsize=(8,8))

        axs1.plot(pareto_set[:,state.input_names.index(state.setx)],pareto_set[:,state.input_names.index(state.sety)], 'or')
        axs1.set_ylabel(state.sety)
        axs1.set_xlabel(state.setx)


        fig2, axs2 = plt.subplots(figsize=(8,8))
        axs2.plot(pareto_front[:,state.objective_names.index(state.frontx)],pareto_front[:,state.objective_names.index(state.fronty)], 'ob')
        axs2.set_xlabel(state.frontx)
        axs2.set_ylabel(state.fronty)
        return fig1,fig2
    
    def dlParetos(self, state):
        grid = sobol_seq.i4_sobol_generate(self.d,1_000)
        mean, _ = self.GPR.predict_y(grid)
        pareto_front = get_pareto_undominated_by(mean.numpy())
        pareto_set = getSetfromFront(grid, mean, pareto_front)
        df =pd.DataFrame(data = np.append(pareto_set,pareto_front,axis=1),columns=state.input_names+state.objective_names)
        return df.to_csv()


def getSetfromFront(xvalues, yvalues, front):
    res = None
    for y in front:
        x = xvalues[np.where(np.all(yvalues==y,axis=1))[0]]
        if res is None:
            res = np.array(x)
        else:
            res = np.append(res,x, axis=0)

    return res