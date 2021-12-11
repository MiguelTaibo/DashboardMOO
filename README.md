# DashboardMOO

## Introduction

This tools serves as an interface with MOOEasyTool, which developes a multi optimization process of black boxes. 

## Usage

1. A user can create an experiment in DashboardMOO, which fixes the hyperparameters of the black box function and the objectives the user watns to optimize. 
2. In each iteration:
    a. The user will recieve a new configuration of hyperparameters to evaluate his function. 
    b. Afterwards, the user, is in charge of evaluating the black box and inserting the results (objectives) in the interface.
    c. The user must send to the API the last sample and ask for a new hyperparameter configuration one for start a new iteration.

This tool specifically targets costly black boxes, which can require hours, days or weeks to be evaluated. Therefore, it might not be feasible to store the experiments data in the browser cache. Therefore, all the data is stored in a database. All experiments data can be recovered with its name, which is set by the user when it is created.

Currently, anybody with a experiment name can introduce new results, which can mess with other people experiments. Introduce an optinal password can be usefull.

## Data stored

Just experiments data is stored, which consist on its name, inputs (name), outputs (name), kernel used, acquisition function used and all samples that the used inseted in it.
