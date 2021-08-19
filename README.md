# ZFO_Distributed-Routing
This repository contains code for the distributed routing test case in the paper "Zeroth-Order Feedback Optimization for Cooperative Multi-Agent Systems". The code is in Matlab.

## File List
- main.m: Script for the simulation without knowledge of function dependence.
- main_f_dependence.m: Script for the simulation with knowledge of function dependence.
- eval_obj_distributed_routing.m: Function that returns a vector of local objective values.
- proj_simplex_delta.m: Function for projection onto the shrunk simplex.
- sample_z.m: Function for generating the perturbation in the zeroth-order gradient estimator.
- global_func_opt.m: Function that returns the global objective value and gradient for <code>fmincon</code>.
- data.mat: Data of the distributed routing test case.
- dist_routing_sigma_X.XX.mat: Simulation results for tests without knowledge of function dependence. <code>X.XX</code> represents the noise level.
- dist_routing_sigma_X.XX_depn_known.mat: Simulation results for tests with knowledge of function dependence. <code>X.XX</code> represents the noise level.
