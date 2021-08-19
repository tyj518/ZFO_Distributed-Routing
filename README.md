# ZFO_Distributed-Routing
This repository contains code for the distributed routing test case in the paper "Zeroth-Order Feedback Optimization for Cooperative Multi-Agent Systems". The code is in Matlab.

## File List
- <code>main.m</code>: Script for the simulation without knowledge of function dependence.
- <code>main_f_dependence.m</code>: Script for the simulation with knowledge of function dependence.
- <code>eval_obj_distributed_routing.m</code>: Function that returns a vector of local objective values.
- <code>proj_simplex_delta.m</code>: Function for projection onto the shrunk simplex.
- <code>sample_z.m</code>: Function for generating the perturbation in the zeroth-order gradient estimator.
- <code>global_func_opt.m</code>: Function that returns the global objective value and gradient for <code>fmincon</code>.
- <code>data.mat</code>: Data of the distributed routing test case.
- <code>dist_routing_sigma_X.XX.mat</code>: Simulation results for tests without knowledge of function dependence. <code>X.XX</code> represents the noise level.
- <code>dist_routing_sigma_X.XX_depn_known.mat</code>: Simulation results for tests with knowledge of function dependence. <code>X.XX</code> represents the noise level.
