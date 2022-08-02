# ComputingLagMeans

This repo contains scripts that are used to produce some of the figures in the manuscript "Computing Lagrangian means" by Kafiabad and Vanneste (submitted to JFM).
These scripts solve the Shallow Water equations and compute the Lagrangian means on-the-fly using the PDEs peresented in the manuscript (instead of particle tracking). The initial condition for the geostrophic flow is contained in 'uvr_2Dturbulence_256.mat' in terms of components of velocity.
The MATLAB script is used to produce the figures in this paper, but the Python counterpart is also included for the users more comfortable with Python (they lead to the same results but the Python scripts are slower).
For quicker run, '' and '' reduce the resolution to 128x128 (instead of the original 256x256 used in the paper).
