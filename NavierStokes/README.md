We adopt dynamically regularized Lagrange multiplier[^1] (DRLM) schemes as the solver for the Navier-Stokes equation. The reference solution is obtained by solving the equations using second-order DRLM scheme, while the filtering process is performed by using a first-order DRLM. 
The Taylor–Green vortex setup is described in [^1], and our external‐force‐driven test emulates flow in an infinite, obstacle‐free channel. To reduce computational cost, the latter experiment is implemented in MATLAB rather than PyTorch. 

[^1]: C.-K. Doan, T.-T.-P. Hoang, L. Ju and R. Lan, Dynamically regularized Lagrange multiplier schemes with energy dissipation for the incompressible Navier-Stokes equations,
Journal of Computational Physics, Volume 521, Part 1, 2025, 113550, ISSN 0021-9991, (https://doi.org/10.1016/j.jcp.2024.113550).
