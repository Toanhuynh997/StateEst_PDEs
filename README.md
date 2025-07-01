We introduce a score-based diffusion model framework fo adaptively learning the time-evolving solutions of stochastic partial differential equations (SPDEs) through recursive Bayesian inference. In addition to traditional uncertainty quantification (UQ) efforts, the adaptive learning mechanism in this work allows to use partial noisy observational data information to reduce uncertainty in the SPDE models and enhance the accuracy of the UQ-enabled SPDE solvers.

The data assimilation technique we ultilized in this work in the Ensemble Score Filter[^1] algorithm which is very effective in tracking high-dimensional nonlinear
dynamical systems. We also consider the scenarios where the observational data is very/extremely sparse, which often undermines the performance of data assimilation. To handle this issue, we incorporate inpaiting techniques[^2][^3][^4] with the EnSF. 

>[!IMPORTANT] 
>All above files are in Jupyter Notebook. Users need to install *torch*, *skimage*, *cv2*, *torch_dct*, *joblib*, *sklearn*, and *cvxpy* in advance.

[^1]: F. Bao, Z. Zhang, G. Zhang, An ensemble score filter for tracking high-dimensional nonlinear dynamical system, Computer Methods in Applied Mechanics and Engineering, 432, Part B, 117447, 2024. (DOI:10.1016/j.cma.2024.117447)(https://www.sciencedirect.com/science/article/abs/pii/S0045782524007023?via%3Dihub).  
[^2]: M. Bertalmio, A. Bertozzi, and G. Sapiro, Navier-stokes, fluid dynamics, and image and video inpainting, Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 1 (2001), pp. I–I.  
[^3]: S. B. Damelin and N. S. Hoang, On surface completion and image inpainting by biharmonic functions: Numerical aspects, International Journal of Mathematics and Mathematical Sciences, 2018 (2018), p. 3950312.  
[^4]: S. Liang, H. Tran, F. Bao, H. Chipilski, P.J. van Leeuwen, G. Zhang, Ensemble score filter with image inpainting for data assimilation in tracking surface quasi-geostrophic dynamics with partial observations, submitted (https://arxiv.org/abs/2501.12419).
