linear_solver
=============

Common interface to the sparse iterative solvers of GMM++ and direct sparse solvers of Eigen3. Provides the GMM++ iterative solvers BiCGSTAB, CG and GMRES with Identity, incomplete LU and incomplete Cholesky pre conditioners as well as the Eigen3 SparseLU, SparseQR, SimplicialCholesky, SuperLU and Cholmod direct solvers.  Allows caching of solver data for repeated solves with different right hand sides.  

See the inline Doxygen documentation in the include/*.h files for usage as well as example/main.cpp for demo code solving a 2D Poisson equation.
