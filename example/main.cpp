#include<iostream>
#include"linear_solver.h"
namespace ls = linear_solver;

int main( int argc, char **argv ){
    int n = 256;
    int N = n*n;
    ls::sparse_matrix<double> A( N, N );
    ls::vector<double> b(N), x(N);
    for( int i=0; i<n; i++ ){
        for( int j=0; j<n; j++ ){
            if( i == 0 || i == n-1 || j == 0 || j == n-1 ){
                A( i+j*n, i+j*n ) = 1.0;
                b[i+j*n] = i+j;
            } else {
                A( i+j*n, (i+0)+(j+0)*n ) =  4.0;
                A( i+j*n, (i-1)+(j+0)*n ) = -1.0;
                A( i+j*n, (i+1)+(j+0)*n ) = -1.0;
                A( i+j*n, (i+0)+(j-1)*n ) = -1.0;
                A( i+j*n, (i+0)+(j+1)*n ) = -1.0;
                b[i+j*n] = 0.0;
            }
        }
    }
    
    ls::solver_cache_data<double> cache;
    ls::solver_options opts;
    opts["VERBOSE"] = "TRUE";
    opts["SOLVER"] = "SUPERLU";
    opts["MAXITERS"] = ls::to_str(500);
    std::cout << "solving first system..." << std::endl;
    solve_square_system( A, x, b, opts, &cache );
    std::cout << "solving again with cached factorization..." << std::endl;
    solve_square_system( A, x, b, opts, &cache );
    std::cout << "Residual is: " << (A*x-b).norm() << std::endl;
    cache.free_cache();
    
    return 0;
}