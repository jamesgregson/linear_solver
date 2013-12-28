#ifndef LINEAR_SOLVER_BLOCK_JACOBI_H
#define LINEAR_SOLVER_BLOCK_JACOBI_H

#include<iostream>

#include<linear_solver_utils.h>
#include<linear_solver_matrix.h>
#include<linear_solver_vector.h>
#include<linear_solver_multithreading.h>

namespace linear_solver {
    
    template< typename real >
    class block_jacobi_data {
    public:
        typedef typename sparse_matrix<real>::matrix_type   mat_type;
        typedef gmm::ildltt_precond< mat_type >             pre_type;
        //typedef gmm::mr_approx_inverse_precond<mat_type>    pre_type;
        //typedef gmm::diagonal_precond<mat_type>             pre_type;
    private:
        std::vector< int >              m_block_start;
        std::vector< pre_type* >        m_pre;
        int                             m_num_threads;
        
        inline int num_blocks() const {
            return m_block_start.size()-1;
        }
        
        inline int block_start( const int blk_id ) const {
            return m_block_start[blk_id];
        }
        
        inline int block_n( const int blk_id ) const {
            return m_block_start[blk_id+1] - m_block_start[blk_id];
        }
        
        void build_subdomain_preconditioner( const sparse_matrix<real> &A, const int dom_id ){
            int s, n;
            s = block_start(dom_id);
            n = block_n(dom_id);
            mat_type blk( n, n );
            gmm::copy( gmm::sub_matrix( A.mat(), gmm::sub_interval(s,n), gmm::sub_interval(s,n) ), blk );
            pre_type *tpre = new pre_type( blk, 40, 1e-4 );
            //pre_type *tpre = new pre_type( blk );

            m_pre[dom_id] = tpre;
        }
        
        void build( const std::vector<int> &block_starts, const sparse_matrix<real> &A, const int num_threads ){
            m_num_threads = num_threads;
            m_block_start = block_starts;
            int n_dom = num_blocks();
            m_pre.resize( n_dom );
            if( num_threads == 1 ){
                for( int i=0; i<n_dom; i++ ){
                    build_subdomain_preconditioner( A, i );
                }
            } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
                auto F = [&]( const int dom ){
                    build_subdomain_preconditioner( A, dom );
                };
                parallel_run_n_times( F, num_threads, n_dom );
#else
                std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;
#endif
            }
        }
        
    public:
        block_jacobi_data( const std::vector<int> &block_starts, const sparse_matrix<real> &A, const int num_threads=1 ){
            build( block_starts, A, num_threads );
        }
        
        ~block_jacobi_data(){
            for( int i=0; i<m_pre.size(); i++ ){
                delete m_pre[i];
            }
            m_pre.clear();
            m_block_start.clear();
        }
        
        inline int num_threads() const {
            return m_num_threads;
        }
        
        vector<real> Pre( const vector<real> &x ) const {
            //return x;
            vector<real> r( x.size(), 0.0 );
            int s, n, n_dom = num_blocks();
            if( m_num_threads == 1 ){
                for( int i=0; i<n_dom; i++ ){
                    s = block_start(i);
                    n = block_n(i);
                    gmm::mult( *m_pre[i],
                              gmm::sub_vector( x.vec(), gmm::sub_interval(s,n) ),
                              gmm::sub_vector( r.vec(), gmm::sub_interval(s,n) )
                              );
                }
            } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
                auto F = [&]( const int dom ){
                    gmm::mult( *m_pre[dom],
                              gmm::sub_vector( x.vec(), gmm::sub_interval( block_start(dom), block_n(dom) ) ),
                              gmm::sub_vector( r.vec(), gmm::sub_interval( block_start(dom), block_n(dom) ) )
                              );
                };
                parallel_run_n_times( F, m_num_threads, n_dom );
#else
                std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;                
#endif
            }
            return r;
        }
        
        vector<real> A( const sparse_matrix<real> &A, const vector<real> &x ) const {
            vector<real> res( x.size(), 0.0 );
            int s, n, n_dom = num_blocks();
            if( m_num_threads == 1 ){
                for( int i=0; i<n_dom; i++ ){
                    s = block_start(i);
                    n = block_n(i);
                    gmm::mult( gmm::sub_matrix( A.mat(), gmm::sub_interval(s,n), gmm::sub_interval(0,x.size()) ),
                              x.vec(),
                              gmm::sub_vector( res.vec(), gmm::sub_interval(s,n) )
                              );
                }
            } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
                auto F = [&]( const int dom ){
                    gmm::mult( gmm::sub_matrix( A.mat(), gmm::sub_interval( block_start(dom), block_n(dom) ), gmm::sub_interval( 0, x.size() ) ),
                              x.vec(),
                              gmm::sub_vector( res.vec(), gmm::sub_interval( block_start(dom), block_n(dom) ) )
                              );
                };
                parallel_run_n_times( F, m_num_threads, n_dom );
#else
                std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;
#endif
            }
            return res;
        }
    };
    
    template< typename real >
    void copy( const vector<real> &src, vector<real> &dst, const int num_threads=1 ){
        if( dst.size() != src.size() )
            dst = vector<real>( src.size(), 0.0 );
        if( num_threads == 1 ){
            gmm::copy( src.vec(), dst.vec() );
        } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
            auto F = [&]( const int tid, const int start, const int end ){
                for( int i=start; i<end; i++ ){
                    dst[i] = src[i];
                }
            };
            parallel_for( F, num_threads, 0, src.size() );
#else
            std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;
#endif
        }
    }
    
    template< typename real >
    real dot( const vector<real> &x, const vector<real> &y, const int num_threads=1 ){
        if( num_threads == 1 ){
            return gmm::vect_sp( x.vec(), y.vec() );
        } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
            real res[LINEAR_SOLVER_MAX_THREADS];
            auto F = [&]( const int tid, const int start, const int end ){
                res[tid] = 0.0;
                for( int i=start; i<end; i++ ){
                    res[tid] += x[i]*y[i];
                }
            };
            parallel_for( F, num_threads, 0, x.size() );
            for( int i=1; i<num_threads; i++ ){
                res[0] += res[i];
            }
            return res[0];
#else
            std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;
#endif
        }
        return 0;
    }
    
    template< typename real >
    real axpy( real alpha, const vector<real> &x, vector<real> &y, const int num_threads=1 ){
        if( num_threads == 1 ){
            gmm::add( gmm::scaled( x.vec(), alpha ), y.vec() );
        } else {
#if defined(LINEAR_SOLVER_MULTITHREADING)
            auto F = [&]( const int tid, const int start, const int end ){
                for( int i=start; i<end; i++ ){
                    y[i] += alpha*x[i];
                }
            };
            parallel_for( F, num_threads, 0, x.size() );
#else
            std::cout << "Multithreading requested but LINEAR_SOLVER_MULTITHREADING not defined!" << std::endl;
#endif
        }
        return 0;
    }
    
    template< typename real >
    void block_jacobi(
                      const std::vector<int>& block_start,
                      const sparse_matrix<real>& A,
                      vector<real>& x,
                      const vector<real>& b,
                      block_jacobi_data<real> &P, solver_options opts=solver_options() ){
        
        // set some default options if they are not already set
        if( opts["MAX_ITERS"] == "" ) opts["MAX_ITERS"] = to_str(500);
        if( opts["CONV_TOL"]  == "" ) opts["CONV_TOL"]  = to_str(1e-6);
        
        int max_iters = from_str<int>(opts["MAX_ITERS"]);
        bool verbose  = opts["VERBOSE"] == std::string("TRUE") ? true : false;
        real rel_tol  = from_str<real>(opts["CONV_TOL"]);
        real abs_tol  = 1e-12;
        
        int n = x.size(), n_threads = P.num_threads();
        real alpha, beta, rr, rr_init, zr, zr_old;
        vector<real> r(n,0.0), z(n,0.0), p(n,0.0), Ap(n,0.0);
        
        r = b - P.A(A,x);
        rr = rr_init = dot<real>(r,r,n_threads); //r.dot(r);
        if( rr < abs_tol ){
            if( verbose )
                std::cout << "cg finished with residual: " << sqrt(rr) << std::endl;
            return;
        } else {
            if( verbose )
                std::cout << "cg started with rel. resid.: " << 1 << std::endl;
        }
        z = P.Pre(r);
        p = z;
        zr = zr_old = dot<real>(z,r,n_threads); // z.dot(r);
        
        for( int iter=0; iter<max_iters; iter++ ){
            Ap = P.A(A,p);
            alpha = zr/dot<real>(p,Ap,n_threads); //alpha = zr_old/p.dot(Ap);
            axpy( alpha, p,  x, n_threads ); // x += alpha*p;
            axpy(-alpha, Ap, r, n_threads ); // r -= alpha*Ap;
            rr = dot<real>(r,r, n_threads );  // rr = r.dot(r);
            if( iter == 0 || iter % 5 == 0 ){
                if( verbose )
                    std::cout << "  iteration [" << iter << "/" << max_iters << "], rel. resid.: " << sqrt(rr/rr_init) << std::endl;
            }
            if( sqrt(rr/rr_init) < rel_tol || rr < abs_tol )
                break;
            
            z = P.Pre(r);
            zr = dot<real>(z,r, n_threads); // z.dot(r);
            beta = zr/zr_old;
            p = z + beta*p;
            
            zr_old = zr;
        }
        if( verbose ){
            std::cout << "cg finished with rel. resid.: " << sqrt(rr/rr_init) << std::endl;
            //std::cout << "cg finished resid.: " << sqrt(rr) << std::endl;
		}
        return;
    }
    
};

#endif