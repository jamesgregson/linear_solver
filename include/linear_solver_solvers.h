#ifndef LINEAR_SOLVER_SOLVERS_H
#define LINEAR_SOLVER_SOLVERS_H

/**
 @file linear_solver_solvers.h
 @author James Gregson (james.gregson@gmail.com)
 @copyright James Gregson 2013. Licensed under MIT license
 @brief File for linear solver code.  Includes a variety of direct and iterative solvers.  Iterative solvers are taken from the GMM++ library and include preconditioned CG, BiCGSTAB and GMRES.  Available preconditioners are Identity, ILDLTT (incomplete Cholesky with fill limit and thresholding) and ILUT (incomplete LU with fill limit and thresholding. Direct solvers are used from Eigen3 and include SparseLU, SparseCholesky, SparseQR, SuperLU and Cholmod.
*/

#include<iostream>
#include<gmm/gmm.h>

#ifdef LINEAR_SOLVER_USES_EIGEN
#include<Eigen/Sparse>
#include<Eigen/SparseLU>
#include<Eigen/SparseQR>
#include<Eigen/OrderingMethods>
#ifdef LINEAR_SOLVER_USES_SUPERLU
#include<Eigen/SuperLUSupport>
#endif
#ifdef LINEAR_SOLVER_USES_CHOLMOD
#include<Eigen/CholmodSupport>
#endif
#endif

#include"linear_solver_utils.h"
#include"linear_solver_vector.h"
#include"linear_solver_matrix.h"

namespace linear_solver {
    /**
     @brief The solver_options type simply defines a string->string mapping allowing for general treatment of solver options. option values are parsed from the strings using the utility functions in linear_solver_utils.h
    */
    typedef std::map<std::string,std::string> solver_options;
    
    /**
     @brief The solver_cache_data provides a way to reuse already computed preconditioners and factorizations when performing repeated solves.  It is up to the user to ensure that the underlying system has not changed since the original solve.  Ths solve methods take a pointer to a solver_cache_data class as an argument, supplying a NULL pointer (the default) will cause the solver to allocate one and free it on returning.  If you are solving the same system multiple times and can spare the RAM, it's recommended that you cache the data using this class
    */
    template< typename real >
    class solver_cache_data {
        /** Identity preconditioner type, doesn't need to be cached but keeps the code cleaner */
        typedef gmm::identity_matrix                                                        identity_precond;
        
        /** Diagonal preconditioner type */
        typedef gmm::diagonal_precond< typename sparse_matrix<real>::matrix_type >          diag_precond;
        
        /** ILU preconditioner type */
        typedef gmm::ilu_precond< typename sparse_matrix<real>::matrix_type >               ilu_precond;
        
        /** ILUT preconditioner type */
        typedef gmm::ilut_precond< typename sparse_matrix<real>::matrix_type >              ilut_precond;
        
        /** ILDLT preconditioner type */
        typedef gmm::ildlt_precond< typename sparse_matrix<real>::matrix_type >             ildlt_precond;
        
        /** ILDLTT preconditioner type */
        typedef gmm::ildltt_precond< typename sparse_matrix<real>::matrix_type >            ildltt_precond;
      
#if defined(LINEAR_SOLVER_USES_EIGEN)
        /** LU factors type for built-in Eigen SparseLU solver */
        typedef Eigen::SparseLU< Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int> >    lu_factors;
        
        /** Cholesky factors type for built-in Eigen SimplicialCholesky solver */
        typedef Eigen::SimplicialCholesky< Eigen::SparseMatrix<real> >                      cholesky_factors;
        
        /** QR factors type for built-in Eigen SparseQR solver */
        typedef Eigen::SparseQR< Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int> >    qr_factors;
#endif
        
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
        /** SuperLU factorization type, requires the SuperLU library */
        typedef Eigen::SuperLU< Eigen::SparseMatrix<real> >                                 superlu_factors;
#endif

#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
        /** Cholmod factorization type, requires the Cholmod library */
        typedef Eigen::CholmodSimplicialLLT< Eigen::SparseMatrix<real> >                    cholmod_factors;
#endif
    private:
        /** Identity preconditioner instance, doesn't need to be cached but keeps the code cleaner */
        identity_precond      *IDENT=NULL;
        
        /** Diagonal preconditioner instance */
        diag_precond         *DIAG=NULL;
        
        /** ILU preconditioner instance */
        ilu_precond          *ILU=NULL;
        
        /** ILUT preconditioner instance */
        ilut_precond          *ILUT=NULL;
        
        /** ILDLT preconditioner instance */
        ildlt_precond         *ILDLT=NULL;
        
        /** ILDLTT preconditioner instance */
        ildltt_precond         *ILDLTT=NULL;
#ifdef LINEAR_SOLVER_USES_EIGEN
    
        /** LU factors instance for built-in Eigen SparseLU solver */
        lu_factors            *LU=NULL;
        
        /** Cholesky factors instance for built-in Eigen SimplicialCholesky solver */
        cholesky_factors      *CHOL=NULL;
        
        /** QR factors instance for built-in Eigen SparseQR solver */
        qr_factors            *QR=NULL;
        
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
        /** SuperLU factorization instance, requires the SuperLU library */
        superlu_factors       *SUPERLU=NULL;
#endif
        
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
        /** Cholmod factorization instance, requires the Cholmod library */
        cholmod_factors        *CHOLMOD=NULL;
#endif
#endif
    public:
        /** @brief constructor initialize all instance variables to NULL */
        solver_cache_data(){
            IDENT=NULL; ILUT=NULL; ILU=NULL;
            ILDLT=NULL; ILDLTT=NULL; DIAG=NULL;
#if defined(LINEAR_SOLVER_USES_EIGEN)
            LU=NULL; CHOL=NULL; QR=NULL;
#endif
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
            SUPERLU=NULL;
#endif
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
            CHOLMOD=NULL;
#endif
        }
        
        /** @brief deconstrutor, free any allocated memory */
        ~solver_cache_data(){
            free_cache();
        }
        
        /** @brief free allocated memory for any non-null instance, reset instance variable to NULL */
        void free_cache(){
            if( IDENT )   delete IDENT;
            if( DIAG )    delete DIAG;
            if( ILU )     delete ILU;
            if( ILUT  )   delete ILUT;
            if( ILDLT )   delete ILDLT;
            if( ILDLTT )  delete ILDLTT;
            IDENT=NULL; DIAG=NULL; ILU=NULL; ILUT=NULL; ILDLT=NULL; ILDLTT=NULL;

#if defined(LINEAR_SOLVER_USES_EIGEN)
            if( LU    )   delete LU;
            if( CHOL  )   delete CHOL;
            if( QR    )   delete QR;
            LU=NULL; CHOL=NULL; QR=NULL;
#endif
            
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
            if( SUPERLU ) delete SUPERLU;
            SUPERLU=NULL;
#endif
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
            if( CHOLMOD ) delete CHOLMOD;
            CHOLMOD=NULL;
#endif
        }
        
        /** @brief returns an identity_preconditioner instance, creating one if it does not exist */
        identity_precond *get_identity_preconditioner( const sparse_matrix<real> &M ){
            if( !IDENT )
                IDENT = new identity_precond();
            return IDENT;
        }
        
        /** @brief returns a diag_preconditioner instance, creating one if it does not exist */
        diag_precond *get_diag_preconditioner( const sparse_matrix<real> &M ){
            if( !DIAG )
                DIAG = new diag_precond(M.mat());
            return DIAG;
        }
        
        /** @brief returns an ilu_preconditioner instance, creating one if it doesn't exist */
        ilu_precond *get_ilu_preconditioner( const sparse_matrix<real> &M ){
            if( !ILU )
                ILU = new ilu_precond(M.mat());
            return ILU;
        }
        
        /** @brief returns an ilut_preconditioner instance, creating one if it does not exist */
        ilut_precond *get_ilut_preconditioner( const sparse_matrix<real> &M, const int nnz, const real drop ){
            if( !ILUT )
                ILUT = new ilut_precond( M.mat(), nnz, drop );
            return ILUT;
        }
        
        /** @brief returns an ildlt_preconditioner instance, creating one if it does not exist */
        ildlt_precond *get_ildlt_preconditioner( const sparse_matrix<real> &M ){
            if( !ILDLT )
                ILDLT = new ildlt_precond( M.mat() );
            return ILDLT;
        }
        
        /** @brief returns and ildltt_preconditioner instance, creating one if it doesn't exist */
        ildltt_precond *get_ildltt_preconditioner( const sparse_matrix<real> &M, const int nnz, const real drop ){
            if( !ILDLTT )
                ILDLTT = new ildltt_precond( M.mat(), nnz, drop );
            return ILDLTT;
        }
#if defined(LINEAR_SOLVER_USES_EIGEN)
        /** @brief returns a lu_factors instance, creating one if it does not exist */
        lu_factors *get_lu_factors( const sparse_matrix<real> &M ){
            if( !LU )
                LU = new lu_factors( M.to_eigen() );
            return LU;
        }
        
        /** @brief returns a cholesky_factors instance, creating one if it does not exist */
        cholesky_factors *get_cholesky_factors( const sparse_matrix<real> &M ){
            if( !CHOL )
                CHOL = new cholesky_factors( M.to_eigen() );
            return CHOL;
        }
        
        /** @brief returns a qr_factors instance, creating one if it does not exist */
        qr_factors *get_qr_factors( const sparse_matrix<real> &M ){
            if( !QR )
                QR = new qr_factors( M.to_eigen() );
            return QR;
        }
#endif

#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
        /** @brief returns a superlu_factors instance, creating one if it does not exist */
        superlu_factors *get_superlu_factors( const sparse_matrix<real> &M ){
            if( !SUPERLU )
                SUPERLU = new superlu_factors( M.to_eigen() );
            return SUPERLU;
        }
#endif
        
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
        /** @brief returns a cholmod_factors instance, creating one if it does not exist */
        cholmod_factors *get_cholmod_factors( const sparse_matrix<real> &M ){
            if( !CHOLMOD )
                CHOLMOD = new cholmod_factors( M.to_eigen() );
            return CHOLMOD;
        }
#endif
        
    };
    
    /** 
     @param[in]  A    Square input matrix
     @param[out] x    solution vector
     @param[in]  b    right-hand side vector
     @param[in] opts  options map containing the solver configuration options
     @param[in] cache pointer to solver_cache_data instance to hold preconditioners/factorizations for multiple solves, if desired
     @brief Solves a square linear system with a single right hand side. Has a variety of options that can be specified by the opts variable:
        
        - opts["VERBOSE"] "TRUE"/"FALSE" to turn on/off solver text output
        - opts["SOLVER"]
            -# "BICGSTAB": Use GMM++ preconditioned BiCGSTAB solver, symmetric/non-symmetric square matrices
            -# "CG": Use GMM++ preconditioned Conjugate Gradient solver, symmetric matrices only
            -# "GMRES": Use GMM++ preconditioned GMRES solver, symmetric/non-symmetric square matrices
            -# "LU": Use Eigen3 built-in SparseLU solver, symmetric/non-symmetric square matrices
            -# "CHOLESKY": Use Eigen3 built-in SimplicialCholesky solver, symmetric square matrices
            -# "QR": Use Eigen3 built-in SparseQR solver, general rectangular matrices
            -# "SUPERLU": Use SuperLU (as exposed by Eigen3), symmetric/non-symmetric square matrices
            -# "CHOLMOD": Use Cholmod Simplicial Cholesky (as exposed by Eigen3), symmetric square matrices
        - opts["PRECOND"]
            -# "NONE": No preconditioning, valid for solvers "BICGSTAB","CG" & "GMRES"
            -# "DIAG": Diagonal preconditioning, valid for "BICGSTAB", "CG" & "GMRES"
            -# "ILU":  Incomplete LU, for "BICGSTAB" and "GMRES"
            -# "ILUT": Use thresholded, fill-limited incomplete LU, preferred for "BICGSTAB", "GMRES"
            -# "ILDLT": Incomplete Cholesky, valid for "CG"
            -# "ILDLTT": Used thresholded, fill-limited incomplete Cholesky, valid for "CG"
        - opts["PRECOND_FILL"] String containing integer listing maximum number of non-zeros per row for "ILUT" and "ILDLT" preconditioners, default 20
        - opts["PRECOND_DROP"] String containing real value listing drop-tolerance for row entries of "ILUT" and "ILDLT" preconditioners, detault 1e-4
        - opts["GMRES_RESTART"] String containing integer value specifying the GMRES restart parameter 'm', default=20
        - opts["CONV_TOL"] String containing real value listing iterative solver convergence tolerance, default 1e-8
        - opts["MAX_ITERS"] String containing integer value listing maximum number of iterative solver iterations
    */
    template< typename real >
    bool solve_square_system( const sparse_matrix<real> &A, vector<real> &x, const vector<real> &b, solver_options opts=solver_options(), solver_cache_data<real> *cache=NULL, void (*iter_callback)(const std::string &msg )=NULL ){
        if( opts["SOLVER"] == "" ) opts["SOLVER"] = "BICGSTAB";
        
        int precond_fill, max_iters, gmres_restart;
        real precond_drop, conv_tol;
        bool own_cache = false;
        if( !cache ){
            own_cache = true;
            cache = new solver_cache_data<real>();
        }
        
        if( opts["SOLVER"] == "BICGSTAB" || opts["SOLVER"] == "CG" || opts["SOLVER"] == "GMRES" || opts["SOLVER"] == "LSCG" ){
            if( opts["MAX_ITERS"]     == "" ) opts["MAX_ITERS"]     = to_str(1000);
            if( opts["CONV_TOL"]      == "" ) opts["CONV_TOL"]      = to_str(1e-8);
            if( opts["PRECOND_FILL"]  == "" ) opts["PRECOND_FILL"]  = to_str(20);
            if( opts["PRECOND_DROP"]  == "" ) opts["PRECOND_DROP"]  = to_str(1e-4);
            if( opts["GMRES_RESTART"] == "" ) opts["GMRES_RESTART"] = to_str(20);
            if( opts["VERBOSE"]       == "" ) opts["VERBOSE"]       = "FALSE";
            
            max_iters     = from_str<int>(opts["MAX_ITERS"]);
            conv_tol      = from_str<real>(opts["CONV_TOL"]);
            precond_fill  = from_str<int>(opts["PRECOND_FILL"]);
            precond_drop  = from_str<real>(opts["PRECOND_DROP"]);
            gmres_restart = from_str<int>(opts["GMRES_RESTART"]);
        }
                
        if( opts["SOLVER"] == "BICGSTAB" ){
            if( opts["PRECOND"] == "" )
                opts["PRECOND"] = "ILUT";
            
            if( opts["PRECOND"] == "NONE" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::bicgstab( A.mat(), x.vec(), b.vec(), *cache->get_identity_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "DIAG" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::bicgstab( A.mat(), x.vec(), b.vec(), *cache->get_diag_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "ILU" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::bicgstab( A.mat(), x.vec(), b.vec(), *cache->get_ilu_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "ILUT" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::bicgstab( A.mat(), x.vec(), b.vec(), *cache->get_ilut_preconditioner(A,precond_fill,precond_drop), iter );
            } else {
                std::cout << "unknown preconditioner for bicgstab: " << opts["PRECOND"] << std::endl;
                std::cout << "choose from one of: NONE, ILUT" << std::endl;
                if( own_cache ) delete cache;
                return false;
            }
        } else if( opts["SOLVER"] == "CG" ){
            if( opts["PRECOND"] == "" )
                opts["PRECOND"] = "ILDLT";
            
            if( opts["PRECOND"] == "NONE" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::cg( A.mat(), x.vec(), b.vec(), *cache->get_identity_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "DIAG" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::cg( A.mat(), x.vec(), b.vec(), *cache->get_diag_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "ILDLT" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::cg( A.mat(), x.vec(), b.vec(), *cache->get_ildlt_preconditioner(A), iter );
            } else if( opts["PRECOND"] == "ILDLTT" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::cg( A.mat(), x.vec(), b.vec(), *cache->get_ildltt_preconditioner(A,precond_fill,precond_drop), iter );
            } else {
                std::cout << "unknown preconditioner for conjugate gradient: " << opts["PRECOND"] << std::endl;
                std::cout << "choose from one of: NONE, ILDLT" << std::endl;
                if( own_cache ) delete cache;
                return false;
            }
            
        } else if( opts["SOLVER"] == "GMRES" ){
            if( opts["PRECOND"] == "" )
                opts["PRECOND"] = "ILUT";
            
            if( opts["PRECOND"] == "NONE" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::gmres( A.mat(), x.vec(), b.vec(), *cache->get_identity_preconditioner(A), gmres_restart, iter );
            } else if( opts["PRECOND"] == "DIAG" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::gmres( A.mat(), x.vec(), b.vec(), *cache->get_diag_preconditioner(A), gmres_restart, iter );
            }else if( opts["PRECOND"] == "ILU" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::gmres( A.mat(), x.vec(), b.vec(), *cache->get_ilu_preconditioner(A), gmres_restart, iter );
            } else if( opts["PRECOND"] == "ILUT" ){
                gmm::iteration iter( conv_tol );
                iter.set_maxiter( max_iters );
                if( opts["VERBOSE"] == "TRUE" )
                    iter.set_noisy(1);
                gmm::gmres( A.mat(), x.vec(), b.vec(), *cache->get_ilut_preconditioner(A,precond_fill,precond_drop), gmres_restart, iter );
            } else {
                std::cout << "unkown preconditioner for gmres: " << opts["PRECOND"] << std::endl;
                std::cout << "choose from one of: NONE, ILUT" << std::endl;
                if( own_cache ) delete cache;
                return false;
            }
            
        } else if( opts["SOLVER"] == "LSCG" ){
            gmm::iteration iter( conv_tol );
            iter.set_maxiter( max_iters );
            if( opts["VERBOSE"] == "TRUE" )
                iter.set_noisy(1);
            gmm::least_squares_cg(A.mat(), x.vec(), b.vec(), iter);
            
        } else if( opts["SOLVER"] == "LU" ){
#ifdef LINEAR_SOLVER_USES_EIGEN
            x.from_eigen( cache->get_lu_factors(A)->solve(b.to_eigen()) );
#else
            std::cout << "Eigen LU solver not enabled. Recompile with LINEAR_SOLVER_USES_EIGEN defined" << std::endl;
            if( own_cache ) delete cache;
            return false;
#endif
        } else if( opts["SOLVER"] == "CHOLESKY" ){
#ifdef LINEAR_SOLVER_USES_EIGEN
            x.from_eigen( cache->get_cholesky_factors(A)->solve(b.to_eigen()) );
#else
            std::cout << "Eigen Cholesky solver not enabled. Recompile with LINEAR_SOLVER_USES_EIGEN defined" << std::endl;
            return false;
#endif
        } else if( opts["SOLVER"] == "QR" ){
#ifdef LINEAR_SOLVER_USES_EIGEN
            x.from_eigen( cache->get_qr_factors(A)->solve(b.to_eigen()) );
#else
            std::cout << "Eigen QR solver not enabled. Recompile with LINEAR_SOLVER_USES_EIGEN defined" << std::endl;
            return false;
#endif
        } else if( opts["SOLVER"] == "SUPERLU" ){
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_SUPERLU)
            x.from_eigen( cache->get_superlu_factors(A)->solve(b.to_eigen()) );
#else
            std::cout << "Eigen SuperLU solver not enabled. Recompile with LINEAR_SOLVER_USES_EIGEN and LINEAR_SOLVER_USES_SUPERLU defined" << std::endl;
            return false;
#endif
        } else if( opts["SOLVER"] == "CHOLMOD" ){
#if defined(LINEAR_SOLVER_USES_EIGEN) && defined(LINEAR_SOLVER_USES_CHOLMOD)
            x.from_eigen( cache->get_cholmod_factors(A)->solve(b.to_eigen()) );
#else
            std::cout << "Eigen Cholmod solver not enabled. Recompile with LINEAR_SOLVER_USES_EIGEN and LINEAR_SOLVER_USES_CHOLMOD defined" << std::endl;
#endif
        } else {
            std::cout << "unsupported solver: " << opts["SOLVER"] << std::endl;
            std::cout << "choose from one of: BICGSTAB, CG, GMRES, LSCG, LU, SUPERLU, CHOLESKY, QR" << std::endl;
            return false;
        }
        
        if( own_cache ) delete cache;
        return true;
    }
    
    /**
     @brief Solves a rectangular least-squares system, see description of solve_square_system().  If opts["SOLVER"] = "LSCG" or "QR" solves the sytem using GMM++ least_squares_cg solver or Eigen3 SparseQR respectively, otherwise forms the normal equations and passes the resulting system to solve_square system()
    */
    template< typename real >
    bool solve_least_squares_system( const sparse_matrix<real> &A, vector<real> &x, const vector<real> &b, solver_options opts=solver_options(), solver_cache_data<real> *cache=NULL ){
        if( opts["SOLVER"] == "LSCG" ){
            return solve_square_system( A, x, b, opts );
        } else if( opts["SOLVER"] == "QR" ){
            return solve_square_system( A, x, b, opts );
        } else {
            sparse_matrix<real> At = A.transpose();
            sparse_matrix<real> AtA = At * A;
            vector<real> Atb = At*b;
            return solve_square_system( AtA, x, Atb, opts, cache );
        }
        return false;
    }
    
    /**
     @brief Solves a rectangular, weighted least-squares system, see description of solve_square_system().  If opts["SOLVER"] = "LSCG" or "QR" solves the sytem using GMM++ least_squares_cg solver or Eigen3 SparseQR respectively, otherwise forms the normal equations and passes the resulting system to solve_square system(). Contributed by Nicholas Vining in September 2013.
     */
    template< typename real >
    bool solve_weighted_least_squares_system( const sparse_matrix<real> &A, vector<real> &x, const vector<real> &b, const vector<real> &w, solver_options opts=solver_options(), solver_cache_data<real> *cache=NULL ){
		
		sparse_matrix<real> W( A.rows(), A.rows() ),
        WA(A.rows(), A.cols()),
        AtWA( A.cols(), A.cols() );
        
		vector<real> Wb( b.size() ),
        AtWb( A.cols() );
        
		sparse_matrix<real> At = A.transpose();
        
		for( int i=0; i<(int)w.size(); i++ ){
			W(i,i) = w[i];
		}
		WA = W * A;
		AtWA = At * WA;
		Wb = W * b;
		AtWb = At * Wb;
        
        return solve_square_system( AtWA, x, AtWb, opts, cache );
    }
    
};

#endif
