#ifndef LINEAR_SOLVER_SOLVERS_MPI_H
#define LINEAR_SOLVER_SOLVERS_MPI_H

#if defined(HAVE_MPI) && defined(LINEAR_SOLVER_USES_TRILINOS)
#include<mpi.h>

// Trilinos (Epetra, AztecOO) headers
#include<Epetra_FECrsMatrix.h>
#include<Epetra_MultiVector.h>
#include<Epetra_Time.h>
#include<Epetra_FEVector.h>
#include<Epetra_Version.h>
#include<Epetra_MpiComm.h>
#include<Epetra_LinearProblem.h>
#include<Epetra_Export.h>
#include<AztecOO.h>

namespace linear_solver {
    
    bool setup_default_options( solver_options &opts ){
        // default solver is GMRES is none is specified
        if( opts["SOLVER"] == "" ){
            opts["SOLVER"] = "GMRES";
        }
        
        // setup default iteration parameters, if unset
        if( opts["SOLVER"] == "GMRES" || opts["SOLVER"] == "CG" || opts["SOLVER"] == "BICGSTAB" ){
            opts["MAX_ITERS"]     = to_str(1000);
            opts["CONV_TOL"]      = to_str(1e-8);
            opts["PRECOND_FILL"]  = to_str(20);
            opts["PRECOND_DROP"]  = to_str(1e-4);
            opts["PRECOND_K"]     = to_str(3);
            opts["GMRES_RESTART"] = to_str(20);
            if( opts["SOLVER"] == "GMRES" || opts["SOLVE"] == "BICGSTAB" ){
                if( opts["PRECOND"] == "" ){
                    opts["PRECOND"] = "ILUT";
                }
            } else if( opts["SOLVER"] == "CG" ){
                if( opts["PRECOND"] == "" ){
                    opts["PRECOND"] = "ILDLTT";
                }
            }
            return true;
        }
        return false;
    }
    
    template< typename real >
    bool solve_square_system_MPI( MPI_Comm mpi_comm, const sparse_matrix<real> &A, vector<real> &x, const vector<real> &b, solver_options opts=solver_options(), solver_cache_data<real> *cache=NULL ){
     
        int rank, size, n;
        MPI_Comm_rank( mpi_comm, &rank );
        MPI_Comm_size( mpi_comm, &size );
        
        // determine the number of degrees of freedom in the problem
        // and broadcast it to all the other processes
        n = rank == 0 ? A.rows() : 0;
        MPI_Bcast( &n, 1, MPI_INT, 0,mpi_comm );
        
        // build an Epetra communicator and initialize an Epetra_map
        // which decides how degrees of freedom are allocated amonst
        // the processes in the communicator
        Epetra_MpiComm comm(mpi_comm);
        Epetra_Map map( n, 0, comm );
        Epetra_FECrsMatrix tA( Copy, map, 1 );
        Epetra_FEVector    tx( tA.OperatorDomainMap() );
        Epetra_FEVector    tb( tA.OperatorDomainMap() );
        
        // only build the system on the first process
        if( rank == 0 ){
            const gmm::row_matrix< gmm::rsvector<real> > &tM = A.mat();
            for( int i=0; i<tM.nrows(); i++ ){
                double xval = x[i], bval = b[i];
                tx.SumIntoGlobalValues( 1, &i, &xval );
                tb.SumIntoGlobalValues( 1, &i, &bval );
                const gmm::rsvector<double> &row = tM.row(i);
                for( typename gmm::rsvector<double>::const_iterator iter=row.begin(); iter!=row.end(); iter++ ){
                    int index = iter->c;
                    double value = iter->e;
                    tA.InsertGlobalValues( i, 1, &value, &index );
                }
            }
        }
        tx.GlobalAssemble();
        tb.GlobalAssemble();
        tA.GlobalAssemble();
        
        Epetra_LinearProblem problem(&tA, &tx, &tb);
        AztecOO solver(problem);
        
        // setup AztecOO options
        setup_default_options( opts );
        
        // set the solver options
        if( opts["SOLVER"] == "CG" ){
            solver.SetAztecOption( AZ_solver, AZ_cg );
            solver.SetAztecOption( AZ_type_overlap, AZ_symmetric );
        } else if( opts["SOLVER"] == "BICGSTAB" ){
            solver.SetAztecOption( AZ_solver, AZ_bicgstab );
        } else if( opts["SOLVER"] == "GMRES" ){
            solver.SetAztecOption( AZ_solver, AZ_gmres );
            solver.SetAztecOption( AZ_kspace, from_str<int>(opts["GMRES_RESTART"]) );
        }
        
        // set the preconditioner options
        if( opts["PRECOND"] == "NONE" ){
            solver.SetAztecOption( AZ_precond, AZ_none );
        } else if( opts["PRECOND"] == "ILU" ){
            solver.SetAztecOption( AZ_precond, AZ_dom_decomp );
            solver.SetAztecOption( AZ_subdomain_solve, AZ_ilu );
            solver.SetAztecOption( AZ_graph_fill, from_str<int>(opts["PRECOND_K"]) );
        } else if( opts["PRECOND"] == "ILUT" ){
            solver.SetAztecOption( AZ_precond, AZ_dom_decomp );
            solver.SetAztecOption( AZ_subdomain_solve, AZ_ilut );
            solver.SetAztecParam( AZ_ilut_fill, from_str<int>(opts["PRECOND_FILL"]) );
            solver.SetAztecParam( AZ_drop,      from_str<double>(opts["PRECOND_DROP"]) );
        } else if( opts["PRECOND"] == "ILDLT" ){
            solver.SetAztecOption( AZ_precond, AZ_icc );
            solver.SetAztecOption( AZ_graph_fill, from_str<int>(opts["PRECOND_K"]) );
        }
        
        // set the output options
        if( opts["VERBOSE"] == "TRUE" ){
            solver.SetAztecOption( AZ_output, 1 );
        } else {
            solver.SetAztecOption( AZ_output, AZ_warnings );
        }
        
        // solve the problem
        int max_iters   = from_str<int>(opts["MAX_ITERS"]);
        double conv_tol = from_str<double>(opts["CONV_TOL"]);
    
        solver.SetAztecOption( AZ_max_iter, max_iters );
        solver.Iterate( max_iters, conv_tol );
        
        // map the solution back into x
        Epetra_Map mapall( -1, rank==0?n:0, 0, comm );
        Epetra_Vector allx( mapall );
        Epetra_Export exporter(map,mapall);
        allx.Export(tx,exporter,Add);
        if( rank == 0 ){
            for( int i=0; i<n; i++ ){
                x[i] = allx[i];
            }
        }
        
        return true;
    }

template< typename real >
bool solve_square_system_MPI_simple( char *process, int nprocs, const sparse_matrix<real> &A, vector<real> &x, const vector<real> &b, solver_options opts=solver_options(), solver_cache_data<real> *cache=NULL ){
    int rank, size;
    MPI_Comm parent_comm, spawned_comm, merged_comm;
    MPI_Comm_get_parent( &parent_comm );
    if( parent_comm == MPI_COMM_NULL ){
        // parent process, spawn a communicator
        MPI_Comm_spawn( process, MPI_ARGV_NULL, nprocs-1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &spawned_comm, MPI_ERRCODES_IGNORE );
        MPI_Intercomm_merge( spawned_comm, false, &merged_comm );
        MPI_Comm_size( merged_comm, &size );
        MPI_Comm_rank( merged_comm, &rank );
        std::cout << "Master process, rank " << rank << "/" << size << std::endl;
    } else {
        // child process
        MPI_Intercomm_merge( parent_comm, true, &merged_comm );
        MPI_Comm_size( merged_comm, &size );
        MPI_Comm_rank( merged_comm, &rank );
        std::cout << "Slave process, rank " << rank << "/" << size << std::endl;
    }
    
    solve_square_system_MPI( MPI_COMM_WORLD, A, x, b, opts );
    
    // close the spawned processes and free the intercommunicators
    MPI_Comm_free( &merged_comm );
    if( rank == 0 ){
        MPI_Comm_free( &spawned_comm );
    } else {
        MPI_Comm_free( &parent_comm );
        MPI_Finalize();
        exit(0);
    }

    return true;
}

};

#endif

#endif