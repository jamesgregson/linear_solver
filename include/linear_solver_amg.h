#ifndef LINEAR_SOLVER_AMG_H
#define LINEAR_SOLVER_AMG_H

#include<vector>

#include<linear_solver.h>
namespace ls=linear_solver;

#include<timed_scope.h>

namespace linear_solver {
	
	class amg_partition_queue_compare {
	private:
		std::vector<int>	&m_valence;
	public:
		amg_partition_queue_compare( std::vector<int> &valence ) : m_valence(valence) { }
		
		inline bool operator()( const int a, const int b ) const {
			const int va=m_valence[a], vb=m_valence[b];
			return (va > vb) || (va == vb && a < b );
		}
	};
	
	template< typename real >
	int amg_partition( const ls::sparse_matrix<real> &A, std::vector<int> &coarse_label ){
		
		// loop over the vertices and find their valence
		std::vector<int> valence( A.rows() );
		coarse_label.resize( A.rows() );
		for( int i=0; i<A.rows(); i++ ){
			valence[i] = A.row_nonzeros(i);
			coarse_label[i] = -1;
		}
		
		// a flag for the valence to indicate that it
		// has been handled, valences are only set to
		// HANDLED when they are added to the coarse
		// or fine set
		const int HANDLED = -10000;
		
		// build the queue comparison object and the queue itself
		// then insert the first vertex
		amg_partition_queue_compare comp( valence );
		std::set< int, partition_queue_compare > queue( comp );
		queue.insert( 0 );
		
		// until the vertex queue has been exhausted
		int vid, nid, nnid, num_coarse=0;
		typename ls::sparse_matrix<real>::const_row_iterator niter, nend, nniter, nnend;
		while( !queue.empty() ){
			
			// grab the first entry from the queue, discard it
			// if the valence is zero (marked as handled)
			vid = *queue.begin();
			queue.erase(queue.begin());
			if( valence[vid] == HANDLED )
				continue;
			
			// label the vertex as coarse and mark it
			// as handled by setting its valence to zero
			coarse_label[vid] = num_coarse++;
			valence[vid] = HANDLED;
			
			// loop over its neighbors and mark them
			// as fine by setting their valence to
			// zero and adding them to the fine set
			niter=A.row_begin(vid); nend=A.row_end(vid);
			for( ; niter!=nend; ++niter ){
				nid = niter->c;
				queue.erase( nid );
				valence[nid] = HANDLED;
			}
			
			// loop over the neighbors
			niter=A.row_begin(vid);
			for( ; niter!=nend; ++niter ){
				nid = niter->c;
				nniter = A.row_begin(nid); nnend=A.row_end(nid);
				for( ; nniter!=nnend; ++nniter ){
					nnid = nniter->c;
					if( valence[nnid] != HANDLED ){
						queue.erase(nnid);
						valence[nnid]--;
						queue.insert(nnid);
					}
				}
			}
		}
		// return number of coarse nodes
		return num_coarse;
	}
	
	template< typename real >
	void amg_build_restriction_and_prolongation_operators( const ls::sparse_matrix<real> &A, const int num_coarse, const std::vector<int> &coarse_label, ls::sparse_matrix<real> &P, ls::sparse_matrix<real> &R ){
		
		// get the number of fine variables
		int num_fine_vars = A.rows();
		
		// resize the prolongation operator
		P.resize( num_fine_vars, num_coarse );
		R.resize( num_coarse, num_fine_vars );
		
		// loop over the rows of the matrix and
		// check if the node is coarse or fine.
		// coarse nodes are set by injection while
		// fine nodes are set by averaging of their
		// adjacent coarse neighbors
		int num_coarse_nbrs, nid;
		std::vector<int> nbr(1024);
		typename ls::sparse_matrix<real>::const_row_iterator niter, nend;
		for( int i=0; i<num_fine_vars; i++ ){
			
			if( coarse_label[i] >= 0 ){
				// coarse degree-of-freedom, set nodal value
				// by injection
				P( i, coarse_label[i] ) = 1.0;
			} else {
				// find degree-of-freedom, set nodal value
				// by averaging over the adjacent coarse
				// values
				num_coarse_nbrs = 0;
				niter=A.row_begin(i); nend=A.row_end(i);
				for( ; niter!=nend; ++niter ){
					nid = niter->c;
					if( coarse_label[nid] >= 0 ){
						nbr[num_coarse_nbrs++] = coarse_label[nid];
					}
				}
				// check that there is at least one adjacent
				// coarse variable
				if( num_coarse_nbrs == 0 ){
					std::cout << "paritioning failed, fine variable has no coarse neighbor!" << std::endl;
				}
				
				// construct the row by averaging
				for( int j=0; j<num_coarse_nbrs; j++ ){
					P( i, nbr[j] ) = real(1.0)/real(num_coarse_nbrs);
				}
			}
		}
		
		// form the restriction operator by taking the transpose
		// of the prolongation operator
		R = P.transpose();
	}
	
	template< typename real >
	void amg_build( const ls::sparse_matrix<real> &A, ls::sparse_matrix<real> &Ac, ls::sparse_matrix<real> &P, ls::sparse_matrix<real> &R ){
		int num_coarse;
		std::vector<int> coarse_label;
		num_coarse = amg_partition( A, coarse_label );
		amg_build_restriction_and_prolongation_operators( A, num_coarse, coarse_label, P, R );
		Ac = (R*A)*P;
	}
	
	template< typename real >
	void amg_solve_two_level( const ls::sparse_matrix<real> &A,
							 ls::vector<real> &x,
							 const ls::vector<real> &b,
							 ls::sparse_matrix<real> &Ac,
							 ls::sparse_matrix<real> &P,
							 ls::sparse_matrix<real> &R,
							 ls::solver_options opt_fine,
							 ls::solver_options opt_coarse,
							 ls::solver_cache_data<real> *cache_fine=NULL,
							 ls::solver_cache_data<real> *cache_coarse=NULL ){
		
		bool verbose=false, own_fine_cache=false, own_coarse_cache=false;
		real norm, conv_tol;
		int iters;
		
		// set options
		iters    = ls::from_str<int>( opt_fine["AMG_MAX_ITERS"] );
		conv_tol = ls::from_str<real>( opt_fine["AMG_CONV_TOL"] );
		verbose  = (opt_fine["AMG_VERBOSE"] == "TRUE" ) ? true : false;
		
		if( verbose )
			std::cout << "solving via algebraic multigrid..." << std::endl;
		
		// coarse system and restriction/prolongation operators do
		// not exist, construct them
		if( Ac.rows() == 0 && P.rows() == 0 && R.rows() == 0 ){
			utilities::timed_scope build_timer;
			amg_build( A, Ac, P, R );
			if( verbose )
				std::cout << "  constructed coarsened system and transfer operators in " << build_timer << " seconds." << std::endl;
		}
		
		// have to solve these systems repeatedly so,
		// check if the user provided a cache for the
		// fine and coarse system respectively and, if
		// not, allocate one that will be deleted at
		// the end of the solve
		if( !cache_fine ){
			cache_fine = new ls::solver_cache_data<real>();
			own_fine_cache=true;
		}
		if( !cache_coarse ){
			cache_coarse = new ls::solver_cache_data<real>();
			own_coarse_cache=true;
		}
		
		// start off by solving a few iterations on the coarse
		// grid to prevent aliasing of the residual
		utilities::timed_scope solve_timer;
		
		ls::solve_square_system( A, x, b, opt_fine, cache_fine );
		
		// now try to solve the system
		for( int k=0; k<iters; k++ ){
			ls::vector<real> r, xc( Ac.rows(), 0.0 );
			
			// compute residual and evaluate convergence criteria
			r = b-A*x;
			norm = r.norm();
			
			if( verbose ) std::cout << "  amg iteration [" << k+1 << "/" << iters << "], fine residual: " << norm << std::endl;
			if( norm < conv_tol )
				break;
			
			// solve for the restricted residual on the coarse mesh
			ls::solve_square_system( Ac, xc, R*r, opt_coarse, cache_coarse );
			
			// prolong correction and solve on the fine mesh
			x += P*xc;
			ls::solve_square_system( A, x, b, opt_fine, cache_fine );
		}
		if( verbose ){
			std::cout << "  finished with residual: " << norm << " in " << solve_timer << " seconds." << std::endl;
		}
		if( own_fine_cache )   delete cache_fine;
		if( own_coarse_cache ) delete cache_coarse;
	}
	
};

#endif