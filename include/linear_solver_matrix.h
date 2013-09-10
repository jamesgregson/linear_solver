#ifndef LINEAR_SOLVER_MATRIX_H
#define LINEAR_SOLVER_MATRIX_H

/**
 @file linear_solver_matrix.h
 @author James Gregson (james.gregson@gmail.com)
 @copyright James Gregson 2013. Licensed under MIT license
 @brief Wraps gmm::row_matrix< gmm::rsvector< > > to provide basic linear algebra, uses the GMM++ BLAS calls for basic operations.
 */

#include<gmm/gmm.h>
#ifdef LINEAR_SOLVER_USES_EIGEN
#include<Eigen/Sparse>
#endif

#include"linear_solver_vector.h"

namespace linear_solver {
    
    /**
     @brief Matrix class wrapping GMM++ sparse matrix class.  Defines basic arithmetic operations to ease writing codes.  Arithmetic operations should not be used when performance is critical due to creation of temporaries, instead the GMM++ BLAS calls can be used on the sparse matrices returns by the mat() method.
     
     I chose to use the GMM++ sparse matrices rather than Eigen's sparse matrices because they support randomized insertion instead of having to use the setFromTriplets() call.  This makes it much more convenient to use the matrices, and if a linear solve is to be performed then the performance is relatively unaffected.
     */
    template< typename real >
    class sparse_matrix {
    public:
        /** @brief underlying GMM++ matrix type */
        typedef gmm::row_matrix< gmm::rsvector<real> > matrix_type;
        
        /** @brief type of writable references returned by indexing operator */
        typedef typename matrix_type::reference reference_type;
        
        /** @brief type of non-writable references returned by indexing operator */
        typedef typename matrix_type::value_type value_type;
        
    private:
        /** @brief matrix instance */
        matrix_type m_M;
    public:
        
        /** @brief construct a sparse matrix with the specified number of rows and columns */
        sparse_matrix( int nrows=0, int ncols=0 ){
            m_M.resize( nrows, ncols );
        }
        
#ifdef LINEAR_SOLVER_USES_EIGEN
        /** @brief returns an Eigen::SparseMatrix as a copy of the current matrix */
        Eigen::SparseMatrix<real> to_eigen() const {
            std::vector< Eigen::Triplet<real> > trips;
            for( int i=0; i<m_M.nrows(); i++ ){
                const gmm::rsvector<real> &r = m_M[i];
                for( typename gmm::rsvector<real>::const_iterator iter=r.begin(); iter!=r.end(); iter++ ){
                    trips.push_back( Eigen::Triplet<real>( i, iter->c, iter->e ) );
                }
            }
            Eigen::SparseMatrix<real> M( rows(), cols() );
            M.setFromTriplets( trips.begin(), trips.end() );
            return M;
        }
#endif
        
        /** @brief returns writable reference to the GMM++ underlying matrix type */
        inline matrix_type &mat(){
            return m_M;
        }
        
        /** @brief returns const reference to the GMM++ underlying matrix type */
        inline const matrix_type &mat() const {
            return m_M;
        }
        
        /** @brief empties the matrix */
        void clear(){
            m_M.clear();
        }
        
        /** @brief returns the number of matrix rows */
        inline int rows() const {
            return m_M.nrows();
        }
        
        /** @brief returns the number of matrix columns */
        inline int cols() const {
            return m_M.ncols();
        }
        
        /** @brief returns a writable reference to the (row'th, col'th) entry */
        reference_type operator()( int row, int col ){
            return m_M(row,col);
        }
        
        /** @brief returns the valud of the (row'th, col'th) entry */
        value_type operator()( int row, int col ) const {
            return m_M(row,col);
        }
        
        /** @brief resizes the matrix to the specified number of rows and columns */
        inline void resize( int rows, int cols ){
            m_M.resize(rows,cols);
        }
        
        /** @brief returns a transposed copy of the matrix */
        inline sparse_matrix<real> transpose() const {
            sparse_matrix<real> T( cols(), rows() );
            gmm::copy( gmm::transposed(m_M), T.m_M );
            return T;
        }
        
        /** @brief returns a negated copy of the matrix */
        inline sparse_matrix<real> operator-() const {
            sparse_matrix<real> res( rows(), cols() );
            gmm::copy( gmm::scaled(m_M,-1.0), res.m_M );
            return res;
        }
        
        /** @brief returns the sum of this matrix with the input matrix */
        inline sparse_matrix<real> operator+( const sparse_matrix<real> &in ) const {
            sparse_matrix<real> res( rows(), cols() );
            gmm::add( m_M, in.m_M, res.m_M );
            return res;
        }
        
        /** @brief adds the input matrix to this matrix */
        inline void operator+=( const sparse_matrix<real> &in ){
            *this = *this+in;
        }
        
        /** @brief returns the difference between this matrix and the input matrix */
        inline sparse_matrix<real> operator-( const sparse_matrix<real> &in ) const {
            sparse_matrix<real> res( rows(), cols() );
            gmm::add( m_M, gmm::scaled(in.m_M,-1.0), res.m_M );
            return res;
        }
        
        /** @brief subtracts the input matrix from this matrix */
        inline void operator-=( const sparse_matrix<real> &in ){
            *this = *this-in;
        }
        
        /** @brief returns a copy of this matrix scaled by the input scalar */
        template< typename scalar >
        inline sparse_matrix<real> operator*( const scalar in ) const {
            sparse_matrix<real> res( rows(), cols() );
            gmm::copy( gmm::scaled(m_M, in), res.m_M );
            return res;
        }
        
        /** @brief scales this matrix by the input scalar */
        template< typename scalar >
        inline void operator*=( const scalar in ){
            *this = *this*in;
        }
        
        /** @brief returns this matrix divided by the input scalar */
        template< typename scalar >
        inline sparse_matrix<real> operator/( const scalar in ) const {
            sparse_matrix<real> res( rows(), cols() );
            gmm::copy( gmm::scaled(m_M, 1.0/in), res.m_M );
            return res;
        }
        
        /** @brief divides by the input scalar */
        template< typename scalar >
        inline void operator/=( const scalar in ){
            *this = *this/in;
        }
        
        /** @brief performs a matrix-vector produce with the input vector */
        inline vector<real> operator*( const vector<real> &in ) const {
            vector<real> res( rows() );
            gmm::mult( m_M, in.vec(), res.vec() );
            return res;
        }
        
        /** @brief multiplies this matrix by the input matrix */
        inline sparse_matrix<real> operator*( const sparse_matrix<real> &in ) const {
            sparse_matrix<real> res( rows(), in.cols() );
            gmm::mult( m_M, in.m_M, res.m_M );
            return res;
        }
    };
    
    /** @brief scales a matrix by a scalar */
    template< typename scalar, typename real >
    sparse_matrix<real> operator*( const scalar a, const sparse_matrix<real> &b ){
        return b*a;
    }
    
};

#endif
