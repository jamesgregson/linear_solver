#ifndef LINEAR_SOLVER_VECTOR_H
#define LINEAR_SOLVER_VECTOR_H

/**
 @file linear_solver_vector.h
 @author James Gregson (james.gregson@gmail.com)
 @copyright James Gregson 2013. Licensed under MIT license
 @brief Wraps std::vector to provide basic linear algebra, uses the GMM++ BLAS calls for basic operations.
 */


#include<vector>
#ifdef LINEAR_SOLVER_USES_EIGEN
#include<Eigen/Dense>
#endif

#include<gmm/gmm.h>

namespace linear_solver {
    
    /**
     @brief Vector class wrapping std::vector.  Defines basic arithmetic operations to ease writing codes.  Arithmetic operations should not be used when performance is critical due to creation of temporaries, instead the GMM++ BLAS calls can be used on the std::vectors returns by the vec() method.
    */
    template< typename real >
    class vector {
    public:
        typedef std::vector<real> vector_type;
    private:
        vector_type   m_V;
    public:
        /**
         @brief Construct a vector of size=size with values=val
        */
        vector( int size=0, const real val=0.0 ){
            m_V = vector_type( size, val );
        }
        
        /** @brief return writable reference to the underlying std::vector */
        inline vector_type &vec(){
            return m_V;
        }
        
        /** @brief return const reference to the underlying st::vector */
        inline const vector_type &vec() const {
            return m_V;
        }
        
#ifdef LINEAR_SOLVER_USES_EIGEN
        /** @brief return an Eigen vector as a copy */
        Eigen::Matrix<real,Eigen::Dynamic,1> to_eigen() const {
            int n = size();
            Eigen::Matrix<real,Eigen::Dynamic,1> V( n );
            for( int i=0; i<n; i++ ){
                V(i) = m_V[i];
            }
            return V;
        }
        
        /** @brief initialize the contents from an Eigen vector */
        void from_eigen( const Eigen::Matrix<real,Eigen::Dynamic,1> &in ){
            int n = in.size();
            resize( n );
            for( int i=0; i<n; i++ ){
                m_V[i] = in(i);
            }
        }
#endif
        
        /** @brief empties the vector */
        void clear(){
            m_V.clear();
        }
        
        /** @brief resizes the vector */
        inline void resize( size_t size ){
            m_V.resize( size );
        }
        
        /** @brief returns the number of elements in the vector */
        inline size_t size() const {
            return m_V.size();
        }

        /** @brief returns the 2-norm of the vector */
        inline real norm() const {
            return gmm::vect_norm2( m_V );
        }
        
        /** @brief returns the 2-norm of the vector squared */
        inline real norm_squared() const {
            return gmm::vect_norm2_sqr( m_V );
        }
        
        /** @brief return a writable reference to an entry of the vector */
        inline real &operator[]( size_t id ){
            return m_V[id];
        }
        
        /** @brief return a const reference to an entry of the vector */
        inline const real &operator[]( size_t id ) const {
            return m_V[id];
        }
        
        /** @brief return a negated copy of the vector */
        inline vector<real> operator-() const {
            vector<real> res(size());
            gmm::copy( gmm::scaled( m_V, -1.0 ), res.m_V );
            return res;
        }
        
        /** @brief returns the sum of this vector with the input vector*/
        inline vector<real> operator+( const vector<real> &in ) const {
            vector<real> res( size() );
            gmm::add( m_V, in.m_V, res.m_V );
            return res;
        }
        
        /** @brief returns the difference between this vector and the input vector */
        inline vector<real> operator-( const vector<real> &in ) const {
            vector<real> res( size() );
            gmm::add( m_V, gmm::scaled(in.m_V,-1.0), res.m_V );
            return res;
        }
        
        /** @brief returns this vector scaled by the input scalar */
        template< typename scalar >
        inline vector<real> operator*( const scalar in ) const {
            vector<real> res( size() );
            gmm::copy( gmm::scaled( m_V, in ), res.m_V );
            return res;
        }
        
        /** @brief returns this vector scaled by the inverse of the input scalar */
        template< typename scalar >
        inline vector<real> operator/( const scalar in ) const {
            vector<real> res( size() );
            gmm::copy( gmm::scaled( m_V, 1.0/in ), res.m_V );
            return res;
        }
    };
    
    /** @brief returns the vector scaled by the input scalar */
    template< typename scalar, typename real >
    vector<real> operator*( const scalar &a, const vector<real> &b ){
        return b*a;
    }
    
};

#endif