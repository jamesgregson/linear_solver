#ifndef LINEAR_SOLVER_UTILS_H
#define LINEAR_SOLVER_UTILS_H

/**
 @file linear_solver_utils.h
 @author James Gregson (james.gregson@gmail.com)
 @copyright James Gregson 2013. Licensed under MIT license
 @brief Contains a few utility routines needed by the option handling of the linear_solvers.h routines.
*/

#include<map>
#include<string>
#include<sstream>

namespace linear_solver {
    
    /** @brief stringifies an input variable */
    template< typename T >
    std::string to_str( const T &v ){
        std::ostringstream iss;
        iss << v;
        return iss.str();
    }
    
    /** @brief de-stringifies an input variable */
    template< typename T >
    T from_str( const std::string &s ){
        std::istringstream oss(s);
        T val;
        oss >> val;
        return val;
    }
    
};

#endif
