#ifndef _KNN_INCLUDE
#define _KNN_INCLUDE

/**
 * @file    Contains KNN specific variables
 */

#include <cstdint>

namespace kapernikov {
    // Defines the number of neirest neighbours to use for the covariance calculation
    const uint32_t KNN_SIZE = 90U;
} // namespace kapernikov

#endif  /* _KNN_INCLUDE */
