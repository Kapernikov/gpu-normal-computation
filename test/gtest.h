#ifndef GTEST_INCLUDE
#define GTEST_INCLUDE

/**
 * @file    Convenience wrapper for Google Test
 */

#include <gtest/gtest.h>

// Convert BDD-style macros to Google Test macros
#define SCENARIO(test, subtest, scenario_message) TEST(test, subtest)
#define GIVEN(message)
#define WHEN(message)
#define THEN(message)

#endif  /* GTEST_INCLUDE */
