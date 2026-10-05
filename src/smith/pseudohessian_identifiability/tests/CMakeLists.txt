# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(pseudohessian_identifiability_test_depends
    smith_pseudohessian_identifiability
    gtest
    )

set(pseudohessian_identifiability_test_sources
    test_pseudohessian_identifiability.cpp
    )

smith_add_tests(
    SOURCES    ${pseudohessian_identifiability_test_sources}
    DEPENDS_ON ${pseudohessian_identifiability_test_depends}
    )
