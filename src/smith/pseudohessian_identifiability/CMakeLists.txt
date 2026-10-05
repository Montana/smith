# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(pseudohessian_identifiability_headers
    pseudohessian_identifiability.hpp
    )

set(pseudohessian_identifiability_sources
    pseudohessian_identifiability.cpp
    )

smith_add_library(
    NAME        smith_pseudohessian_identifiability
    HEADERS     ${pseudohessian_identifiability_headers}
    SOURCES     ${pseudohessian_identifiability_sources}
    DEPENDS_ON  smith_infrastructure
    )

smith_write_unified_header(
    NAME    pseudohessian_identifiability
    HEADERS ${pseudohessian_identifiability_headers}
    )

install(FILES ${pseudohessian_identifiability_headers}
        DESTINATION include/smith/pseudohessian_identifiability)

install(TARGETS              smith_pseudohessian_identifiability
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
