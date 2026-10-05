// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file pseudohessian_identifiability.hpp
 *
 * @brief Interfaces for pseudohessian identifiability analysis.
 */

#pragma once

#include <functional>
#include <tuple>
#include <vector>

#include "mfem.hpp"

namespace smith::pseudohessian_identifiability {

using ObjectiveFunction = std::function<void(const std::vector<double>&, double&, mfem::Vector&)>;
using EigenSystem = std::tuple<mfem::Vector, std::vector<mfem::Vector>>;

mfem::DenseMatrix calcPseudoHessian(const std::vector<double>& params, const ObjectiveFunction& sim_func);
mfem::DenseMatrix calcPseudoHessian(const std::vector<double>& params, const std::vector<ObjectiveFunction>& sim_func_vec);

EigenSystem calcEigensystem(const std::vector<double>& params, const std::vector<ObjectiveFunction>& sim_vector);

EigenSystem calcEigensystem(const std::vector<double>& params, const ObjectiveFunction& obj_func);

int testCalculation();

}  // namespace smith::pseudohessian_identifiability
