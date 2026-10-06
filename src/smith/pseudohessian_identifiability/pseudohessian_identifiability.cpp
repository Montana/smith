// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file pseudohessian_identifiability.cpp
 *
 * @brief Implementations for pseudohessian identifiability analysis.
 */

#include "smith/pseudohessian_identifiability/pseudohessian_identifiability.hpp"

#include <iostream>

namespace smith::pseudohessian_identifiability {

mfem::DenseMatrix calcPseudoHessian(const std::vector<double>& params, const ObjectiveFunction& sim_func)
{
  const int num_params = static_cast<int>(params.size());

  double objective = 0.0;
  mfem::Vector gradient(num_params);
  gradient = 0.0;
  sim_func(params, objective, gradient);

  mfem::DenseMatrix dense_hessian(num_params);
  for (int i = 0; i < num_params; ++i) {
    for (int j = 0; j < num_params; ++j) {
      dense_hessian(i, j) = gradient[i] * gradient[j];
    }
  }

  return dense_hessian;
}

mfem::DenseMatrix calcPseudoHessian(const std::vector<double>& params,
                                    const std::vector<ObjectiveFunction>& sim_func_vec)
{
  const int num_params = static_cast<int>(params.size());
  std::vector<mfem::Vector> gradient_vec;
  gradient_vec.reserve(sim_func_vec.size());
  for (const auto& sim_func : sim_func_vec) {
    double objective = 0.0;
    gradient_vec.emplace_back(num_params);
    gradient_vec.back() = 0.0;
    sim_func(params, objective, gradient_vec.back());
  }

  mfem::DenseMatrix dense_matrix(num_params);
  dense_matrix = 0.0;
  for (const auto& v : gradient_vec) {
    for (int i = 0; i < num_params; ++i) {
      for (int j = 0; j < num_params; ++j) {
        dense_matrix(i, j) += v[i] * v[j];
      }
    }
  }
  return dense_matrix;
}

EigenSystem calcEigensystem(const std::vector<double>& params, const std::vector<ObjectiveFunction>& sim_vector)
{
  const int num_params = static_cast<int>(params.size());
  // mfem::DenseMatrix dense_hessian(num_params);
  // dense_hessian = 0.0;
  //
  // for (const auto& sim_func : sim_vector) {
  //   dense_hessian.Add(1.0, calcPseudoHessian(params, sim_func));
  // }
  //
  // dense_hessian.Symmetrize();

  mfem::DenseMatrix dense_hessian = calcPseudoHessian(params, sim_vector);
  dense_hessian.Symmetrize();

  mfem::Vector eigenvalues;
  mfem::DenseMatrix dense_eigenvectors;
  dense_hessian.Eigensystem(eigenvalues, dense_eigenvectors);

  std::vector<mfem::Vector> eigenvectors;
  eigenvectors.reserve(params.size());
  for (int i = 0; i < num_params; ++i) {
    eigenvectors.emplace_back(num_params);
    dense_eigenvectors.GetColumn(i, eigenvectors.back());
  }

  return {eigenvalues, eigenvectors};
}

EigenSystem calcEigensystem(const std::vector<double>& params, const ObjectiveFunction& obj_func)
{
  const std::vector<ObjectiveFunction> obj_func_vec{obj_func};
  return calcEigensystem(params, obj_func_vec);
}

int testCalculation()
{
  std::cout << "Test" << std::endl;
  return 0;
}

}  // namespace smith::pseudohessian_identifiability
