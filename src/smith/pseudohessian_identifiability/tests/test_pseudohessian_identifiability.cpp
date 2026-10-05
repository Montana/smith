// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/pseudohessian_identifiability/pseudohessian_identifiability.hpp"

namespace {

void squaredSumObjective(const std::vector<double>& params, double& objective, mfem::Vector& gradient)
{
  const double sum = params[0] + params[1];
  objective = sum * sum;

  gradient.SetSize(static_cast<int>(params.size()));
  gradient = 0.0;
  gradient[0] = 2.0 * sum;
  gradient[1] = 2.0 * sum;
}

void squaredDifferenceObjective(const std::vector<double>& params, double& objective, mfem::Vector& gradient)
{
  const double difference = params[0] - params[1];
  objective = difference * difference;

  gradient.SetSize(static_cast<int>(params.size()));
  gradient = 0.0;
  gradient[0] = 2.0 * difference;
  gradient[1] = -2.0 * difference;
}

}  // namespace

TEST(PseudoHessianIdentifiability, ComputesQuadraticObjectiveEigensystem)
{
  constexpr double tolerance = 1.0e-12;
  const std::vector<double> params = {1.0, 2.0, 4.0};

  const auto pseudo_hessian = smith::pseudohessian_identifiability::calcPseudoHessian(params, squaredSumObjective);
  EXPECT_NEAR(pseudo_hessian(0, 0), 36.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(0, 1), 36.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 0), 36.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 1), 36.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(0, 2), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 2), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 0), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 1), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 2), 0.0, tolerance);

  const auto [eigenvalues, eigenvectors] =
      smith::pseudohessian_identifiability::calcEigensystem(params, squaredSumObjective);

  ASSERT_EQ(eigenvalues.Size(), 3);
  ASSERT_EQ(eigenvectors.size(), 3);
  EXPECT_NEAR(eigenvalues[0], 0.0, tolerance);
  EXPECT_NEAR(eigenvalues[1], 0.0, tolerance);
  EXPECT_NEAR(eigenvalues[2], 72.0, tolerance);

  const mfem::Vector& dominant_eigenvector = eigenvectors[2];
  ASSERT_EQ(dominant_eigenvector.Size(), 3);
  EXPECT_NEAR(std::abs(dominant_eigenvector[0]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(std::abs(dominant_eigenvector[1]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(dominant_eigenvector[0] * dominant_eigenvector[1], 0.5, tolerance);
  EXPECT_NEAR(dominant_eigenvector[2], 0.0, tolerance);
}

TEST(PseudoHessianIdentifiability, ComputesMultipleObjectiveEigensystem)
{
  constexpr double tolerance = 1.0e-12;
  const std::vector<double> params = {1.0, 2.0, 4.0};
  const std::vector<smith::pseudohessian_identifiability::ObjectiveFunction> objectives = {squaredSumObjective,
                                                                                           squaredDifferenceObjective};

  const auto pseudo_hessian = smith::pseudohessian_identifiability::calcPseudoHessian(params, objectives);
  EXPECT_NEAR(pseudo_hessian(0, 0), 40.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(0, 1), 32.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 0), 32.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 1), 40.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(0, 2), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(1, 2), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 0), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 1), 0.0, tolerance);
  EXPECT_NEAR(pseudo_hessian(2, 2), 0.0, tolerance);

  const auto [eigenvalues, eigenvectors] = smith::pseudohessian_identifiability::calcEigensystem(params, objectives);

  ASSERT_EQ(eigenvalues.Size(), 3);
  ASSERT_EQ(eigenvectors.size(), 3);
  EXPECT_NEAR(eigenvalues[0], 0.0, tolerance);
  EXPECT_NEAR(eigenvalues[1], 8.0, tolerance);
  EXPECT_NEAR(eigenvalues[2], 72.0, tolerance);

  const mfem::Vector& difference_eigenvector = eigenvectors[1];
  ASSERT_EQ(difference_eigenvector.Size(), 3);
  EXPECT_NEAR(std::abs(difference_eigenvector[0]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(std::abs(difference_eigenvector[1]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(difference_eigenvector[0] * difference_eigenvector[1], -0.5, tolerance);
  EXPECT_NEAR(difference_eigenvector[2], 0.0, tolerance);

  const mfem::Vector& sum_eigenvector = eigenvectors[2];
  ASSERT_EQ(sum_eigenvector.Size(), 3);
  EXPECT_NEAR(std::abs(sum_eigenvector[0]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(std::abs(sum_eigenvector[1]), std::sqrt(0.5), tolerance);
  EXPECT_NEAR(sum_eigenvector[0] * sum_eigenvector[1], 0.5, tolerance);
  EXPECT_NEAR(sum_eigenvector[2], 0.0, tolerance);
}

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
