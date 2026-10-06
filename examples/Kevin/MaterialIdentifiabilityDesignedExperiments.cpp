// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file MaterialIdentifiabilityDesignedExperiments.cpp
 * @brief Material-identifiability example combining volumetric and shear experiments.
 */

#include <array>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axom/sidre.hpp"
#include "mfem.hpp"
#include "mpi.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/pseudohessian_identifiability/pseudohessian_identifiability.hpp"

namespace {

constexpr int DIM = 3;
constexpr int ORDER = 1;
constexpr int NUM_PARAMETERS = 3;
constexpr double TEN_PERCENT_STRAIN = 0.10;
constexpr double TWENTY_PERCENT_STRAIN = 0.20;
constexpr double LOADED_FACE_AREA = 1.0;

const std::array<const char*, NUM_PARAMETERS> PARAMETER_NAMES = {"bulk_modulus", "shear_modulus",
                                                                 "nonlinear_stiffening"};
// A normalized value of one represents the nominal physical value of each material parameter.
constexpr std::array<double, NUM_PARAMETERS> PARAMETER_SCALES = {20.0, 5.0, 10.0};

using ParameterSpace = smith::L2<0>;
using Solid = smith::SolidMechanics<ORDER, DIM, smith::Parameters<ParameterSpace, ParameterSpace, ParameterSpace>>;

/**
 * @brief Third-order extension of the St. Venant-Kirchhoff hyperelastic model.
 *
 * The strain energy is
 *
 *   W = K/2 tr(E)^2 + G dev(E):dev(E) + C/3 tr(E)^3,
 *
 * where E is the Green-Lagrange strain. K, G, and C are supplied as
 * differentiable finite-element fields.
 */
struct ParameterizedThirdOrderElastic {
  using State = smith::Empty;

  template <int dim, typename GradType, typename BulkType, typename ShearType, typename StiffeningType>
  SMITH_HOST_DEVICE auto operator()(State&, const smith::tensor<GradType, dim, dim>& displacement_gradient,
                                    const BulkType& bulk_field, const ShearType& shear_field,
                                    const StiffeningType& stiffening_field) const
  {
    constexpr auto I = smith::Identity<dim>();
    const auto F = I + displacement_gradient;
    const auto green_strain = smith::solid_mechanics::greenStrain(displacement_gradient);
    const auto volumetric_strain = smith::tr(green_strain);

    const auto bulk_modulus = smith::get<0>(bulk_field);
    const auto shear_modulus = smith::get<0>(shear_field);
    const auto nonlinear_stiffening = smith::get<0>(stiffening_field);

    const auto volumetric_stress =
        bulk_modulus * volumetric_strain + nonlinear_stiffening * volumetric_strain * volumetric_strain;
    const auto second_piola_stress = volumetric_stress * I + 2.0 * shear_modulus * smith::dev(green_strain);
    return smith::dot(F, second_piola_stress);
  }

  double density = 1.0;
};

enum class LoadingMode
{
  HydrostaticCompression,
  HydrostaticTension,
  SimpleShear
};

/** @brief One displacement-controlled virtual material experiment. */
class MaterialExperiment {
 public:
  MaterialExperiment(std::string name, LoadingMode loading_mode, double final_strain,
                     const std::vector<double>& initial_normalized_parameters)
      : name_(std::move(name)), loading_mode_(loading_mode)
  {
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(20, 20, 20, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0), name_ + "_mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("x_min", smith::by_attr<DIM>(5));
    mesh_->addDomainOfBoundaryElements("x_max", smith::by_attr<DIM>(3));
    mesh_->addDomainOfBoundaryElements("y_min", smith::by_attr<DIM>(2));
    mesh_->addDomainOfBoundaryElements("y_max", smith::by_attr<DIM>(4));
    mesh_->addDomainOfBoundaryElements("z_min", smith::by_attr<DIM>(1));
    mesh_->addDomainOfBoundaryElements("z_max", smith::by_attr<DIM>(6));

    smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                    .relative_tol = 1.0e-10,
                                                    .absolute_tol = 1.0e-12,
                                                    .max_iterations = 30,
                                                    .print_level = 0};

    solid_ = std::make_unique<Solid>(nonlinear_options, smith::solid_mechanics::direct_linear_options,
                                     smith::solid_mechanics::default_quasistatic_options, name_, mesh_,
                                     std::vector<std::string>{PARAMETER_NAMES.begin(), PARAMETER_NAMES.end()});

    setNormalizedMaterialParameters(initial_normalized_parameters);
    solid_->setMaterial(smith::DependsOn<0, 1, 2>{}, ParameterizedThirdOrderElastic{}, mesh_->entireBody());
    setBoundaryConditions(final_strain);
    solid_->completeSetup();
  }

  /** @brief Evaluate the effective face stress and its gradient with respect to normalized material parameters. */
  void evaluate(const std::vector<double>& normalized_parameters, double& objective, mfem::Vector& gradient)
  {
    SLIC_ERROR_ROOT_IF(
        normalized_parameters.size() != NUM_PARAMETERS,
        std::format("Expected {} material parameters, but received {}.", NUM_PARAMETERS, normalized_parameters.size()));

    setNormalizedMaterialParameters(normalized_parameters);
    solid_->resetStates();
    solid_->advanceTimestep(1.0);

    auto reaction_direction = createReactionDirection();
    objective = smith::innerProduct(solid_->dual("reactions"), reaction_direction);

    smith::FiniteElementDual displacement_adjoint_load(solid_->displacement().space(),
                                                       name_ + "_displacement_adjoint_load");
    displacement_adjoint_load = 0.0;
    solid_->setAdjointLoad({{"displacement", displacement_adjoint_load}});
    solid_->setDualAdjointBcs({{"reactions", reaction_direction}});
    solid_->reverseAdjointTimestep();

    gradient.SetSize(NUM_PARAMETERS);
    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      const auto index = static_cast<std::size_t>(parameter_index);
      const auto field_sensitivity = solid_->computeTimestepSensitivity(index);
      // Apply the chain rule for physical_parameter = parameter_scale * normalized_parameter.
      gradient[parameter_index] = PARAMETER_SCALES[index] * globalConstantFieldDerivative(field_sensitivity);
    }
  }

 private:
  void setBoundaryConditions(double final_strain)
  {
    if (loading_mode_ == LoadingMode::SimpleShear) {
      solid_->setFixedBCs(mesh_->domain("x_min"));
      solid_->setDisplacementBCs(
          [final_strain](smith::vec3, double time) { return smith::vec3{{0.0, final_strain * time, 0.0}}; },
          mesh_->domain("x_max"), smith::Component::Y);
      return;
    }

    solid_->setFixedBCs(mesh_->domain("x_min"), smith::Component::X);
    solid_->setFixedBCs(mesh_->domain("y_min"), smith::Component::Y);
    solid_->setFixedBCs(mesh_->domain("z_min"), smith::Component::Z);

    const double loading_sign = loading_mode_ == LoadingMode::HydrostaticTension ? 1.0 : -1.0;
    auto prescribed_displacement = [final_strain, loading_sign](smith::vec3, double time) {
      const double displacement = loading_sign * final_strain * time;
      return smith::vec3{{displacement, displacement, displacement}};
    };
    solid_->setDisplacementBCs(prescribed_displacement, mesh_->domain("x_max"), smith::Component::X);
    solid_->setDisplacementBCs(prescribed_displacement, mesh_->domain("y_max"), smith::Component::Y);
    solid_->setDisplacementBCs(prescribed_displacement, mesh_->domain("z_max"), smith::Component::Z);
  }

  void setNormalizedMaterialParameters(const std::vector<double>& normalized_parameters)
  {
    SLIC_ERROR_ROOT_IF(
        normalized_parameters.size() != NUM_PARAMETERS,
        std::format("Expected {} material parameters, but received {}.", NUM_PARAMETERS, normalized_parameters.size()));

    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      const auto index = static_cast<std::size_t>(parameter_index);
      smith::FiniteElementState parameter_field(solid_->parameter(index));
      mfem::ConstantCoefficient constant_parameter(PARAMETER_SCALES[index] * normalized_parameters[index]);
      parameter_field.project(constant_parameter);
      solid_->setParameter(index, parameter_field);
    }
  }

  smith::FiniteElementState createReactionDirection() const
  {
    smith::FiniteElementState reaction_direction(solid_->dual("reactions").space(), name_ + "_reaction_direction");
    reaction_direction = 0.0;

    const int component = loading_mode_ == LoadingMode::SimpleShear ? 1 : 0;
    const double sign = loading_mode_ == LoadingMode::HydrostaticCompression ? -1.0 : 1.0;
    mfem::VectorFunctionCoefficient direction(DIM, [component, sign](const mfem::Vector&, mfem::Vector& value) {
      value = 0.0;
      value[component] = sign / LOADED_FACE_AREA;
    });
    reaction_direction.project(direction, mesh_->domain("x_max"));
    return reaction_direction;
  }

  static double globalConstantFieldDerivative(const smith::FiniteElementDual& field_sensitivity)
  {
    double local_derivative = field_sensitivity.Sum();
    double global_derivative = 0.0;
    MPI_Allreduce(&local_derivative, &global_derivative, 1, MPI_DOUBLE, MPI_SUM, field_sensitivity.comm());
    return global_derivative;
  }

  std::string name_;
  LoadingMode loading_mode_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::unique_ptr<Solid> solid_;
};

void printGradient(const std::string& experiment, const mfem::Vector& gradient)
{
  for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
    SLIC_INFO_ROOT(std::format("d({})/d(normalized {}) = {}", experiment,
                               PARAMETER_NAMES[static_cast<std::size_t>(parameter_index)], gradient[parameter_index]));
  }
}

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "material_identifiability_designed_experiments_data");

  const std::vector<double> normalized_material_parameters = {1.0, 1.0, 1.0};
  MaterialExperiment hydrostatic_compression("hydrostatic_compression_10", LoadingMode::HydrostaticCompression,
                                             TEN_PERCENT_STRAIN, normalized_material_parameters);
  MaterialExperiment hydrostatic_tension("hydrostatic_tension_20", LoadingMode::HydrostaticTension,
                                         TWENTY_PERCENT_STRAIN, normalized_material_parameters);
  MaterialExperiment shear("designed_simple_shear_10", LoadingMode::SimpleShear, TEN_PERCENT_STRAIN,
                           normalized_material_parameters);

  double hydrostatic_compression_stress = 0.0;
  double hydrostatic_tension_stress = 0.0;
  double shear_stress = 0.0;
  mfem::Vector hydrostatic_compression_gradient;
  mfem::Vector hydrostatic_tension_gradient;
  mfem::Vector shear_gradient;

  smith::pseudohessian_identifiability::ObjectiveFunction hydrostatic_compression_objective =
      [&](const std::vector<double>& parameters, double& objective, mfem::Vector& gradient) {
        hydrostatic_compression.evaluate(parameters, objective, gradient);
        hydrostatic_compression_stress = objective;
        hydrostatic_compression_gradient = gradient;
      };
  smith::pseudohessian_identifiability::ObjectiveFunction hydrostatic_tension_objective =
      [&](const std::vector<double>& parameters, double& objective, mfem::Vector& gradient) {
        hydrostatic_tension.evaluate(parameters, objective, gradient);
        hydrostatic_tension_stress = objective;
        hydrostatic_tension_gradient = gradient;
      };
  smith::pseudohessian_identifiability::ObjectiveFunction shear_objective =
      [&](const std::vector<double>& parameters, double& objective, mfem::Vector& gradient) {
        shear.evaluate(parameters, objective, gradient);
        shear_stress = objective;
        shear_gradient = gradient;
      };

  const std::vector<smith::pseudohessian_identifiability::ObjectiveFunction> objectives = {
      hydrostatic_compression_objective, hydrostatic_tension_objective, shear_objective};
  const auto [eigenvalues, eigenvectors] =
      smith::pseudohessian_identifiability::calcEigensystem(normalized_material_parameters, objectives);

  SLIC_INFO_ROOT(
      std::format("Hydrostatic-compression effective stress at 10% strain: {}", hydrostatic_compression_stress));
  printGradient("10% hydrostatic-compression stress", hydrostatic_compression_gradient);
  SLIC_INFO_ROOT(std::format("Hydrostatic-tension effective stress at 20% strain: {}", hydrostatic_tension_stress));
  printGradient("20% hydrostatic-tension stress", hydrostatic_tension_gradient);
  SLIC_INFO_ROOT(std::format("Simple-shear effective stress at 10% lateral strain: {}", shear_stress));
  printGradient("10% shear stress", shear_gradient);

  for (int eigenpair = 0; eigenpair < eigenvalues.Size(); ++eigenpair) {
    SLIC_INFO_ROOT(std::format("Pseudo-Hessian eigenvalue {}: {}", eigenpair, eigenvalues[eigenpair]));
    const auto& eigenvector = eigenvectors[static_cast<std::size_t>(eigenpair)];
    for (int component = 0; component < eigenvector.Size(); ++component) {
      SLIC_INFO_ROOT(std::format("  eigenvector[{}] = {}", component, eigenvector[component]));
    }
  }

  return 0;
}
