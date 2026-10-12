// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file constrained_temperature_stress.cpp
 * @brief Temperature-driven LCE mechanics at a fixed initial free-face displacement.
 */

#include <format>
#include <fstream>
#include <memory>
#include <vector>

#include "axom/slic.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "thermoelastic_material.hpp"

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "constrained_temperature_stress");

  constexpr int dim = 3;
  constexpr int order = 2;
  constexpr double beam_length = 1.0;

  auto mesh =
      std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(80, 20, 20, mfem::Element::HEXAHEDRON, beam_length,
                                                                0.1 * beam_length, 0.1 * beam_length),
                                    "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::CG,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-12,
                                            .max_iterations = 1000,
                                            .print_level = 1};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-8,
                                                  .max_iterations = 200,
                                                  .max_line_search_iterations = 20,
                                                  .print_level = 2};

  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = true};
  auto solid_system = smith::buildSolidMechanicsSystem<dim, order, smith::QuasiStaticSecondOrderTimeIntegrationRule>(
      nonlinear_options, linear_options, solid_options, mesh, smith::FieldType<smith::H1<order>>("temperature"));

  using Material = smith::examples::kevin::ThermomechanicalLCE<dim>;

  solid_system->setMaterial(Material{}, mesh->entireBodyName());
  solid_system->setDisplacementBC(mesh->domain("right"));

  constexpr double T_stop = 1.0;
  constexpr double dt = 0.01;
  constexpr int num_steps = static_cast<int>(T_stop / dt);
  auto field_store = solid_system->field_store;
  auto temperature = field_store->getParameter("param_temperature");

  auto update_temperature = [&](double time) {
    temperature.get()->setFromFieldFunction([=](smith::tensor<double, dim> X) {
      constexpr double heating_rate = 100.0;
      return heating_rate * time * (1.0 + 0.0 * X[0] / beam_length) + 300.0;
    });
  };

  using DisplacementSpace = smith::H1<order, dim>;
  const auto displacement = field_store->getField("displacement");
  auto& constrained_face = mesh->domain("left");

  smith::Functional<double(DisplacementSpace)> integrated_displacement_x({&displacement.get()->space()});
  integrated_displacement_x.AddSurfaceIntegral(
      smith::DependsOn<0>{},
      [](double, auto /*position*/, auto displacement_field) {
        return smith::get<smith::VALUE>(displacement_field)[0];
      },
      constrained_face);

  smith::Functional<double(DisplacementSpace)> integrated_constrained_face_area({&displacement.get()->space()});
  integrated_constrained_face_area.AddSurfaceIntegral(
      smith::DependsOn<>{}, [](double, auto /*position*/) { return 1.0; }, constrained_face);

  const double constrained_face_area = integrated_constrained_face_area(0.0, *displacement.get());
  SLIC_ERROR_ROOT_IF(constrained_face_area <= 0.0, "The constrained-face area must be positive.");

  smith::FiniteElementState constrained_face_x_direction(displacement.get()->space(), "constrained_face_x_direction");
  constrained_face_x_direction = 0.0;
  mfem::VectorFunctionCoefficient x_direction(dim, [](const mfem::Vector&, mfem::Vector& direction) {
    direction = 0.0;
    direction[0] = 1.0;
  });
  constrained_face_x_direction.project(x_direction, constrained_face);

  auto average_axial_stress = [&](const smith::FiniteElementDual& reactions) {
    // Boundary attribute 3 is the +x face, so its x reaction divided by reference area is tension-positive.
    return smith::innerProduct(reactions, constrained_face_x_direction) / constrained_face_area;
  };

  update_temperature(0.0);
  const smith::TimeInfo initial_time_info(-dt, dt);
  {
    SLIC_INFO_ROOT("Starting initial free-face solve");
    const auto initial_solution = solid_system->solve(initial_time_info).front();
    SLIC_INFO_ROOT("Finished initial free-face solve");

    *field_store->getField("displacement_solve_state").get() = *initial_solution.get();
    *field_store->getField("displacement").get() = *initial_solution.get();
    *field_store->getField("velocity").get() = 0.0;
    *field_store->getField("acceleration").get() = 0.0;
  }

  const double initial_average_displacement_x =
      integrated_displacement_x(0.0, *displacement.get()) / constrained_face_area;
  SLIC_INFO_ROOT(std::format("Constraining the left-face x displacement to its initial average value, {}",
                             initial_average_displacement_x));

  solid_system->disp_bc->setVectorBCs<dim>(constrained_face, std::vector<int>{0},
                                           [initial_average_displacement_x](double, smith::tensor<double, dim>) {
                                             smith::tensor<double, dim> prescribed_displacement{};
                                             prescribed_displacement[0] = initial_average_displacement_x;
                                             return prescribed_displacement;
                                           });

  smith::FiniteElementDual initial_reactions(displacement.get()->space(), "initial_reactions");
  {
    SLIC_INFO_ROOT("Starting constrained initial solve");
    const auto constrained_initial_solution = solid_system->solve(initial_time_info).front();
    SLIC_INFO_ROOT("Finished constrained initial solve");

    *field_store->getField("displacement_solve_state").get() = *constrained_initial_solution.get();
    *field_store->getField("displacement").get() = *constrained_initial_solution.get();
    *field_store->getField("velocity").get() = 0.0;
    *field_store->getField("acceleration").get() = 0.0;

    const auto initial_stress = solid_system->stress_output_system->solve(initial_time_info).front();
    *field_store->getField("stress").get() = *initial_stress.get();

    const auto reactions = solid_system->computeReactions(initial_time_info, field_store->getStateFields());
    SLIC_ERROR_ROOT_IF(reactions.size() != 1, "Expected one mechanics reaction field.");
    initial_reactions = *reactions.front().get();
  }
  field_store->graph()->reset_graph();

  auto physics = smith::makeDifferentiablePhysics(solid_system, "constrained_temperature_stress");
  auto writer = smith::createParaviewWriter(*mesh, physics->getFieldStatesAndParamStates(),
                                            "paraview_constrained_temperature_stress",
                                            smith::ParaviewWriter::Options{.write_duals = false});

  std::ofstream stress_history;
  if (mfem::Mpi::Root()) {
    stress_history.open("constrained_temperature_stress.csv");
    SLIC_ERROR_ROOT_IF(!stress_history, "Unable to open constrained_temperature_stress.csv.");
    stress_history.precision(17);
    stress_history << "time,temperature,average_axial_stress\n";
  }

  auto record_stress = [&](double time, double current_temperature, const smith::FiniteElementDual& reactions) {
    const double stress = average_axial_stress(reactions);
    if (mfem::Mpi::Root()) {
      stress_history << time << ',' << current_temperature << ',' << stress << '\n';
      stress_history.flush();
    }
    return stress;
  };

  writer.write(physics->cycle(), physics->time(), physics->getFieldStatesAndParamStates());
  const double initial_temperature = smith::max(physics->parameter("param_temperature"));
  const double initial_average_stress = record_stress(physics->time(), initial_temperature, initial_reactions);
  SLIC_INFO_ROOT(
      std::format("initial: temperature = {}, average axial stress = {}", initial_temperature, initial_average_stress));

  for (int step = 1; step <= num_steps; ++step) {
    update_temperature(physics->time() + dt);
    physics->advanceTimestep(dt);
    writer.write(physics->cycle(), physics->time(), physics->getFieldStatesAndParamStates());

    const double current_temperature = smith::max(physics->parameter("param_temperature"));
    const double stress = record_stress(physics->time(), current_temperature, physics->dual("reactions"));
    SLIC_INFO_ROOT(std::format("step {}: time = {}, temperature = {}, average axial stress = {}", step, physics->time(),
                               current_temperature, stress));
    smith::logger::flush();
  }

  return 0;
}
