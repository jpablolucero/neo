#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/function_map.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition_block.templates.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <MFOperator.h>
#include <EquationData.h>
#include <ResidualSimpleConstraints.h>
#include <PSCPreconditioner.h>
#include <MFPSCPreconditioner.h>

#include <string>
#include <fstream>

template <int dim=2,bool same_diagonal = true, unsigned int degree = 1>
class Simulator final
{
public:
  Simulator (dealii::TimerOutput &timer_,
             MPI_Comm &mpi_communicator_,
             dealii::ConditionalOStream &pcout_);
  ~Simulator ();
  Simulator (const Simulator &) = delete ;
  Simulator &operator = (const Simulator &) = delete;
  void run ();
  unsigned int n_levels ;
  unsigned int smoothing_steps ;
private:
  void setup_system ();
  void setup_multigrid ();
  void assemble_mg_interface ();
  void assemble_system ();
  void solve ();
  void compute_error () const;
  void refine_mesh ();
  void output_results (const unsigned int cycle) const;

  typedef MFOperator<dim, degree, same_diagonal, true> SystemMatrixType;
  typedef MFOperator<dim, degree, same_diagonal, false> SmootherMatrixType;

  dealii::IndexSet           locally_owned_dofs;
  dealii::IndexSet           locally_relevant_dofs;
  MPI_Comm                   &mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim>                        mapping;
  dealii::ConstraintMatrix                            constraints;
  dealii::MGConstrainedDoFs                           mg_constrained_dofs;
  dealii::FESystem<dim>                               fe;
  ReferenceFunction<dim>                              reference_function;


  dealii::DoFHandler<dim>      dof_handler;

  SystemMatrixType      system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_tmp;
  LA::MPI::Vector       right_hand_side;

  dealii::MGLevelObject<SmootherMatrixType >            mg_matrix ;
  dealii::MGLevelObject<LA::MPI::SparseMatrix>        mg_matrix_down;
  dealii::MGLevelObject<LA::MPI::SparseMatrix>        mg_matrix_up;

  dealii::ConditionalOStream &pcout;

  dealii::TimerOutput &timer;
};
#ifdef HEADER_IMPLEMENTATION
#include <Simulator.cc>
#endif

#endif // SIMULATOR_H
