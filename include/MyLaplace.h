#ifndef MYLAPLACE_H
#define MYLAPLACE_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
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
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <LaplaceOperator.h>
#include <EquationData.h>
#include <ResidualSimpleConstraints.h>

#include <string>
#include <fstream>

template <int dim=2,bool same_diagonal = true, unsigned int degree = 1>
class MyLaplace
{
public:
  MyLaplace (dealii::TimerOutput &timer_,
             MPI_Comm &mpi_communicator_,
             dealii::ConditionalOStream &pcout_);
  ~MyLaplace ();
  void run ();

private:
  void setup_system ();
  void setup_multigrid ();
  void assemble_system ();
  void solve ();
  void compute_error () const;
  void output_results (const unsigned int cycle) const;

  typedef LaplaceOperator<dim, degree, same_diagonal> SystemMatrixType;

  dealii::IndexSet           locally_owned_dofs;
  dealii::IndexSet           locally_relevant_dofs;
  MPI_Comm                   &mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim>                        mapping;
  dealii::ConstraintMatrix                            constraints;
  dealii::FESystem<dim>                               fe;
  ReferenceFunction<dim>                              reference_function;


  dealii::DoFHandler<dim>      dof_handler;

  SystemMatrixType             system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_tmp;
  LA::MPI::Vector       right_hand_side;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;

  dealii::ConditionalOStream &pcout;

  dealii::TimerOutput &timer;
};

#endif // MYLAPLACE_H
