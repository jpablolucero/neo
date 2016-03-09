#ifndef MYLAPLACE_H
#define MYLAPLACE_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
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
#include <RHSIntegrator.h>
#include <MatrixIntegrator.h>

#include <string>
#include <fstream>

template <int dim>
class MyLaplace
{
public:
  MyLaplace ();
  ~MyLaplace ();  
  void run ();

private:
  void setup_system ();
  void setup_multigrid ();
  void assemble_system ();
  void solve ();
  void compute_error () const;
  void output_results (const unsigned int cycle) const;

  typedef LaplaceOperator<dim,1> SystemMatrixType;

  dealii::IndexSet           locally_owned_dofs;
  dealii::IndexSet           locally_relevant_dofs;
  MPI_Comm                   mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim>                        mapping;
  dealii::ConstraintMatrix                            constraints;
  ReferenceFunction<dim>                              reference_function;

#ifdef CG
  dealii::FE_Q<dim>            fe;
#else
  dealii::FE_DGQ<dim>          fe;
#endif
  dealii::DoFHandler<dim>      dof_handler;
  RHSIntegrator<dim>           rhs_integrator ;

  SystemMatrixType             system_matrix;

  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_tmp;
  LA::MPI::Vector       right_hand_side;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;
  LA::MPI::SparseMatrix                               coarse_matrix ;
//  dealii::FullMatrix<double>                          coarse_matrix ;

  dealii::ConditionalOStream pcout;
};

#endif // MYLAPLACE_H
