#ifndef MYLAPLACE_H
#define MYLAPLACE_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition_block.templates.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <LaplaceOperator.h>
#include <BlockIntegrators.h>

#include <string>
#include <fstream>

template <int dim,bool same_diagonal = true>
class MyLaplace
{
public:
  MyLaplace ();
  ~MyLaplace ();
  void run ();

private:
  void setup_system ();
  void setup_multigrid ();
  void assemble_rhs ();
  void solve ();
  void solve_psc ();
  void solve_blockjacobi ();
  void output_results () const;

  typedef LaplaceOperator<dim, 1, same_diagonal> SystemMatrixType;

  dealii::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim> mapping;
  dealii::FESystem<dim>          fe;
  dealii::DoFHandler<dim>      dof_handler;

  SystemMatrixType             system_matrix;

  dealii::Vector<double>       solution;
  dealii::Vector<double>       right_hand_side;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;
  dealii::FullMatrix<double>                          coarse_matrix ;

  BRHSIntegrator<dim>           rhs_integrator;
  const bool use_psc = false;
};

#endif // MYLAPLACE_H
