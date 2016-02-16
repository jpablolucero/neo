#ifndef MYLAPLACE_H
#define MYLAPLACE_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
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
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_generator.h>

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
  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve (dealii::Vector<double> &solution);
  void output_results () const;

  typedef LaplaceOperator<dim,1,double> SystemMatrixType;

  dealii::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim> mapping;

  dealii::FE_DGQ<dim>          fe;
  dealii::DoFHandler<dim>      dof_handler;

  SystemMatrixType     system_matrix;

  dealii::Vector<double>       solution;
  dealii::Vector<double>       right_hand_side;

  RHSIntegrator<dim>    rhs_integrator ;
    
};

#endif // MYLAPLACE_H
