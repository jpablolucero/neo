#ifndef LAPLACEPRECONDITIONERMG_H
#define LAPLACEPRECONDITIONERMG_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition_block.templates.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <LaplaceOperator.h>
#include <MatrixIntegratorMG.h>

template <int dim, int fe_degree, typename number>
class LaplacePreconditionerMG : public dealii::Subscriptor
{
public:
  LaplacePreconditionerMG (); 

  ~LaplacePreconditionerMG (); 

  void reinit (dealii::DoFHandler<dim> * dof_handler_,
	       dealii::FE_DGQ<dim> * fe_,
	       dealii::Triangulation<dim> * triangulation_,
	       const dealii::MappingQ1<dim> * mapping_);
  
  void vmult (dealii::Vector<number> &dst,
              const dealii::Vector<number> &src) const;
  void Tvmult (dealii::Vector<number> &dst,
               const dealii::Vector<number> &src) const;
  void vmult_add (dealii::Vector<number> &dst,
                  const dealii::Vector<number> &src) const;
  void Tvmult_add (dealii::Vector<number> &dst,
                   const dealii::Vector<number> &src) const;

private:
  dealii::DoFHandler<dim> * dof_handler; 
  dealii::FE_DGQ<dim> * fe;
  dealii::Triangulation<dim> * triangulation;
  const dealii::MappingQ1<dim> * mapping;
  MatrixIntegratorMG<dim> matrix_integrator ;

  dealii::MGLevelObject<LaplaceOperator<dim,fe_degree,number> > mg_matrix_laplace ;
  dealii::MGLevelObject<LaplaceOperator<dim,fe_degree,number> > mg_matrix_preconditioner ;
  dealii::MGTransferPrebuilt<dealii::Vector<number> > mg_transfer;
  dealii::FullMatrix<number> coarse_matrix;
  dealii::MGCoarseGridSVD<number, dealii::Vector<number> > mg_coarse;
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  mutable dealii::mg::Matrix<dealii::Vector<number> > mgmatrixlaplace;
  dealii::MGSmootherPrecondition<LaplaceOperator<dim,fe_degree,number>,
				 dealii::PreconditionBlockJacobi<LaplaceOperator<dim,fe_degree,number> >,
				 dealii::Vector<double> > mg_smoother;
};

#endif // LAPLACEPRECONDITIONERMG_H
