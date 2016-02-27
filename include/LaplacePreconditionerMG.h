#ifndef LAPLACEPRECONDITIONERMG_H
#define LAPLACEPRECONDITIONERMG_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/relaxation_block.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <MatrixIntegratorMG.h>

template <int dim, int fe_degree, typename number>
class LaplacePreconditionerMG : public dealii::Subscriptor
{
public:
  LaplacePreconditionerMG (); 
  ~LaplacePreconditionerMG (); 

  void clear () ;

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

  dealii::MGLevelObject<dealii::SparsityPattern> mg_sparsity;
  dealii::MGLevelObject<dealii::SparseMatrix<number> > mg_matrix;
  dealii::MGLevelObject<dealii::SparseMatrix<number> > mg_matrix_up;
  dealii::MGLevelObject<dealii::SparseMatrix<number> > mg_matrix_down;
  dealii::MGTransferPrebuilt<dealii::Vector<number> > mg_transfer;
  dealii::FullMatrix<number> coarse_matrix;
  dealii::MGCoarseGridSVD<number, dealii::Vector<number> > mg_coarse;

};

#endif // LAPLACEPRECONDITIONERMG_H
