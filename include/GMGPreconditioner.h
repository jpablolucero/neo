#ifndef GMGPRECONDITIONER_H
#define GMGPRECONDITIONER_H

#include <deal.II/distributed/tria.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_control.h>

#include <MFOperator.h>
#include <PSCPreconditioner.h>
#include <Mesh.h>
#include <Dofs.h>

#include <memory>

template <int dim,
	  typename VectorType=dealii::parallel::distributed::Vector<double>,
	  typename number=double,
	  bool same_diagonal=false,
	  unsigned int fe_degree = 1,
	  typename Smoother=PSCPreconditioner<dim,MFOperator<dim,fe_degree,number> > >
class GMGPreconditioner final
{
 public:
  GMGPreconditioner (Mesh<dim> & mesh_,
		     Dofs<dim> & dofs_,
		     FiniteElement<dim> & fe_) ;
  
  void setup(const VectorType & solution, unsigned int min_level_ = 0);

  void vmult(VectorType &dst, const VectorType &src) const;

  void Tvmult(VectorType &dst, const VectorType &src) const;

  void vmult_add(VectorType &dst, const VectorType &src) const;

  void Tvmult_add(VectorType &dst, const VectorType &src) const;
  
  typedef MFOperator<dim,fe_degree,number> SystemMatrixType;
 
  int min_level ;
  int smoothing_steps ;

  Mesh<dim> &          mesh ;
  Dofs<dim> &          dofs ;
  FiniteElement<dim> & fe ;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;
  dealii::MGLevelObject<VectorType>                   mg_solution ;

  std::unique_ptr<dealii::ReductionControl>              coarse_solver_control;

  dealii::PreconditionIdentity id ;
  std::unique_ptr<dealii::SolverGMRES<VectorType> >              coarse_solver;
  std::unique_ptr<dealii::MGCoarseGridIterativeSolver<VectorType,
						      dealii::SolverGMRES<VectorType>,
						      SystemMatrixType,
						      decltype(id)> >   mg_coarse;

  //typedef MFPSCPreconditioner<dim, VectorType, number> Smoother;
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  dealii::MGSmootherPrecondition<SystemMatrixType,Smoother,VectorType> mg_smoother;

  // Setup Multigrid-Transfer
  std::unique_ptr<dealii::MGTransferPrebuilt<VectorType> > mg_transfer ;

  dealii::mg::Matrix<VectorType>         mglevel_matrix;
  std::unique_ptr<dealii::Multigrid<VectorType> > mg ;

  std::unique_ptr<dealii::PreconditionMG<dim, VectorType, dealii::MGTransferPrebuilt<VectorType> > > preconditioner ;
  
};

#include <GMGPreconditioner.templates.h>

#endif // PRECONDITIONER_H

