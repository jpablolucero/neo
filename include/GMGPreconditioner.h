#ifndef GMGPRECONDITIONER_H
#define GMGPRECONDITIONER_H

#include <deal.II/distributed/tria.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_control.h>

#include <MGTransferMF.h>
#include <MFOperator.h>
#include <MfreeOperator.h>
#include <PSCPreconditioner.h>
#include <Mesh.h>
#include <Dofs.h>

#include <memory>

template <int dim,typename VectorType=LA::MPI::Vector, typename number=double,bool same_diagonal = false, unsigned int fe_degree = 1>
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
  
#ifdef MATRIXFREE
  typedef MfreeOperator<dim,fe_degree,fe_degree+1,number> SystemMatrixType;
#else
  typedef MFOperator<dim,fe_degree,number> SystemMatrixType;
#endif // MATRIXFREE
 
  int min_level ;
  int smoothing_steps ;

  Mesh<dim> &          mesh ;
  Dofs<dim> &          dofs ;
  FiniteElement<dim> & fe ;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;
  dealii::MGLevelObject<VectorType>                   mg_solution ;
  // dealii::MGConstrainedDoFs                           mg_constrained_dofs;

  std::unique_ptr<dealii::ReductionControl>              coarse_solver_control;
  dealii::PreconditionIdentity id ;

#if PARALLEL_LA < 3
  dealii::TrilinosWrappers::PreconditionSSOR coarse_preconditioner ;
  std::unique_ptr<dealii::SolverGMRES<VectorType> >              coarse_solver;
  std::unique_ptr<dealii::MGCoarseGridIterativeSolver<VectorType,
						      dealii::SolverGMRES<VectorType>,
						      LA::MPI::SparseMatrix,
						      decltype(coarse_preconditioner)> >   mg_coarse;
#else // PARALLEL_LA == 3
  std::unique_ptr<dealii::SolverCG<VectorType> >              coarse_solver;
  std::unique_ptr<dealii::MGCoarseGridIterativeSolver<VectorType,
						      dealii::SolverCG<VectorType>,
						      LA::MPI::SparseMatrix,
						      id> >   mg_coarse;

#endif
  typedef PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal> Smoother;
  //typedef MFPSCPreconditioner<dim, VectorType, number> Smoother;
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  dealii::MGSmootherPrecondition<SystemMatrixType,Smoother,VectorType> mg_smoother;

  // Setup Multigrid-Transfer
#ifdef MATRIXFREE
  std::unique_ptr<dealii::MGTransferMF<dim,SystemMatrixType> > mg_transfer ;
#else // MATRIXFREE OFF
  std::unique_ptr<dealii::MGTransferPrebuilt<VectorType> > mg_transfer ;
#endif // MATRIXFREE

  dealii::mg::Matrix<VectorType>         mglevel_matrix;
  std::unique_ptr<dealii::Multigrid<VectorType> > mg ;

#ifdef MATRIXFREE
  std::unique_ptr<dealii::PreconditionMG<dim, VectorType, dealii::MGTransferMF<dim,SystemMatrixType> > > preconditioner ;
#else
  std::unique_ptr<dealii::PreconditionMG<dim, VectorType, dealii::MGTransferPrebuilt<VectorType> > > preconditioner ;
#endif // MATRIXFREE
  
};

#include <GMGPreconditioner.templates.h>

#endif // PRECONDITIONER_H

