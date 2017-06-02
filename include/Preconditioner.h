#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

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

template <int dim=2,bool same_diagonal = true, unsigned int fe_degree = 1>
class Preconditioner final
{
 public:
  Preconditioner (Mesh<dim> & mesh_,
		  Dofs<dim> & dofs_,
		  FiniteElement<dim> & fe_,
		  dealii::TimerOutput &timer_,
		  MPI_Comm &mpi_communicator_) ;

  void setup(LA::MPI::Vector & solution);

#ifdef MATRIXFREE
  typedef MfreeOperator<dim,fe_degree,fe_degree+1,double> SystemMatrixType;
#else
  typedef MFOperator<dim,fe_degree,double> SystemMatrixType;
#endif // MATRIXFREE
 
  int min_level ;
  int smoothing_steps ;
  dealii::TimerOutput &timer;
  MPI_Comm                   &mpi_communicator;

  Mesh<dim> &          mesh ;
  Dofs<dim> &          dofs ;
  FiniteElement<dim> & fe ;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;
  dealii::MGLevelObject<LA::MPI::Vector>              mg_solution ;
  dealii::MGConstrainedDoFs                           mg_constrained_dofs;

  std::unique_ptr<dealii::SolverControl>              coarse_solver_control;
  dealii::PreconditionIdentity id;

#if PARALLEL_LA < 3
  std::unique_ptr<dealii::SolverGMRES<LA::MPI::Vector> >              coarse_solver;
  std::unique_ptr<dealii::MGCoarseGridIterativeSolver<LA::MPI::Vector,
						      dealii::SolverGMRES<LA::MPI::Vector>,
						      LA::MPI::SparseMatrix,
						      dealii::PreconditionIdentity> >   mg_coarse;
#else // PARALLEL_LA == 3
  std::unique_ptr<dealii::SolverCG<LA::MPI::Vector> >              coarse_solver;
  std::unique_ptr<dealii::MGCoarseGridIterativeSolver<LA::MPI::Vector,
						      dealii::SolverCG<LA::MPI::Vector>,
						      LA::MPI::SparseMatrix,
						      dealii::PreconditionIdentity> >   mg_coarse;

#endif
  typedef PSCPreconditioner<dim, LA::MPI::Vector, double, same_diagonal> Smoother;
  //typedef MFPSCPreconditioner<dim, LA::MPI::Vector, double> Smoother;
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  dealii::MGSmootherPrecondition<SystemMatrixType,Smoother,LA::MPI::Vector> mg_smoother;

  // Setup Multigrid-Transfer
#ifdef MATRIXFREE
  std::unique_ptr<dealii::MGTransferMF<dim,SystemMatrixType> > mg_transfer ;
#else // MATRIXFREE OFF
  std::unique_ptr<dealii::MGTransferPrebuilt<LA::MPI::Vector> > mg_transfer ;
#endif // MATRIXFREE

  dealii::mg::Matrix<LA::MPI::Vector>         mglevel_matrix;
  std::unique_ptr<dealii::Multigrid<LA::MPI::Vector> > mg ;

#ifdef MATRIXFREE
  std::unique_ptr<dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferMF<dim,SystemMatrixType> > > preconditioner ;
#else
  std::unique_ptr<dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferPrebuilt<LA::MPI::Vector> > > preconditioner ;
#endif // MATRIXFREE
  
};

#ifdef HEADER_IMPLEMENTATION
#include <Preconditioner.cc>
#endif

#endif // PRECONDITIONER_H

