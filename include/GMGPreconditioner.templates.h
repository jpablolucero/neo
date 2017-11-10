#include <GMGPreconditioner.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::GMGPreconditioner (Mesh<dim> & mesh_,
										  Dofs<dim> & dofs_,
										  FiniteElement<dim> & fe_):
  min_level(0),
  smoothing_steps(1),
  mesh(mesh_),
  dofs(dofs_),
  fe(fe_)
{}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::setup (const VectorType & solution, unsigned int min_level_)
{
  const unsigned int n_global_levels = mesh.triangulation.n_global_levels();
  min_level = min_level_ ;
  coarse_preconditioner.clear();
  preconditioner.reset(nullptr);
  mg.reset(nullptr);
  mg_coarse.reset(nullptr);
  mg_transfer.reset(nullptr);
  mg_smoother.clear();
  mglevel_matrix.reset();
  mg_matrix.resize(min_level, n_global_levels-1);
#ifndef MATRIXFREE
  dealii::MGTransferPrebuilt<VectorType> mg_transfer_tmp;
  mg_transfer_tmp.build_matrices(dofs.dof_handler);
  mg_solution.resize(min_level, n_global_levels-1);
  mg_transfer_tmp.copy_to_mg(dofs.dof_handler,mg_solution,solution);
  for (auto l = n_global_levels-1 ; l > 0 ; --l)
    mg_transfer_tmp.restrict_and_add(l,mg_solution[l-1], mg_solution[l]);
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].reinit(&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),level,mg_solution[level]);
    }
#else // MATRIXFREE ON
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].reinit(&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),*mpi_communicator,level);
    }
#endif // MATRIXFREE
  timer->enter_subsection("solve::mg_initialization");
  // Setup coarse solver
  coarse_solver_control.reset(new dealii::ReductionControl(dofs.dof_handler.n_dofs(min_level)*10, 1e-20, 1e-10, false, false));
#if PARALLEL_LA < 3
  mg_matrix[min_level].build_coarse_matrix();
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[min_level].get_coarse_matrix();
  coarse_solver.reset(new dealii::SolverGMRES<VectorType> (*coarse_solver_control) );
  coarse_preconditioner.initialize(coarse_matrix,1.0);
  mg_coarse.reset(new dealii::MGCoarseGridIterativeSolver<VectorType,
		  dealii::SolverGMRES<VectorType>,
		  LA::MPI::SparseMatrix,
		  decltype(coarse_preconditioner)>
		  (*coarse_solver,
		   coarse_matrix,
		   coarse_preconditioner));
#else // PARALLEL_LA == 3
  // TODO allow for Matrix-based solver for dealii MPI vectors
  coarse_solver.reset(new dealii::SolverCG<VectorType>(*coarse_solver_control));
  mg_coarse.reset(new dealii::MGCoarseGridIterativeSolver<VectorType,
		  dealii::SolverCG<VectorType>,
		  SystemMatrixType,
		  decltype(coarse_preconditioner)>
		  (coarse_solver,
                   mg_matrix[min_level],
                   id));
#endif

  // Setup Multigrid-Smoother
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());
  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      smoother_data[level].dof_handler = &(dofs.dof_handler);
      smoother_data[level].level = level;
      smoother_data[level].n_levels = n_global_levels ;
      smoother_data[level].mapping = &(fe.mapping);
      smoother_data[level].relaxation = 1.;
      // smoother_data[level].mg_constrained_dofs = mg_constrained_dofs;
#ifndef MATRIXFREE
      smoother_data[level].solution = &mg_solution[level];
#endif // MATRIXFREE
      //      uncomment to use the dictionary
      // if(!same_diagonal)
      //  {
      //    smoother_data[level].use_dictionary = true;
      //    smoother_data[level].tol = 0.05;
      //  }
      smoother_data[level].patch_type = Smoother::AdditionalData::cell_patches;
      smoother_data[level].smoother_type = Smoother::AdditionalData::additive;
      // smoother_data[level].set_fullsweep();
    }
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(smoothing_steps);

  // Setup Multigrid-Transfer
#ifdef MATRIXFREE
  mg_transfer.reset(new dealii::MGTransferMF<dim,SystemMatrixType> {mg_matrix});
  mg_transfer->build(dofs.dof_handler);
#else // MATRIXFREE OFF
  mg_transfer.reset(new dealii::MGTransferPrebuilt<VectorType> {});
#ifdef CG
  // mg_transfer->initialize_constraints(dofs.constraints, mg_constrained_dofs);
#endif // CG
  mg_transfer->build_matrices(dofs.dof_handler);
#endif // MATRIXFREE

  // Setup (Multigrid-)Preconditioner
  mglevel_matrix.initialize(mg_matrix);
  mg.reset(new dealii::Multigrid<VectorType> (mglevel_matrix,
					      *mg_coarse,
					      *mg_transfer,
					      mg_smoother,
					      mg_smoother,
					      min_level) );
  // mg.set_debug(10);
  mg->set_minlevel(mg_matrix.min_level());
  mg->set_maxlevel(mg_matrix.max_level());
#ifdef MATRIXFREE
  preconditioner.reset(new dealii::PreconditionMG<dim, VectorType, dealii::MGTransferMF<dim,SystemMatrixType> >
		       (dofs.dof_handler, *mg, *mg_transfer));
#else
  preconditioner.reset(new dealii::PreconditionMG<dim, VectorType, dealii::MGTransferPrebuilt<VectorType> >
		       (dofs.dof_handler, *mg, *mg_transfer));
#endif // MATRIXFREE
  timer->leave_subsection();
}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::vmult (VectorType &dst,
									   const VectorType &src) const
{
  preconditioner->vmult(dst,src);
}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::Tvmult (VectorType &/*dst*/,
									    const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::vmult_add (VectorType &dst,
									       const VectorType &src) const
{
  preconditioner->vmult_add(dst,src);
}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int degree>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,degree>::Tvmult_add (VectorType &/*dst*/,
										const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}
