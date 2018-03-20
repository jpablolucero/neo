#include <GMGPreconditioner.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
GMGPreconditioner ():
  min_level(0),
  smoothing_steps(1)
{}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
template <typename M,typename P>
typename std::enable_if<std::is_same<M,dealii::TrilinosWrappers::SparseMatrix>::value and
			std::is_same<P,dealii::TrilinosWrappers::PreconditionAMG>::value>::type
GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
configure_coarse_solver ()
{
  mg_matrix[min_level].build_coarse_matrix();
  const dealii::TrilinosWrappers::SparseMatrix &coarse_matrix = mg_matrix[min_level].get_coarse_matrix();
  typename CoarsePreconditionerType::AdditionalData pdata(false,false,1,false,1e-4,std::vector<std::vector<bool> >{},2,0,
							  false,"block Gauss-Seidel","Amesos-KLU");
  coarse_preconditioner.initialize(coarse_matrix, pdata);
  mg_coarse.reset(new dealii::MGCoarseGridIterativeSolver<VectorType,
		  dealii::SolverGMRES<VectorType>,
		  CoarseMatrixType,
		  CoarsePreconditionerType>
		  (*coarse_solver,
		   coarse_matrix,
		   coarse_preconditioner));
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
template <typename M,typename P>
typename std::enable_if<std::is_same<M,MFOperator<dim,fe_degree,number> >::value and
			std::is_same<P,Smoother>::value>::type
GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
configure_coarse_solver ()
{
  coarse_preconditioner.initialize(mg_matrix[min_level],smoother_data[min_level]);
  mg_coarse.reset(new dealii::MGCoarseGridIterativeSolver<VectorType,
		  dealii::SolverGMRES<VectorType>,
		  CoarseMatrixType,
		  CoarsePreconditionerType>
		  (*coarse_solver,
		   mg_matrix[min_level],
		   coarse_preconditioner));
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
template <typename M,typename P>
typename std::enable_if<std::is_same<M,MFOperator<dim,fe_degree,number> >::value and
			std::is_same<P,dealii::PreconditionIdentity>::value>::type
GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
configure_coarse_solver ()
{
  mg_coarse.reset(new dealii::MGCoarseGridIterativeSolver<VectorType,
		  dealii::SolverGMRES<VectorType>,
		  CoarseMatrixType,
		  CoarsePreconditionerType>
		  (*coarse_solver,
		   mg_matrix[min_level],
		   coarse_preconditioner));
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
initialize(const SystemMatrixType & system_matrix_,const AdditionalData &data)
{
  timer->enter_subsection("GMG::init(...)");
  Mesh<dim> & mesh = *(data.mesh) ;
  Dofs<dim> & dofs = *(data.dofs) ;
  FiniteElement<dim> & fe = *(data.fe) ;
  VectorType & solution = *(data.solution) ;
  const unsigned int n_global_levels = mesh.triangulation.n_global_levels();
  min_level = data.min_level ;
  preconditioner.reset(nullptr);
  mg.reset(nullptr);
  mg_coarse.reset(nullptr);
  mg_transfer.reset(nullptr);
  mg_smoother.clear();
  mglevel_matrix.reset();
  mg_matrix.resize(min_level, n_global_levels-1);
  dealii::MGTransferPrebuilt<VectorType> mg_transfer_tmp;
  mg_transfer_tmp.build_matrices(dofs.dof_handler);
  mg_solution.resize(min_level, n_global_levels-1);
  dealii::IndexSet locally_owned_level_dofs = dofs.dof_handler.locally_owned_mg_dofs(n_global_levels-1);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(dofs.dof_handler, n_global_levels-1, locally_relevant_level_dofs);
  mg_solution[n_global_levels-1].reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  mg_transfer_tmp.copy_to_mg(dofs.dof_handler,mg_solution,solution);
  mg_solution[n_global_levels-1].update_ghost_values();
  for (auto l = n_global_levels-1 ; l > 0 ; --l)
    {
      dealii::IndexSet locally_owned_level_dofs2 = dofs.dof_handler.locally_owned_mg_dofs(l-1);
      dealii::IndexSet locally_relevant_level_dofs2;
      dealii::DoFTools::extract_locally_relevant_level_dofs(dofs.dof_handler, l-1, locally_relevant_level_dofs);
      mg_solution[l-1].reinit(locally_owned_level_dofs2,locally_relevant_level_dofs2,*mpi_communicator);
      mg_transfer_tmp.restrict_and_add(l,mg_solution[l-1], mg_solution[l]);
      mg_solution[l-1].update_ghost_values();
    }
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].reinit(&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),level,mg_solution[level]);
    }

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
      smoother_data[level].relaxation = 0.7;
      // smoother_data[level].mg_constrained_dofs = mg_constrained_dofs;
      smoother_data[level].solution = &mg_solution[level];
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

  coarse_solver_control.reset(new dealii::ReductionControl(dofs.dof_handler.n_dofs(min_level)*10, 1.e-20, 1.e-10, false, false));
  coarse_solver.reset(new dealii::SolverGMRES<VectorType> (*coarse_solver_control) );
  configure_coarse_solver();

  // Setup Multigrid-Transfer
  mg_transfer.reset(new dealii::MGTransferPrebuilt<VectorType> {});
#ifdef CG
  // mg_transfer->initialize_constraints(dofs.constraints, mg_constrained_dofs);
#endif // CG
  mg_transfer->build_matrices(dofs.dof_handler);

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
  preconditioner.reset(new dealii::PreconditionMG<dim, VectorType, dealii::MGTransferPrebuilt<VectorType> >
		       (dofs.dof_handler, *mg, *mg_transfer));
  timer->leave_subsection();
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
vmult (VectorType &dst,const VectorType &src) const
{
  timer->enter_subsection("GMG::vmult(...)");
  preconditioner->vmult(dst,src);
  timer->leave_subsection();
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
Tvmult (VectorType &/*dst*/,const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim,typename VectorType,typename number,bool same_diagonal,unsigned int fe_degree,typename Smoother,
	  typename CoarseMatrixType,typename CoarsePreconditionerType>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
vmult_add (VectorType &dst,const VectorType &src) const
{
  timer->enter_subsection("GMG::vmult_+(...)");
  preconditioner->vmult_add(dst,src);
  timer->leave_subsection();
}

template <int dim, typename VectorType, typename number, bool same_diagonal, unsigned int fe_degree,
	  typename Smoother, typename CoarseMatrixType, typename CoarsePreconditionerType>
void GMGPreconditioner<dim,VectorType,number,same_diagonal,fe_degree,Smoother,CoarseMatrixType,CoarsePreconditionerType>::
Tvmult_add (VectorType &/*dst*/,const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}
