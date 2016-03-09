#include <LaplaceOperator.h>
#include <GlobalTimer.h>

template <int dim, int fe_degree, bool same_diagonal>
LaplaceOperator<dim, fe_degree, same_diagonal>::LaplaceOperator()
{}

template <int dim, int fe_degree, bool same_diagonal>
LaplaceOperator<dim, fe_degree, same_diagonal>::~LaplaceOperator()
{
  dof_handler = NULL ;
  fe = NULL ;
  mapping = NULL ;
  delete dof_info ;
  dof_info = NULL ;
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::clear()
{
  dof_handler = NULL ;
  fe = NULL ;
  mapping = NULL ;
  delete dof_info ;
  dof_info = NULL ;
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::reinit (dealii::DoFHandler<dim> * dof_handler_,
                         const dealii::MappingQ1<dim> * mapping_,
                         const dealii::ConstraintMatrix *constraints_,
                         const MPI_Comm &mpi_communicator_,
							     const unsigned int level_)
{
  global_timer.enter_subsection("LaplaceOperator::reinit");
  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=level_;
  constraints = constraints_;
  dof_info = new dealii::MeshWorker::DoFInfo<dim>{*dof_handler};
  Assert(fe->degree == fe_degree, dealii::ExcInternalError());
  const unsigned int n_gauss_points = dof_handler->get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
    dealii::update_quadrature_points |
    dealii::update_values |
    dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);

  dealii::IndexSet locally_owned_dofs;
  locally_owned_dofs = dof_handler->locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs
  (*dof_handler, locally_relevant_dofs);
  ghosted_vector.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator_);
  global_timer.leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::build_matrix ()
{  
  global_timer.enter_subsection("LaplaceOperator::build_matrix");
  info_box.initialize(*fe, *mapping);
  dealii::MGLevelObject<LA::MPI::SparseMatrix> mg_matrix ;
  mg_matrix.resize(level,level);
  const unsigned int block_size = dof_handler->block_info().local().block_size(0) ;
  const unsigned int n_blocks = same_diagonal ? 1 : dof_handler->n_dofs(level) / block_size ;
  sparsity.reinit(dof_handler->n_dofs(level),dof_handler->n_dofs(level),block_size); 
  for (unsigned int b=0;b<n_blocks;++b)
    for (unsigned int i=block_size*b;i<block_size+block_size*b;++i)
      for (unsigned int j=block_size*b;j<block_size+block_size*b;++j)
	sparsity.add(i,j);
  sparsity.compress();
  matrix.reinit(sparsity);
  mg_matrix[level].reinit(sparsity);
  dealii::MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(constraints);
#endif
  dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(level),
  						  same_diagonal ? ++dof_handler->begin_mg(level) : 
						  dof_handler->end_mg(level),
  						  *dof_info, info_box, 
  						  matrix_integrator, assembler);
  matrix.copy_from(mg_matrix[level]);
  global_timer.leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::vmult (LA::MPI::Vector &dst,
                        const LA::MPI::Vector &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::Tvmult (LA::MPI::Vector &dst,
                         const LA::MPI::Vector &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::vmult_add (LA::MPI::Vector &dst,
                        const LA::MPI::Vector &src) const
{
  global_timer.enter_subsection("LaplaceOperator::vmult_add");
  ghosted_vector = src;
  dealii::AnyData dst_data;
  dst_data.add<LA::MPI::Vector*>(&dst, "dst");
  dealii::MGLevelObject<LA::MPI::Vector > mg_src ;
  mg_src.resize(level,level) ;
  mg_src[level]= std::move(src) ;
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(&mg_src,"src");
  info_box.initialize(*fe, *mapping, src_data, 
              dealii::MGLevelObject<LA::MPI::Vector >{});
  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > assembler;
  assembler.initialize(dst_data);
#ifdef CG
  assembler.initialize(constraints);
#endif
  dealii::MeshWorker::integration_loop<dim, dim>
    (dof_handler->begin_mg(level), dof_handler->end_mg(level),
     *dof_info,info_box,residual_integrator,assembler);
  global_timer.leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::Tvmult_add (LA::MPI::Vector &dst,
                         const LA::MPI::Vector &src) const
{  
  vmult_add(dst, src);
}

template class LaplaceOperator<2, 1, true>;
template class LaplaceOperator<2, 1, false>;
template class LaplaceOperator<3, 1, true>;
template class LaplaceOperator<3, 1, false>;
