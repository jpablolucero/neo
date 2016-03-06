#include <LaplaceOperator.h>

template<int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::LaplaceOperator()
{}

template<int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::~LaplaceOperator()
{
  dof_handler = NULL ;
  fe = NULL ;
  mapping = NULL ;
  delete dof_info ;
  dof_info = NULL ;
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::reinit (dealii::DoFHandler<dim> * dof_handler_,
					     dealii::FE_DGQ<dim> * fe_,
					     const dealii::MappingQ1<dim> * mapping_,
					     const unsigned int level_)
{
  dof_handler = dof_handler_ ;
  fe = fe_ ;
  mapping = mapping_ ;
  level=level_;
  dof_info = new dealii::MeshWorker::DoFInfo<dim>{*dof_handler};
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
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::build_matrix (bool same_diagonal)
{  
  info_box.initialize(*fe, *mapping);
  dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix ;
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
  dealii::MeshWorker::Assembler::MGMatrixSimple<dealii::SparseMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
  matrix_integrator_mg.same_diagonal = same_diagonal ;
  dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(level),
  						  same_diagonal ? ++dof_handler->begin_mg(level) : 
						  dof_handler->end_mg(level),
  						  *dof_info, info_box, 
  						  matrix_integrator_mg, assembler);
  matrix.copy_from(mg_matrix[level]);
}

template<int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::vmult (dealii::Vector<double> &dst,
					    const dealii::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::Tvmult (dealii::Vector<double> &dst,
					     const dealii::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::vmult_add (dealii::Vector<double> &dst,
						const dealii::Vector<double> &src) const 
{
  dst = 0;
  dealii::AnyData dst_data;
  dst_data.add<dealii::Vector<double>*>(&dst, "dst");
  dealii::MGLevelObject<dealii::Vector<double> > mg_src ;
  mg_src.resize(level,level) ;
  mg_src[level]= std::move(src) ;
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<dealii::Vector<double> >*>(&mg_src,"src");
  info_box.initialize(*fe, *mapping, src_data, 
		      dealii::MGLevelObject<dealii::Vector<double> >{});
  dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > assembler;
  assembler.initialize(dst_data);
  dealii::MeshWorker::integration_loop<dim, dim>
    (dof_handler->begin_mg(level), dof_handler->end_mg(level),
     *dof_info,info_box,matrix_integrator,assembler);
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::Tvmult_add (dealii::Vector<double> &dst,
						 const dealii::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template class LaplaceOperator<2,1>;
template class LaplaceOperator<3,1>;
