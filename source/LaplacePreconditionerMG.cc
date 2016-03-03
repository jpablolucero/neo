#include <LaplacePreconditionerMG.h>

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::LaplacePreconditionerMG()
{}

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::~LaplacePreconditionerMG()
{
  dof_handler = NULL ;
  fe = NULL ;
  triangulation = NULL ;
  mapping = NULL ;
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::reinit (dealii::DoFHandler<dim> * dof_handler_,
							    dealii::FE_DGQ<dim> * fe_,
							    dealii::Triangulation<dim> * triangulation_,
							    const dealii::MappingQ1<dim> * mapping_)
{
  dof_handler = dof_handler_ ;
  fe = fe_ ;
  triangulation = triangulation_ ;
  mapping = mapping_ ;
  const unsigned int n_gauss_points = dof_handler->get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points ;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, *mapping);

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler->block_info());

  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0;l<triangulation->n_levels();++l)
    dealii::deallog << ' ' << dof_handler->n_dofs(l);
  dealii::deallog << std::endl;
  
  mg_transfer.build_matrices(*dof_handler);
  const unsigned int n_levels = triangulation->n_levels();
  mg_sparsity.resize(0, n_levels-1);
  mg_matrix.resize(0, n_levels-1);
  mg_matrix_laplace.resize(0, n_levels-1);
  
  for (unsigned int level=mg_sparsity.min_level();
       level<=mg_sparsity.max_level();++level)
    {
      dealii::DynamicSparsityPattern c_sparsity(dof_handler->n_dofs(level));
      dealii::MGTools::make_flux_sparsity_pattern(*dof_handler, c_sparsity, level);
      mg_sparsity[level].copy_from(c_sparsity);
      mg_matrix[level].reinit(mg_sparsity[level]);
      mg_matrix_laplace[level].reinit(dof_handler,fe,triangulation,mapping,level,true);
    }

  dealii::deallog << "Assemble MG matrices" << std::endl;

  dealii::MeshWorker::Assembler::MGMatrixSimple<dealii::SparseMatrix<number> > assembler;
  assembler.initialize(mg_matrix);
  dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(),
						  dof_handler->end_mg(),
						  dof_info, info_box, matrix_integrator, assembler);
  coarse_matrix.reinit(0,0);
  coarse_matrix.copy_from (mg_matrix[mg_matrix.min_level()]);
  mg_coarse.initialize(coarse_matrix, 1.e-15);

  typename dealii::PreconditionBlockJacobi<dealii::SparseMatrix<number> >::AdditionalData 
    smoother_data(dof_handler->block_info().local().block_size(0),1.0,true,true);
  mg_smoother.initialize(mg_matrix, smoother_data);
  mgmatrix.initialize(mg_matrix);
}


template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::vmult (dealii::Vector<number> & dst,
	    const dealii::Vector<number> & src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::Tvmult (dealii::Vector<number> & dst,
	     const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::vmult_add (dealii::Vector<number> & dst,
							       const dealii::Vector<number> & src) const
{
  dealii::mg::Matrix<dealii::Vector<number> > mgmatrixlaplace;
  mgmatrixlaplace.initialize(mg_matrix_laplace);
  dealii::Multigrid<dealii::Vector<number> > mglaplace(*dof_handler, mgmatrixlaplace,
						       mg_coarse, mg_transfer,
						       mg_smoother, mg_smoother);
  mglaplace.set_minlevel(mg_matrix_laplace.min_level());
  mglaplace.set_maxlevel(mg_matrix_laplace.max_level());
  dealii::PreconditionMG<dim, dealii::Vector<number>,
			 dealii::MGTransferPrebuilt<dealii::Vector<number> > >
    preconditionerlaplace(*dof_handler, mglaplace, mg_transfer);
  preconditionerlaplace.vmult_add(dst,src) ;
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::Tvmult_add (dealii::Vector<number> & dst,
							  const dealii::Vector<number> &src) const
{
  vmult_add(dst, src);
}

template class LaplacePreconditionerMG<2,1,double>;
