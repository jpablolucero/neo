#include <LaplacePreconditionerMG.h>

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::LaplacePreconditionerMG()
{}

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::~LaplacePreconditionerMG()
{
  delete preconditionerlaplace ;
  preconditionerlaplace = NULL ;
  delete mglaplace ;
  mglaplace = NULL ;

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
  mg_transfer.build_matrices(*dof_handler);
  const unsigned int n_levels = triangulation->n_levels();
  mg_matrix_laplace.resize(0, n_levels-1);
  mg_matrix_preconditioner.resize(0, n_levels-1);  
  for (unsigned int level=0;level<n_levels;++level)
    {
      mg_matrix_laplace[level].reinit(dof_handler,fe,triangulation,mapping,level,true);
      mg_matrix_preconditioner[level].reinit(dof_handler,fe,triangulation,mapping,level,true);
      mg_matrix_preconditioner[level].build_matrix();
    }
  coarse_matrix.reinit(dof_handler->n_dofs(0),dof_handler->n_dofs(0));
  coarse_matrix.copy_from(mg_matrix_preconditioner[0]) ;
  mg_coarse.initialize(coarse_matrix, 1.e-15);
  typename dealii::PreconditionBlockJacobi<LaplaceOperator<dim,fe_degree,number> >::AdditionalData 
    smoother_data(dof_handler->block_info().local().block_size(0),1.0,true,true);
  mg_smoother.initialize(mg_matrix_preconditioner, smoother_data);
  mgmatrixlaplace.initialize(mg_matrix_laplace);
  mglaplace = new dealii::Multigrid<dealii::Vector<number> > (*dof_handler, mgmatrixlaplace,
							      mg_coarse, mg_transfer,
							      mg_smoother, mg_smoother);
  mglaplace->set_minlevel(mg_matrix_laplace.min_level());
  mglaplace->set_maxlevel(mg_matrix_laplace.max_level());
  preconditionerlaplace = new dealii::PreconditionMG<dim, dealii::Vector<number>,
						     dealii::MGTransferPrebuilt<dealii::Vector<number> > >
    (*dof_handler, *mglaplace, mg_transfer);
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
  dst = 0;
  preconditionerlaplace->vmult_add(dst,src) ;
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::Tvmult_add (dealii::Vector<number> & dst,
							  const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template class LaplacePreconditionerMG<2,1,double>;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2,1,double>,double >;
