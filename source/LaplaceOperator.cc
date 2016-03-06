#include <LaplaceOperator.h>

template<int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::LaplaceOperator()
{}

template<int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::~LaplaceOperator()
{
  dof_handler = NULL ;
  fe = NULL ;
  triangulation = NULL ;
  mapping = NULL ;
  delete dof_info ;
  dof_info = NULL ;
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::reinit (dealii::DoFHandler<dim> * dof_handler_,
					     dealii::FE_DGQ<dim> * fe_,
					     dealii::Triangulation<dim> * triangulation_,
					     const dealii::MappingQ1<dim> * mapping_,
					     unsigned int level_,
					     bool level_matrix_)
{
  dof_handler = dof_handler_ ;
  fe = fe_ ;
  triangulation = triangulation_ ;
  mapping = mapping_ ;
  level=level_;
  level_matrix=level_matrix_;
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
void LaplaceOperator<dim,fe_degree>::build_matrix ()
{  
  info_box.initialize(*fe, *mapping);
  dealii::MGLevelObject<dealii::FullMatrix<double> > mg_matrix ;
  mg_matrix.resize(0,0);
  mg_matrix[0].reinit(dof_handler->n_dofs(0),dof_handler->n_dofs(0));
  dealii::MeshWorker::Assembler::MGMatrixSimple<dealii::FullMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
  dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(0),
						  dof_handler->end_mg(0),
						  *dof_info, info_box, matrix_integrator_mg, assembler);
  matrix.copy_from(mg_matrix[0]);
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
  dealii::AnyData dst_data;
  dst_data.add<dealii::Vector<double>*>(&dst, "dst");
  dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > assembler;
  assembler.initialize(dst_data);
  if (level_matrix)
    {
      dealii::MGLevelObject<dealii::Vector<double> > mg_src ;
      mg_src.resize(level,level) ;
      mg_src[level] = src ;
      dealii::AnyData src_data ;
      src_data.add<const dealii::MGLevelObject<dealii::Vector<double> >*>(&mg_src,"src");
      info_box.initialize(*fe, *mapping, src_data, dealii::MGLevelObject<dealii::Vector<double> >{});
      dealii::MeshWorker::integration_loop<dim, dim>
	(dof_handler->begin_mg(level), dof_handler->end_mg(level),
	 *dof_info, info_box,matrix_integrator,assembler);
    }
  else
    {
      dealii::AnyData src_data ;
      src_data.add<const dealii::Vector<double>*>(&src,"src");
      info_box.initialize(*fe, *mapping, src_data, dealii::Vector<double>{});
      dealii::MeshWorker::integration_loop<dim, dim>
  	(dof_handler->begin_active(), dof_handler->end(),
  	 *dof_info, info_box,matrix_integrator,assembler);
    }
}

template <int dim, int fe_degree>
void LaplaceOperator<dim,fe_degree>::Tvmult_add (dealii::Vector<double> &dst,
						 const dealii::Vector<double> &src) const
{
  vmult_add(dst, src);
}

template class LaplaceOperator<2,1>;
template class LaplaceOperator<3,1>;
