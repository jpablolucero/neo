#include <LaplaceOperator.h>

template<int dim, int fe_degree, typename number>
LaplaceOperator<dim, fe_degree, number>::LaplaceOperator(const dealii::Triangulation<dim>& triangulation_,
							 const dealii::MappingQ1<dim>&  mapping_,
							 const dealii::FE_DGQ<dim>&  fe_,
							 const dealii::DoFHandler<dim>&  dof_handler_) : 
  triangulation(triangulation_),
  mapping(mapping_),
  fe(fe_),
  dof_handler(dof_handler_)
{}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::vmult (dealii::Vector<number> &dst,
	    const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::Tvmult (dealii::Vector<number> &dst,
	     const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::vmult_add (dealii::Vector<number> &dst,
						       const dealii::Vector<number> &src) const
{
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  dealii::AnyData src_data ;
  src_data.add<const dealii::Vector<double>*>(&src,"src");

  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);


  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
    dealii::update_values            |
    dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping, src_data, dealii::Vector<double>{});

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > assembler;
  dealii::AnyData dst_data;
  dst_data.add<dealii::Vector<double>*>(&dst, "dst");
  assembler.initialize(dst_data);

  MatrixIntegrator<dim> matrix_integrator ;

  dealii::MeshWorker::integration_loop<dim, dim>
    (dof_handler.begin_active(), dof_handler.end(),
     dof_info, info_box,
     matrix_integrator,
     assembler);

}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::Tvmult_add (dealii::Vector<number> &dst,
		 const dealii::Vector<number> &src) const
{
  vmult_add(dst, src);
}

template class LaplaceOperator<2,1,double>;
