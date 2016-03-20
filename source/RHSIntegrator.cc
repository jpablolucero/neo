#include <RHSIntegrator.h>

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe_v = info.fe_values();
  dealii::Vector<double> &local_vector = dinfo.vector(0).block(0);

  std::vector<double> exact_laplacian(local_vector.size());
  exact_solution.laplacian_list(fe_v.get_quadrature_points(), exact_laplacian);

  dealii::LocalIntegrators::L2::L2(local_vector,fe_v,exact_laplacian,-1.);
#ifdef CG
  dealii::LocalIntegrators::Laplace::cell_matrix(dinfo.matrix(0,false).matrix, info.fe_values());
#endif
}



template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                  typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
#ifndef CG
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  dealii::Vector<double> &local_vector = dinfo.vector(0).block(0);

  std::vector<double> boundary_values(fe.n_quadrature_points);
  exact_solution.value_list(fe.get_quadrature_points(), boundary_values);

  const unsigned int deg = fe.get_fe().tensor_degree();
  const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();

  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
      local_vector(i) += (fe.shape_value(i,k) * penalty * boundary_values[k]
                          - (fe.normal_vector(k) * fe.shape_grad(i,k)) * boundary_values[k])
                         * fe.JxW(k);
#endif
}




template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template class RHSIntegrator<2>;
template class RHSIntegrator<3>;
