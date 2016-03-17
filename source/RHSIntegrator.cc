#include <RHSIntegrator.h>

template <int dim>
RHSIntegrator<dim>::RHSIntegrator()
{}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  dealii::Vector<double> &local_vector = dinfo.vector(0).block(0);

  // const unsigned int n_blocks = fe.get_fe().n_blocks();
  // Assert(n_blocks == fe.get_fe().n_components(), dealii::ExcDimensionMismatch(n_blocks, fe.get_fe().n_components()));

  std::vector<double> rhs_values;
  rhs_values.resize(local_vector.size(),1.0);
  dealii::LocalIntegrators::L2::L2(local_vector,fe,rhs_values);
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &, typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template class RHSIntegrator<2>;
template class RHSIntegrator<3>;
