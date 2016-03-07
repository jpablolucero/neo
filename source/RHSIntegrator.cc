#include <RHSIntegrator.h>

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe_v = info.fe_values();
  dealii::Vector<double> &local_vector = dinfo.vector(0).block(0);
  const std::vector<double> input_vector(local_vector.size(),1.);
  dealii::LocalIntegrators::L2::L2(local_vector,fe_v,input_vector,1.);
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
