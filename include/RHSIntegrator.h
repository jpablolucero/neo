#ifndef RHSINTEGRATOR_H
#define RHSINTEGRATOR_H

#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/integrators/l2.h>

template <int dim>
class RHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  RHSIntegrator() ;
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

#endif // RHSINTEGRATOR_H
