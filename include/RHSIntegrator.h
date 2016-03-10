#ifndef RHSINTEGRATOR_H
#define RHSINTEGRATOR_H

#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/base/function_lib.h>

#include "referencefunction.h"

template <int dim>
class RHSIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const;

private:
  //ReferenceFunction<dim> exact_solution;
  dealii::Functions::SlitSingularityFunction<dim> exact_solution;
};

#endif // RHSINTEGRATOR_H
