#ifndef MATRIXINTEGRATOR_H
#define MATRIXINTEGRATOR_H

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/local_integrator.h>

template <int dim>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	    dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const;
};

#endif // MATRIXINTEGRATOR_H
