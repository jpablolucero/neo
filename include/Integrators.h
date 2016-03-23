#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/local_integrator.h>

template <int dim,bool same_diagonal=true>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  MatrixIntegrator();
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	    dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const;
};

template <int dim>
class ResidualIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  ResidualIntegrator();
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	  dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	  typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	  typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const;
};

#endif // INTEGRATORS_H
