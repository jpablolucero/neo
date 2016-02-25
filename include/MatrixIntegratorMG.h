#ifndef MATRIXINTEGRATORMG_H
#define MATRIXINTEGRATORMG_H

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/fe/fe_update_flags.h>

template <int dim>
class MatrixIntegratorMG final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  MatrixIntegratorMG();
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	    dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

#endif // MATRIXINTEGRATOR_H
