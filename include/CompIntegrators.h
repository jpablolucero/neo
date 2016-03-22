#ifndef COMPINTEGRATORS_H
#define COMPINTEGRATORS_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/matrix_block.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>

#include <Diffusion.h>
#include <EquationData.h>

template <int dim,bool same_diagonal=true>
class CMatrixIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  CMatrixIntegrator();
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	    dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	    typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
 private:
  Coefficient<dim> diffcoeff;
};

template <int dim>
class CResidualIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  CResidualIntegrator();
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
	  dealii::MeshWorker::DoFInfo<dim> &dinfo2,
	  typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
	  typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
 private:
  Coefficient<dim> diffcoeff;
};

template <int dim>
class CRHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
 public:
  CRHSIntegrator() ;
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

#endif // COMPINTEGRATORS_H
