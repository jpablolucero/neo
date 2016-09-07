#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/matrix_block.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/base/vector_slice.h>

#include <Diffusion.h>
#include <Transport.h>
#include <EquationData.h>

template <int dim>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  MatrixIntegrator();
  MatrixIntegrator (const MatrixIntegrator &) = delete ;
  MatrixIntegrator &operator = (const MatrixIntegrator &) = delete;
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
protected:
  Coefficient<dim> diffcoeff;
};

template <int dim>
class ResidualIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  ResidualIntegrator();
  ResidualIntegrator (const ResidualIntegrator &) = delete ;
  ResidualIntegrator &operator = (const ResidualIntegrator &) = delete;
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
class RHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  RHSIntegrator(unsigned int n_components);
  RHSIntegrator (const RHSIntegrator &) = delete ;
  RHSIntegrator &operator = (const RHSIntegrator &) = delete;
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
private:
  Coefficient<dim> diffcoeff;
  ReferenceFunction<dim> exact_solution;
};

#ifdef HEADER_IMPLEMENTATION
#include <Integrators.cc>
#endif

#endif // INTEGRATORS_H
