#ifndef PSCINTEGRATORS_H
#define PSCINTEGRATORS_H
#include <Integrators.h>

#ifdef MATRIXFREE
template <int dim>
class PSCMatrixIntegrator final : public MatrixIntegrator<dim>
{
public:
  PSCMatrixIntegrator();
  PSCMatrixIntegrator (const PSCMatrixIntegrator &) = delete ;
  PSCMatrixIntegrator &operator = (const PSCMatrixIntegrator &) = delete;
  // virtual void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  // virtual void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
  //      typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  // virtual void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
  //        dealii::MeshWorker::DoFInfo<dim> &dinfo2,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

#else // MATRIXFREE OFF
template <int dim>
class PSCMatrixIntegrator final : public MatrixIntegrator<dim>
{
public:
  PSCMatrixIntegrator();
  PSCMatrixIntegrator (const PSCMatrixIntegrator &) = delete ;
  PSCMatrixIntegrator &operator = (const PSCMatrixIntegrator &) = delete;
  // virtual void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  // virtual void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
  //      typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  // virtual void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
  //        dealii::MeshWorker::DoFInfo<dim> &dinfo2,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
  //        typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};
#endif // MATRIXFREE
#endif // PSCINTEGRATORS
