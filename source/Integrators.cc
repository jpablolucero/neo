#include <Integrators.h>
#include <deal.II/integrators/elasticity.h>
#include <deal.II/integrators/grad_div.h>
#include <deal.II/integrators/laplace.h>

template <int dim>
MatrixIntegrator<dim>::MatrixIntegrator()
{}

template <int dim>
void MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  AssertDimension(dinfo.n_matrices(), 1);

  dealii::LocalIntegrators::Elasticity::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), 2. * material_param.mu);
  dealii::LocalIntegrators::GradDiv::cell_matrix(
      dinfo.matrix(0, false).matrix, info.fe_values(0), material_param.lambda);
}

template <int dim>
void MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                 dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  dealii::LocalIntegrators::Elasticity::ip_matrix(
    dinfo1.matrix(0, false).matrix,
    dinfo1.matrix(0, true).matrix,
    dinfo2.matrix(0, true).matrix,
    dinfo2.matrix(0, false).matrix,
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * material_param.mu);
}

template <int dim>
void MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                     typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  if (dinfo.face->boundary_id() == 0 || dinfo.face->boundary_id() == 1)
  {
    dealii::LocalIntegrators::Elasticity::nitsche_matrix(
      dinfo.matrix(0, false).matrix,
      info.fe_values(0),
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * material_param.mu);
  }
}

template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator()
{}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  Assert(info.values.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));
  Assert(info.gradients.size() >= 1, dealii::ExcDimensionMismatch(info.values.size(), 1));

  dealii::LocalIntegrators::Elasticity::cell_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.gradients[0], 0, dim),
      2. * material_param.mu);
  dealii::LocalIntegrators::GradDiv::cell_residual(dinfo.vector(0).block(0),
						   info.fe_values(0),
						   dealii::make_slice(info.gradients[0], 0, dim),
						   material_param.lambda);
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const unsigned int deg = info1.fe_values(0).get_fe().tensor_degree();
  dealii::LocalIntegrators::Elasticity::ip_residual(
    dinfo1.vector(0).block(0),
    dinfo2.vector(0).block(0),
    info1.fe_values(0),
    info2.fe_values(0),
    dealii::make_slice(info1.values[0], 0, dim),
    dealii::make_slice(info1.gradients[0], 0, dim),
    dealii::make_slice(info2.values[0], 0, dim),
    dealii::make_slice(info2.gradients[0], 0, dim),
    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1, dinfo2, deg, deg),
    2. * material_param.mu);
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  std::vector<std::vector<double>> null(
    dim, std::vector<double>(info.fe_values(0).n_quadrature_points, 0.));

  boundary_values->vector_values(info.fe_values(0).get_quadrature_points(), null);

  if (dinfo.face->boundary_id() == 0 || dinfo.face->boundary_id() == 1)
  {
    const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
    dealii::LocalIntegrators::Elasticity::nitsche_residual(
      dinfo.vector(0).block(0),
      info.fe_values(0),
      dealii::make_slice(info.values[0], 0, dim),
      dealii::make_slice(info.gradients[0], 0, dim),
      null,
      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo, dinfo, deg, deg),
      2. * material_param.mu);
  }
}

// RHS INTEGRATOR
template <int dim>
RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components)
  : exact_solution(n_components)
{
  this->use_cell = true;
#ifdef CG
  this->use_boundary = false;
#else
  this->use_boundary = true;
#endif
  this->use_face = true;
}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
			      dealii::MeshWorker::DoFInfo<dim> &,
			      typename dealii::MeshWorker::IntegrationInfo<dim> &,
			      typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

#ifndef HEADER_IMPLEMENTATION
#include "Integrators.inst"
#endif

