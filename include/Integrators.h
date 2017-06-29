#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <deal.II/base/vector_slice.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

#include <deal.II/lac/matrix_block.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>

#include <Diffusion.h>
#include <Transport.h>
#include <EquationData.h>
#include <GenericLinearAlgebra.h>



#ifndef MATRIXFREE
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

// template <int dim>
// class RHSIntegrator final : public dealii::MeshWorker::LocalIntegrator<dim>
// {
// public:
//   RHSIntegrator(unsigned int n_components);
//   RHSIntegrator (const RHSIntegrator &) = delete ;
//   RHSIntegrator &operator = (const RHSIntegrator &) = delete;
//   void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
//   void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
// private:
//   Coefficient<dim> diffcoeff;
//   Solution<dim> exact_solution;
// };

#else // MATRIXFREE ON
template <int dim>
class MatrixIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  MatrixIntegrator () : dealii::MeshWorker::LocalIntegrator<dim>::LocalIntegrator() {};
  MatrixIntegrator (const MatrixIntegrator &) = delete ;
  MatrixIntegrator &operator = (const MatrixIntegrator &) = delete;
  void cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const override;
  void face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
            dealii::MeshWorker::DoFInfo<dim> &dinfo2,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
            typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const override;
};

template <int dim, int fe_degree, int n_q_points_1d = fe_degree+1,
          int n_comp = 1, typename number = double >
class MFIntegrator final
{
public:
  MFIntegrator ();
  MFIntegrator (const MFIntegrator &) = delete ;
  MFIntegrator &operator= (const MFIntegrator &) = delete;
  void cell(const dealii::MatrixFree<dim,number>       &data,
            LA::MPI::Vector                            &dst,
            const LA::MPI::Vector                      &src,
            const std::pair<unsigned int,unsigned int> &cell_range) const;
  void boundary(const dealii::MatrixFree<dim,number>       &data,
                LA::MPI::Vector                            &dst,
                const LA::MPI::Vector                      &src,
                const std::pair<unsigned int,unsigned int> &face_range) const;
  void face(const dealii::MatrixFree<dim,number>       &data,
            LA::MPI::Vector                            &dst,
            const LA::MPI::Vector                      &src,
            const std::pair<unsigned int,unsigned int> &face_range) const;
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
private:
  MFRightHandSide<dim> ref_rhs;
  MFSolution<dim> ref_solution;
};
#endif // MATRIXFREE ON

#ifdef HEADER_IMPLEMENTATION
#include <Integrators.cc>
#endif

#endif // INTEGRATORS_H
