#ifndef ELASTICITY_H
#define ELASTICITY_H

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

namespace LocalIntegrators
{
  namespace Elasticity
  {
    template <int dim>
    inline void cell_matrix (
      dealii::FullMatrix<double> &M,
      const dealii::FEValuesBase<dim> &fe,
      const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertDimension(fe.get_fe().n_components(), dim);
      AssertDimension(M.m(), n_dofs);
      AssertDimension(M.n(), n_dofs);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d1=0; d1<dim; ++d1)
                for (unsigned int d2=0; d2<dim; ++d2)
                  M(i,j) += dx * .25 *
                            (fe.shape_grad_component(j,k,d1)[d2] + fe.shape_grad_component(j,k,d2)[d1]) *
                            (fe.shape_grad_component(i,k,d1)[d2] + fe.shape_grad_component(i,k,d2)[d1]);
        }
    }


    template <int dim, typename number>
    inline void
    cell_residual  (
      dealii::Vector<number> &result,
      const dealii::FEValuesBase<dim> &fe,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &input,
      double factor = 1.)
    {
      const unsigned int nq = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertDimension(fe.get_fe().n_components(), dim);
      AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      Assert(result.size() == n_dofs, dealii::ExcDimensionMismatch(result.size(), n_dofs));

      for (unsigned int k=0; k<nq; ++k)
        {
          const double dx = factor * fe.JxW(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d1=0; d1<dim; ++d1)
              for (unsigned int d2=0; d2<dim; ++d2)
                {
                  result(i) += dx * .25 *
                               (input[d1][k][d2] + input[d2][k][d1]) *
                               (fe.shape_grad_component(i,k,d1)[d2] + fe.shape_grad_component(i,k,d2)[d1]);
                }
        }
    }


    template <int dim>
    inline void nitsche_matrix (
      dealii::FullMatrix<double> &M,
      const dealii::FEValuesBase<dim> &fe,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertDimension(fe.get_fe().n_components(), dim);
      AssertDimension(M.m(), n_dofs);
      AssertDimension(M.n(), n_dofs);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d1=0; d1<dim; ++d1)
                {
                  const double u = fe.shape_value_component(j,k,d1);
                  const double v = fe.shape_value_component(i,k,d1);
                  M(i,j) += dx * 2. * penalty * u * v;
                  for (unsigned int d2=0; d2<dim; ++d2)
                    {
                      // v . nabla u n
                      M(i,j) -= .5*dx* fe.shape_grad_component(j,k,d1)[d2] *n[d2]* v;
                      // v (nabla u)^T n
                      M(i,j) -= .5*dx* fe.shape_grad_component(j,k,d2)[d1] *n[d2]* v;
                      // u  nabla v n
                      M(i,j) -= .5*dx* fe.shape_grad_component(i,k,d1)[d2] *n[d2]* u;
                      // u (nabla v)^T n
                      M(i,j) -= .5*dx* fe.shape_grad_component(i,k,d2)[d1] *n[d2]* u;
                    }
                }
        }
    }


    template <int dim>
    inline void nitsche_tangential_matrix (
      dealii::FullMatrix<double> &M,
      const dealii::FEValuesBase<dim> &fe,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertDimension(fe.get_fe().n_components(), dim);
      AssertDimension(M.m(), n_dofs);
      AssertDimension(M.n(), n_dofs);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              {
                double udotn = 0.;
                double vdotn = 0.;
                double ngradun = 0.;
                double ngradvn = 0.;

                for (unsigned int d=0; d<dim; ++d)
                  {
                    udotn += n[d]*fe.shape_value_component(j,k,d);
                    vdotn += n[d]*fe.shape_value_component(i,k,d);
                    ngradun += n*fe.shape_grad_component(j,k,d)*n[d];
                    ngradvn += n*fe.shape_grad_component(i,k,d)*n[d];
                  }
                for (unsigned int d1=0; d1<dim; ++d1)
                  {
                    const double u = fe.shape_value_component(j,k,d1) - udotn * n[d1];
                    const double v = fe.shape_value_component(i,k,d1) - vdotn * n[d1];
                    M(i,j) += dx * 2. * penalty * u * v;
                    // Correct the gradients below and subtract normal component
                    M(i,j) += dx * (ngradun * v + ngradvn * u);
                    for (unsigned int d2=0; d2<dim; ++d2)
                      {
                        // v . nabla u n
                        M(i,j) -= .5*dx* fe.shape_grad_component(j,k,d1)[d2] *n[d2]* v;
                        // v (nabla u)^T n
                        M(i,j) -= .5*dx* fe.shape_grad_component(j,k,d2)[d1] *n[d2]* v;
                        // u  nabla v n
                        M(i,j) -= .5*dx* fe.shape_grad_component(i,k,d1)[d2] *n[d2]* u;
                        // u (nabla v)^T n
                        M(i,j) -= .5*dx* fe.shape_grad_component(i,k,d2)[d1] *n[d2]* u;
                      }
                  }
              }
        }
    }


    template <int dim, typename number>
    void nitsche_residual (
      dealii::Vector<number> &result,
      const dealii::FEValuesBase<dim> &fe,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &data,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      AssertVectorVectorDimension(Dinput, dim, fe.n_quadrature_points);
      AssertVectorVectorDimension(data, dim, fe.n_quadrature_points);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d1=0; d1<dim; ++d1)
              {
                const double u= input[d1][k];
                const double v= fe.shape_value_component(i,k,d1);
                const double g= data[d1][k];
                result(i) += dx * 2.*penalty * (u-g) * v;

                for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    // v . nabla u n
                    result(i) -= .5*dx* v * Dinput[d1][k][d2] * n[d2];
                    // v . (nabla u)^T n
                    result(i) -= .5*dx* v * Dinput[d2][k][d1] * n[d2];
                    // u  nabla v n
                    result(i) -= .5*dx * (u-g) * fe.shape_grad_component(i,k,d1)[d2] * n[d2];
                    // u  (nabla v)^T n
                    result(i) -= .5*dx * (u-g) * fe.shape_grad_component(i,k,d2)[d1] * n[d2];
                  }
              }
        }
    }


    template <int dim, typename number>
    void nitsche_residual_data_only (
      dealii::Vector<number> &result,
      const dealii::FEValuesBase<dim> &fe,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &data,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertVectorVectorDimension(data, dim, fe.n_quadrature_points);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d1=0; d1<dim; ++d1)
              {
                const double v= fe.shape_value_component(i,k,d1);
                const double g= data[d1][k];
                result(i) += dx * 2.*penalty * g * v;

                for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    // u  nabla v n
                    result(i) -= .5*dx * g * fe.shape_grad_component(i,k,d1)[d2] * n[d2];
                    // u  (nabla v)^T n
                    result(i) -= .5*dx * g * fe.shape_grad_component(i,k,d2)[d1] * n[d2];
                  }
              }
        }
    }


    template <int dim, typename number>
    inline void nitsche_tangential_residual (
      dealii::Vector<number> &result,
      const dealii::FEValuesBase<dim> &fe,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &data,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      AssertVectorVectorDimension(Dinput, dim, fe.n_quadrature_points);
      AssertVectorVectorDimension(data, dim, fe.n_quadrature_points);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            {
              double udotn = 0.;
              double gdotn = 0.;
              double vdotn = 0.;
              double ngradun = 0.;
              double ngradvn = 0.;

              for (unsigned int d=0; d<dim; ++d)
                {
                  udotn += n[d]*input[d][k];
                  gdotn += n[d]*data[d][k];
                  vdotn += n[d]*fe.shape_value_component(i,k,d);
                  ngradun += n*Dinput[d][k]*n[d];
                  ngradvn += n*fe.shape_grad_component(i,k,d)*n[d];
                }
              for (unsigned int d1=0; d1<dim; ++d1)
                {
                  const double u= input[d1][k] - udotn*n[d1];
                  const double v= fe.shape_value_component(i,k,d1) - vdotn*n[d1];
                  const double g= data[d1][k] - gdotn*n[d1];
                  result(i) += dx * 2.*penalty * (u-g) * v;
                  // Correct the gradients below and subtract normal component
                  result(i) += dx * (ngradun * v + ngradvn * (u-g));
                  for (unsigned int d2=0; d2<dim; ++d2)
                    {
                      // v . nabla u n
                      result(i) -= .5*dx* Dinput[d1][k][d2] *n[d2]* v;
                      // v (nabla u)^T n
                      result(i) -= .5*dx* Dinput[d2][k][d1] *n[d2]* v;
                      // u  nabla v n
                      result(i) -= .5*dx* (u-g) * fe.shape_grad_component(i,k,d1)[d2] *n[d2];
                      // u (nabla v)^T n
                      result(i) -= .5*dx* (u-g) * fe.shape_grad_component(i,k,d2)[d1] *n[d2];
                    }
                }
            }
        }
    }


    template <int dim, typename number>
    void nitsche_residual_homogeneous (
      dealii::Vector<number> &result,
      const dealii::FEValuesBase<dim> &fe,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput,
      double penalty,
      double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;

      AssertVectorVectorDimension(input, dim, fe.n_quadrature_points);
      AssertVectorVectorDimension(Dinput, dim, fe.n_quadrature_points);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
        {
          const double dx = factor * fe.JxW(k);
          const dealii::Tensor<1,dim> n = fe.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d1=0; d1<dim; ++d1)
              {
                const double u= input[d1][k];
                const double v= fe.shape_value_component(i,k,d1);
                result(i) += dx * 2.*penalty * u * v;

                for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    // v . nabla u n
                    result(i) -= .5*dx* v * Dinput[d1][k][d2] * n[d2];
                    // v . (nabla u)^T n
                    result(i) -= .5*dx* v * Dinput[d2][k][d1] * n[d2];
                    // u  nabla v n
                    result(i) -= .5*dx * u * fe.shape_grad_component(i,k,d1)[d2] * n[d2];
                    // u  (nabla v)^T n
                    result(i) -= .5*dx * u * fe.shape_grad_component(i,k,d2)[d1] * n[d2];
                  }
              }
        }
    }


    template <int dim>
    inline void ip_matrix (
      dealii::FullMatrix<double> &M11,
      dealii::FullMatrix<double> &M12,
      dealii::FullMatrix<double> &M21,
      dealii::FullMatrix<double> &M22,
      const dealii::FEValuesBase<dim> &fe1,
      const dealii::FEValuesBase<dim> &fe2,
      const double pen,
      const double int_factor = 1.,
      const double ext_factor = -1.)
    {
      const unsigned int n_dofs = fe1.dofs_per_cell;

      AssertDimension(fe1.get_fe().n_components(), dim);
      AssertDimension(fe2.get_fe().n_components(), dim);
      AssertDimension(M11.m(), n_dofs);
      AssertDimension(M11.n(), n_dofs);
      AssertDimension(M12.m(), n_dofs);
      AssertDimension(M12.n(), n_dofs);
      AssertDimension(M21.m(), n_dofs);
      AssertDimension(M21.n(), n_dofs);
      AssertDimension(M22.m(), n_dofs);
      AssertDimension(M22.n(), n_dofs);

      const double nu1 = int_factor;
      const double nu2 = (ext_factor < 0) ? int_factor : ext_factor;
      const double penalty = .5 * pen * (nu1 + nu2);

      for (unsigned int k=0; k<fe1.n_quadrature_points; ++k)
        {
          const double dx = fe1.JxW(k);
          const dealii::Tensor<1,dim> n = fe1.normal_vector(k);
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d1=0; d1<dim; ++d1)
                {
                  const double u1 = fe1.shape_value_component(j,k,d1);
                  const double u2 = fe2.shape_value_component(j,k,d1);
                  const double v1 = fe1.shape_value_component(i,k,d1);
                  const double v2 = fe2.shape_value_component(i,k,d1);

                  M11(i,j) += dx * penalty * u1*v1;
                  M12(i,j) -= dx * penalty * u2*v1;
                  M21(i,j) -= dx * penalty * u1*v2;
                  M22(i,j) += dx * penalty * u2*v2;

                  for (unsigned int d2=0; d2<dim; ++d2)
                    {
                      // v . nabla u n
                      M11(i,j) -= .25 * dx * nu1 * fe1.shape_grad_component(j,k,d1)[d2] * n[d2] * v1;
                      M12(i,j) -= .25 * dx * nu2 * fe2.shape_grad_component(j,k,d1)[d2] * n[d2] * v1;
                      M21(i,j) += .25 * dx * nu1 * fe1.shape_grad_component(j,k,d1)[d2] * n[d2] * v2;
                      M22(i,j) += .25 * dx * nu2 * fe2.shape_grad_component(j,k,d1)[d2] * n[d2] * v2;
                      // v (nabla u)^T n
                      M11(i,j) -= .25 * dx * nu1 * fe1.shape_grad_component(j,k,d2)[d1] * n[d2] * v1;
                      M12(i,j) -= .25 * dx * nu2 * fe2.shape_grad_component(j,k,d2)[d1] * n[d2] * v1;
                      M21(i,j) += .25 * dx * nu1 * fe1.shape_grad_component(j,k,d2)[d1] * n[d2] * v2;
                      M22(i,j) += .25 * dx * nu2 * fe2.shape_grad_component(j,k,d2)[d1] * n[d2] * v2;
                      // u  nabla v n
                      M11(i,j) -= .25 * dx * nu1 * fe1.shape_grad_component(i,k,d1)[d2] * n[d2] * u1;
                      M12(i,j) += .25 * dx * nu1 * fe1.shape_grad_component(i,k,d1)[d2] * n[d2] * u2;
                      M21(i,j) -= .25 * dx * nu2 * fe2.shape_grad_component(i,k,d1)[d2] * n[d2] * u1;
                      M22(i,j) += .25 * dx * nu2 * fe2.shape_grad_component(i,k,d1)[d2] * n[d2] * u2;
                      // u (nabla v)^T n
                      M11(i,j) -= .25 * dx * nu1 * fe1.shape_grad_component(i,k,d2)[d1] * n[d2] * u1;
                      M12(i,j) += .25 * dx * nu1 * fe1.shape_grad_component(i,k,d2)[d1] * n[d2] * u2;
                      M21(i,j) -= .25 * dx * nu2 * fe2.shape_grad_component(i,k,d2)[d1] * n[d2] * u1;
                      M22(i,j) += .25 * dx * nu2 * fe2.shape_grad_component(i,k,d2)[d1] * n[d2] * u2;
                    }
                }
        }
    }


    template <int dim, typename number>
    void
    ip_residual(
      dealii::Vector<number> &result1,
      dealii::Vector<number> &result2,
      const dealii::FEValuesBase<dim> &fe1,
      const dealii::FEValuesBase<dim> &fe2,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input1,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput1,
      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input2,
      const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput2,
      double pen,
      double int_factor = 1.,
      double ext_factor = -1.)
    {
      const unsigned int n1 = fe1.dofs_per_cell;

      AssertDimension(fe1.get_fe().n_components(), dim);
      AssertDimension(fe2.get_fe().n_components(), dim);
      AssertVectorVectorDimension(input1, dim, fe1.n_quadrature_points);
      AssertVectorVectorDimension(Dinput1, dim, fe1.n_quadrature_points);
      AssertVectorVectorDimension(input2, dim, fe2.n_quadrature_points);
      AssertVectorVectorDimension(Dinput2, dim, fe2.n_quadrature_points);

      const double nu1 = int_factor;
      const double nu2 = (ext_factor < 0) ? int_factor : ext_factor;
      const double penalty = .5 * pen * (nu1 + nu2);

      for (unsigned int k=0; k<fe1.n_quadrature_points; ++k)
        {
          const double dx = fe1.JxW(k);
          const dealii::Tensor<1,dim> n = fe1.normal_vector(k);

          for (unsigned int i=0; i<n1; ++i)
            for (unsigned int d1=0; d1<dim; ++d1)
              {
                const double v1 = fe1.shape_value_component(i,k,d1);
                const double v2 = fe2.shape_value_component(i,k,d1);
                const double u1 = input1[d1][k];
                const double u2 = input2[d1][k];

                result1(i) += dx * penalty * u1*v1;
                result1(i) -= dx * penalty * u2*v1;
                result2(i) -= dx * penalty * u1*v2;
                result2(i) += dx * penalty * u2*v2;

                for (unsigned int d2=0; d2<dim; ++d2)
                  {
                    // v . nabla u n
                    result1(i) -= .25*dx* (nu1*Dinput1[d1][k][d2]+nu2*Dinput2[d1][k][d2]) * n[d2] * v1;
                    result2(i) += .25*dx* (nu1*Dinput1[d1][k][d2]+nu2*Dinput2[d1][k][d2]) * n[d2] * v2;
                    // v . (nabla u)^T n
                    result1(i) -= .25*dx* (nu1*Dinput1[d2][k][d1]+nu2*Dinput2[d2][k][d1]) * n[d2] * v1;
                    result2(i) += .25*dx* (nu1*Dinput1[d2][k][d1]+nu2*Dinput2[d2][k][d1]) * n[d2] * v2;
                    // u  nabla v n
                    result1(i) -= .25*dx* nu1*fe1.shape_grad_component(i,k,d1)[d2] * n[d2] * (u1-u2);
                    result2(i) -= .25*dx* nu2*fe2.shape_grad_component(i,k,d1)[d2] * n[d2] * (u1-u2);
                    // u  (nabla v)^T n
                    result1(i) -= .25*dx* nu1*fe1.shape_grad_component(i,k,d2)[d1] * n[d2] * (u1-u2);
                    result2(i) -= .25*dx* nu2*fe2.shape_grad_component(i,k,d2)[d1] * n[d2] * (u1-u2);
                  }
              }
        }
    }

  } // end NAMESPACE = Elasticity
} // end NAMESPACE = LocalIntegrators
#endif // ELASTICITY_H
