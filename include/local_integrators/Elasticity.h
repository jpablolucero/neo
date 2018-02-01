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
                    // g nabla v n
                    result(i) -= .5*dx * g * fe.shape_grad_component(i,k,d1)[d2] * n[d2];
                    // g (nabla v)^T n
                    result(i) -= .5*dx * g * fe.shape_grad_component(i,k,d2)[d1] * n[d2];
                  }
              }
        }
    }
  } // end NAMESPACE = Elasticity
} // end NAMESPACE = LocalIntegrators
#endif // ELASTICITY_H
