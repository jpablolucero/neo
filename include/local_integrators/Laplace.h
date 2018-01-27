#ifndef LAPLACE_H
#define LAPLACE_H

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

namespace LocalIntegrators
{
  namespace Laplace
  {
    template <int dim>
    void nitsche_residual_data_only (
	  dealii::Vector<double> &result,
	  const dealii::FEValuesBase<dim> &fe,
	  const dealii::VectorSlice<const std::vector<std::vector<double> > > &data,
	  double penalty,
	  double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comp = fe.get_fe().n_components();

      AssertVectorVectorDimension(data, n_comp, fe.n_quadrature_points);

      for (unsigned int k=0; k<fe.n_quadrature_points; ++k)
	{
	  const double dx = factor * fe.JxW(k);
	  const dealii::Tensor<1,dim> n = fe.normal_vector(k);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    for (unsigned int d=0; d<n_comp; ++d)
	      {
		const double dnv = fe.shape_grad_component(i,k,d) * n;
		const double v= fe.shape_value_component(i,k,d);
		const double g= data[d][k];

		result(i) += dx*(2.*penalty*g*v - dnv*g);
	      }
	}
    }

  } // namespace Laplace
}   // namespace LocalIntegrators
#endif	// LALPACE_H
