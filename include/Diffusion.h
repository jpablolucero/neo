#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

namespace LocalIntegrators
{
  namespace Diffusion
  {
    template <int dim>
    inline void
    cell_residual(dealii::Vector<double> &result,
		  const dealii::FEValuesBase<dim> &fe,
		  const std::vector<dealii::Tensor<1,dim> > &input,
		  const dealii::Function<dim> &scalar_coeff,
		  double factor = 1.) 
    {
      const unsigned int n_quads = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;
      Assert(input.size() == n_quads, dealii::ExcDimensionMismatch(input.size(), n_quads));
      Assert(result.size() == n_dofs, dealii::ExcDimensionMismatch(result.size(), n_dofs));

      for (unsigned int q=0; q<n_quads; ++q)
	{
	  const double dx = factor * fe.JxW(q);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    result(i) += dx * (input[q] * fe.shape_grad(i,q)) * scalar_coeff.value(fe.quadrature_point(q)) ;
	}
    }


    template<int dim>
    void ip_residual(dealii::Vector<double> &resultINT,
    		     dealii::Vector<double> &resultEXT,
    		     const dealii::FEValuesBase<dim> &feINT,
    		     const dealii::FEValuesBase<dim> &feEXT,
    		     const std::vector<double> &inputINT,
    		     const std::vector<dealii::Tensor<1,dim> > &DinputINT,
    		     const std::vector<double> &inputEXT,
    		     const std::vector<dealii::Tensor<1,dim> > &DinputEXT,
    		     const dealii::Function<dim> &scalar_coeff,
    		     double pen,
    		     double int_factor = 1.,
    		     double ext_factor = -1.)
    {
      Assert(feINT.get_fe().n_components() == 1,
    	     dealii::ExcDimensionMismatch(feINT.get_fe().n_components(), 1));
      Assert(feEXT.get_fe().n_components() == 1,
    	     dealii::ExcDimensionMismatch(feEXT.get_fe().n_components(), 1));
 
      const double nuINT = int_factor;
      const double nuEXT = (ext_factor < 0) ? int_factor : ext_factor;
      const double penalty = .5 * pen  * (nuINT + nuEXT);
 
      const unsigned int n_dofs = feINT.dofs_per_cell;
 
      for (unsigned int q=0; q<feINT.n_quadrature_points; ++q)
    	{
    	  const double dx = feINT.JxW(q);
    	  const dealii::Tensor<1,dim> normal_vectorINT = feINT.normal_vector(q);
          const dealii::Point<dim> quad_point = feINT.quadrature_point(q);
 
    	  for (unsigned int i=0; i<n_dofs; ++i)
    	    {
    	      const double vINT = feINT.shape_value(i,q);
    	      const dealii::Tensor<1,dim> &DvINT = feINT.shape_grad(i,q);
    	      const double dnvINT = DvINT * normal_vectorINT;
    	      const double vEXT = feEXT.shape_value(i,q);
    	      const dealii::Tensor<1,dim> &DvEXT = feEXT.shape_grad(i,q);
    	      const double dnvEXT = DvEXT * normal_vectorINT;
 
    	      const double uINT = inputINT[q];
    	      const dealii::Tensor<1,dim> &DuINT = DinputINT[q];
    	      const double dnuINT = DuINT * normal_vectorINT;
    	      const double uEXT = inputEXT[q];
    	      const dealii::Tensor<1,dim> &DuEXT = DinputEXT[q];
    	      const double dnuEXT = DuEXT * normal_vectorINT;
 
    	      resultINT(i) += dx*scalar_coeff.value(quad_point)*(penalty*uINT*vINT
								 -.5*(nuINT*uINT*dnvINT+nuINT*dnuINT*vINT) );
    	      resultINT(i) += dx*scalar_coeff.value(quad_point)*(-penalty*uEXT*vINT 
								 +.5*(nuINT*uEXT*dnvINT-nuEXT*dnuEXT*vINT) );
    	      resultEXT(i) += dx*scalar_coeff.value(quad_point)*(-penalty*uINT*vEXT
								 -.5*(nuEXT*uINT*dnvEXT-nuINT*dnuINT*vEXT) );
	      resultEXT(i) += dx*scalar_coeff.value(quad_point)*(penalty*uEXT*vEXT
								 +.5*(nuEXT*uEXT*dnvEXT+nuEXT*dnuEXT*vEXT) );
    	    }
    	}
    }

  // dealii::LocalIntegrators::Laplace::nitsche_residual(dst,
    // 						      fe,
    // 						      src,
    // 						      Dsrc,
    // 						      data,
    // 						      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg),
    // 						      0.27);


  }
}

#endif // DIFFUSION_H
