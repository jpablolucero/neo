#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

#include <EquationData.h>

namespace LocalIntegrators
{
  namespace Diffusion
  {
    template <int dim, typename CoefficientTYPE>
    inline void
    cell_matrix(dealii::FullMatrix<double> &M,
		 const dealii::FEValuesBase<dim> &fe,
		 const double factor = 1.)
    {
      const CoefficientTYPE scalar_coeff;
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_components = fe.get_fe().n_components();
      
      for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
	{
	  const double dx = fe.JxW(q) * factor;
          const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    {
	      double Mii = 0.0;
	      for (unsigned int d=0; d<n_components; ++d)
		Mii += dx * scalar_coeff.value(quad_point) *
		  (fe.shape_grad_component(i,q,d) * fe.shape_grad_component(i,q,d));
	      
	      M(i,i) += Mii;
	      
	      for (unsigned int j=i+1; j<n_dofs; ++j)
		{
		  double Mij = 0.0;
		  for (unsigned int d=0; d<n_components; ++d)
		    Mij += dx * scalar_coeff.value(quad_point) *
		      (fe.shape_grad_component(j,q,d) * fe.shape_grad_component(i,q,d));
		  
		  M(i,j) += Mij;
		  M(j,i) += Mij;
		}
	    }
	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    ip_matrix(dealii::FullMatrix<double> &M11,
	      dealii::FullMatrix<double> &M12,
	      dealii::FullMatrix<double> &M21,
	      dealii::FullMatrix<double> &M22,
	      const dealii::FEValuesBase<dim> &fe1,
	      const dealii::FEValuesBase<dim> &fe2,
	      double penalty,
	      double factor1 = 1.,
	      double factor2 = -1.)
    {
       CoefficientTYPE scalar_coeff;
       const unsigned int n_dofs = fe1.dofs_per_cell;
       AssertDimension(M11.n(), n_dofs);
       AssertDimension(M11.m(), n_dofs);
       AssertDimension(M12.n(), n_dofs);
       AssertDimension(M12.m(), n_dofs);
       AssertDimension(M21.n(), n_dofs);
       AssertDimension(M21.m(), n_dofs);
       AssertDimension(M22.n(), n_dofs);
       AssertDimension(M22.m(), n_dofs);
       
       const double nui = factor1;
       const double nue = (factor2 < 0) ? factor1 : factor2;
       const double nu = .5*(nui+nue);
       
       for (unsigned int k=0; k<fe1.n_quadrature_points; ++k)
         {
	   const dealii::Point<dim> quad_point = fe1.quadrature_point(k);
	   const double dx = fe1.JxW(k) * scalar_coeff.value(quad_point);
	   const dealii::Tensor<1,dim> n = fe1.normal_vector(k);
           for (unsigned int d=0; d<fe1.get_fe().n_components(); ++d)
             {
               for (unsigned int i=0; i<n_dofs; ++i)
                 {
                   for (unsigned int j=0; j<n_dofs; ++j)
                     {
                       const double vi = fe1.shape_value_component(i,k,d);
                       const double dnvi = n * fe1.shape_grad_component(i,k,d);
                       const double ve = fe2.shape_value_component(i,k,d);
                       const double dnve = n * fe2.shape_grad_component(i,k,d);
                       const double ui = fe1.shape_value_component(j,k,d);
                       const double dnui = n * fe1.shape_grad_component(j,k,d);
                       const double ue = fe2.shape_value_component(j,k,d);
                       const double dnue = n * fe2.shape_grad_component(j,k,d);
                       M11(i,j) += dx*(-.5*nui*dnvi*ui-.5*nui*dnui*vi+nu*penalty*ui*vi);
                       M12(i,j) += dx*( .5*nui*dnvi*ue-.5*nue*dnue*vi-nu*penalty*vi*ue);
                       M21(i,j) += dx*(-.5*nue*dnve*ui+.5*nui*dnui*ve-nu*penalty*ui*ve);
                       M22(i,j) += dx*( .5*nue*dnve*ue+.5*nue*dnue*ve+nu*penalty*ue*ve);		    }
		}
	    }
	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    nitsche_matrix (dealii::FullMatrix<double> &M,
		    const dealii::FEValuesBase<dim> &fe,
		    double penalty,
		    double factor = 1.)
    {
      const CoefficientTYPE scalar_coeff;
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comp = fe.get_fe().n_components();
      
      Assert (M.m() == n_dofs, dealii::ExcDimensionMismatch(M.m(), n_dofs));
      Assert (M.n() == n_dofs, dealii::ExcDimensionMismatch(M.n(), n_dofs));
 
      for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
	{
	  const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	  const double dx = fe.JxW(q) * scalar_coeff.value(quad_point) * factor;
	  const dealii::Tensor<1,dim> n = fe.normal_vector(q);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    for (unsigned int j=0; j<n_dofs; ++j)
	      for (unsigned int d=0; d<n_comp; ++d)
		M(i,j) += dx * (2.*penalty * fe.shape_value_component(i,q,d) * fe.shape_value_component(j,q,d)
				- (n * fe.shape_grad_component(i,q,d)) * fe.shape_value_component(j,q,d)
				- (n * fe.shape_grad_component(j,q,d)) * fe.shape_value_component(i,q,d));
	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    cell_residual(dealii::Vector<double> &result,
		  const dealii::FEValuesBase<dim> &fe,
		  const std::vector<dealii::Tensor<1,dim> > &input,
		  double factor = 1.) 
    {
      const CoefficientTYPE scalar_coeff;
      const unsigned int n_quads = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;

      Assert(input.size() == n_quads, dealii::ExcDimensionMismatch(input.size(), n_quads));
      Assert(result.size() == n_dofs, dealii::ExcDimensionMismatch(result.size(), n_dofs));

      for (unsigned int q=0; q<n_quads; ++q)
	{
          const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	  const double dx = factor * fe.JxW(q);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    result(i) += dx * (input[q] * fe.shape_grad(i,q)) * scalar_coeff.value(quad_point) ;
	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    ip_residual(dealii::Vector<double> &resultINT,
    		     dealii::Vector<double> &resultEXT,
    		     const dealii::FEValuesBase<dim> &feINT,
    		     const dealii::FEValuesBase<dim> &feEXT,
    		     const std::vector<double> &inputINT,
    		     const std::vector<dealii::Tensor<1,dim> > &DinputINT,
    		     const std::vector<double> &inputEXT,
    		     const std::vector<dealii::Tensor<1,dim> > &DinputEXT,
		     double penalty,
    		     double int_factor = 1.,
    		     double ext_factor = -1.)
    {
      Assert(feINT.get_fe().n_components() == 1,
    	     dealii::ExcDimensionMismatch(feINT.get_fe().n_components(), 1));
      Assert(feEXT.get_fe().n_components() == 1,
    	     dealii::ExcDimensionMismatch(feEXT.get_fe().n_components(), 1));
 
      const CoefficientTYPE scalar_coeff;
      const double nuINT = int_factor;
      const double nuEXT = (ext_factor < 0) ? int_factor : ext_factor;
      const double nupenalty = .5 * penalty  * (nuINT + nuEXT);
      const unsigned int n_dofs = feINT.dofs_per_cell;
 
      for (unsigned int q=0; q<feINT.n_quadrature_points; ++q)
    	{
          const dealii::Point<dim> quad_point = feINT.quadrature_point(q);
    	  const double dx = feINT.JxW(q) * scalar_coeff.value(quad_point);
    	  const dealii::Tensor<1,dim> normal_vectorINT = feINT.normal_vector(q);
 
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
 
    	      resultINT(i) += dx*( nupenalty*uINT*vINT  - .5*(nuINT*uINT*dnvINT + nuINT*dnuINT*vINT) );
    	      resultINT(i) += dx*( -nupenalty*uEXT*vINT + .5*(nuINT*uEXT*dnvINT - nuEXT*dnuEXT*vINT) );
    	      resultEXT(i) += dx*( -nupenalty*uINT*vEXT - .5*(nuEXT*uINT*dnvEXT - nuINT*dnuINT*vEXT) );
	      resultEXT(i) += dx*( nupenalty*uEXT*vEXT  + .5*(nuEXT*uEXT*dnvEXT + nuEXT*dnuEXT*vEXT) );
    	    }
    	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    nitsche_residual(dealii::Vector<double> &result,
		     const dealii::FEValuesBase<dim> &fe,
		     const std::vector<double> &input,
		     const std::vector<dealii::Tensor<1,dim> > &Dinput,
		     const std::vector<double> &data,
		     double penalty,
		     double factor = 1.)
     {
       const CoefficientTYPE scalar_coeff;
       const unsigned int n_dofs = fe.dofs_per_cell;
      
       AssertDimension(input.size(), fe.n_quadrature_points);
       AssertDimension(Dinput.size(), fe.n_quadrature_points);
       AssertDimension(data.size(), fe.n_quadrature_points);
 
       for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
         {
	   const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	   const double dx = factor * fe.JxW(q);
	   const dealii::Tensor<1,dim> n = fe.normal_vector(q);
	   for (unsigned int i=0; i<n_dofs; ++i)
             {
	       const double u= input[q];
               const double g= data[q];
               const double dnv = fe.shape_grad(i,q) * n;
               const double dnu = Dinput[q] * n;
               const double v= fe.shape_value(i,q);
 
               result(i) += dx*scalar_coeff.value(quad_point)*(2.*penalty*(u-g)*v - (u-g)*dnv - dnu*v);
             }
         }
     }
  } // end NAMESPACE = Diffusion
} // end NAMESPACE = LocalIntegrators

#endif // DIFFUSION_H
