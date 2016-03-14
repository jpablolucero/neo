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
    ip_matrix(dealii::FullMatrix<double> &M_INT_INT,
	      dealii::FullMatrix<double> &M_INT_EXT,
	      dealii::FullMatrix<double> &M_EXT_INT,
	      dealii::FullMatrix<double> &M_EXT_EXT,
	      const dealii::FEValuesBase<dim> &feINT,
	      const dealii::FEValuesBase<dim> &feEXT,
	      double penalty,
	      double int_factor = 1.,
	      double ext_factor = -1.)
    {
       CoefficientTYPE scalar_coeff;
       const unsigned int n_dofs = feINT.dofs_per_cell;
       AssertDimension(M_INT_INT.n(), n_dofs);
       AssertDimension(M_INT_INT.m(), n_dofs);
       AssertDimension(M_INT_EXT.n(), n_dofs);
       AssertDimension(M_INT_EXT.m(), n_dofs);
       AssertDimension(M_EXT_INT.n(), n_dofs);
       AssertDimension(M_EXT_INT.m(), n_dofs);
       AssertDimension(M_EXT_EXT.n(), n_dofs);
       AssertDimension(M_EXT_EXT.m(), n_dofs);
       
       const double nuINT = int_factor;
       const double nuEXT = (ext_factor < 0) ? int_factor : ext_factor;
       const double nupenalty = .5*(nuINT+nuEXT)*penalty;
       
       for (unsigned int q=0; q<feINT.n_quadrature_points; ++q)
         {
	   const dealii::Point<dim> quad_point = feINT.quadrature_point(q);
	   const double dx = feINT.JxW(q) * scalar_coeff.value(quad_point);
	   const dealii::Tensor<1,dim> n = feINT.normal_vector(q);
           for (unsigned int d=0; d<feINT.get_fe().n_components(); ++d)
             {
               for (unsigned int i=0; i<n_dofs; ++i)
                 {
                   for (unsigned int j=0; j<n_dofs; ++j)
                     {
                       const double vINT = feINT.shape_value_component(i,q,d);
                       const double dnvINT = n * feINT.shape_grad_component(i,q,d);
                       const double vEXT = feEXT.shape_value_component(i,q,d);
                       const double dnvEXT = n * feEXT.shape_grad_component(i,q,d);
                       const double uINT = feINT.shape_value_component(j,q,d);
                       const double dnuINT = n * feINT.shape_grad_component(j,q,d);
                       const double uEXT = feEXT.shape_value_component(j,q,d);
                       const double dnuEXT = n * feEXT.shape_grad_component(j,q,d);
                       M_INT_INT(i,j) += dx*(-.5*nuINT*dnvINT*uINT-.5*nuINT*dnuINT*vINT+nupenalty*uINT*vINT);
                       M_INT_EXT(i,j) += dx*( .5*nuINT*dnvINT*uEXT-.5*nuEXT*dnuEXT*vINT-nupenalty*vINT*uEXT);
                       M_EXT_INT(i,j) += dx*(-.5*nuEXT*dnvEXT*uINT+.5*nuINT*dnuINT*vEXT-nupenalty*uINT*vEXT);
                       M_EXT_EXT(i,j) += dx*( .5*nuEXT*dnvEXT*uEXT+.5*nuEXT*dnuEXT*vEXT+nupenalty*uEXT*vEXT);
		     }
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
    cell_residualvs(dealii::Vector<double> &result,
		  const dealii::FEValuesBase<dim> &fe,
		  const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &input,
		  double factor = 1.) 
    {
      const CoefficientTYPE scalar_coeff;
      const unsigned int n_quads = fe.n_quadrature_points;
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comp = fe.get_fe().n_components();

      AssertVectorVectorDimension(input, n_comp, n_quads);
      Assert(result.size() == n_dofs, dealii::ExcDimensionMismatch(result.size(), n_dofs));

      std::vector<std::vector<double> > coeffs{n_comp};
      coeffs[0].resize(n_quads);
      coeffs[1].resize(n_quads);
      scalar_coeff.value_list(fe.get_quadrature_points(), coeffs[0]);
      scalar_coeff.value_list(fe.get_quadrature_points(), coeffs[1]);

      for (unsigned int q=0; q<n_quads; ++q)
	{
          const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	  const double dx = factor * fe.JxW(q);
	  for (unsigned int i=0; i<n_dofs; ++i)
	    for(unsigned int d=0; d<n_comp; ++d)
	      result(i) += dx * (input[d][q] * fe.shape_grad_component(i,q,d)) * coeffs[d][q]/*scalar_coeff.value(quad_point)*/ ;
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
    	      const double dnvINT = feINT.shape_grad(i,q) * normal_vectorINT;
    	      const double vEXT = feEXT.shape_value(i,q);
    	      const double dnvEXT = feEXT.shape_grad(i,q) * normal_vectorINT;
   	      const double uINT = inputINT[q];
    	      const double dnuINT = DinputINT[q] * normal_vectorINT;
    	      const double uEXT = inputEXT[q];
    	      const double dnuEXT = DinputEXT[q] * normal_vectorINT;
  	      resultINT(i) += dx*( nupenalty*vINT*uINT  - .5*(nuINT*dnvINT*uINT + nuINT*vINT*dnuINT) );
    	      resultINT(i) += dx*( -nupenalty*vINT*uEXT + .5*(nuINT*dnvINT*uEXT - nuEXT*vINT*dnuEXT) );
    	      resultEXT(i) += dx*( -nupenalty*vEXT*uINT - .5*(nuEXT*dnvEXT*uINT - nuINT*vEXT*dnuINT) );
	      resultEXT(i) += dx*( nupenalty*vEXT*uEXT  + .5*(nuEXT*dnvEXT*uEXT + nuEXT*vEXT*dnuEXT) );
    	    }
    	}
    }


    template <int dim, typename CoefficientTYPE>
    inline void
    ip_residualvs(dealii::Vector<double> &resultINT,
    		     dealii::Vector<double> &resultEXT,
    		     const dealii::FEValuesBase<dim> &feINT,
    		     const dealii::FEValuesBase<dim> &feEXT,
		  const dealii::VectorSlice<const std::vector<std::vector<double> > > &inputINT,
		  const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &DinputINT,
		  const dealii::VectorSlice<const std::vector<std::vector<double> > > &inputEXT,
		  const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &DinputEXT,
		     double penalty,
    		     double int_factor = 1.,
    		     double ext_factor = -1.)
    {
      const CoefficientTYPE scalar_coeff;
      const unsigned int n_comp = feINT.get_fe().n_components();
      const unsigned int n_dofs = feINT.dofs_per_cell;
      const unsigned int n_quads = feINT.n_quadrature_points;
    
      AssertVectorVectorDimension(inputINT, n_comp, n_quads);
      AssertVectorVectorDimension(DinputINT, n_comp, n_quads);
      AssertVectorVectorDimension(inputEXT, n_comp, n_quads);
      AssertVectorVectorDimension(DinputEXT, n_comp, n_quads);

      const double nuINT = int_factor;
      const double nuEXT = (ext_factor < 0) ? int_factor : ext_factor;
      const double nupenalty = .5 * penalty  * (nuINT + nuEXT);
 
      for (unsigned int q=0; q<n_quads; ++q)
    	{
          const dealii::Point<dim> quad_point = feINT.quadrature_point(q);
    	  const double dx = feINT.JxW(q) * scalar_coeff.value(quad_point);
    	  const dealii::Tensor<1,dim> normal_vectorINT = feINT.normal_vector(q);
 
    	  for (unsigned int i=0; i<n_dofs; ++i)
	    for (unsigned int d=0; d<n_comp; ++d)
	      {
		const double vINT = feINT.shape_value_component(i,q,d);
		const double dnvINT = feINT.shape_grad_component(i,q,d) * normal_vectorINT;
		const double vEXT = feEXT.shape_value_component(i,q,d);
		const double dnvEXT = feEXT.shape_grad_component(i,q,d) * normal_vectorINT;
		const double uINT = inputINT[d][q];
		const double dnuINT = DinputINT[d][q] * normal_vectorINT;
		const double uEXT = inputEXT[d][q];
		const double dnuEXT = DinputEXT[d][q] * normal_vectorINT;
		resultINT(i) += dx*( nupenalty*vINT*uINT  - .5*(nuINT*dnvINT*uINT + nuINT*vINT*dnuINT) );
		resultINT(i) += dx*( -nupenalty*vINT*uEXT + .5*(nuINT*dnvINT*uEXT - nuEXT*vINT*dnuEXT) );
		resultEXT(i) += dx*( -nupenalty*vEXT*uINT - .5*(nuEXT*dnvEXT*uINT - nuINT*vEXT*dnuINT) );
		resultEXT(i) += dx*( nupenalty*vEXT*uEXT  + .5*(nuEXT*dnvEXT*uEXT + nuEXT*vEXT*dnuEXT) );
	      }
    	}
    }

    template <int dim, typename CoefficientTYPE>
    inline void
    nitsche_residual(dealii::Vector<double> &result,
		     const dealii::FEValuesBase<dim> &fe,
		     const std::vector<double> &input,
		     const std::vector<dealii::Tensor<1,dim> > &Dinput,
		     const std::vector<double> &boundary_data,
		     double penalty,
		     double factor = 1.)
    {
       const CoefficientTYPE scalar_coeff;
       const unsigned int n_dofs = fe.dofs_per_cell;
            
       AssertDimension(input.size(), fe.n_quadrature_points);
       AssertDimension(Dinput.size(), fe.n_quadrature_points);
       AssertDimension(boundary_data.size(), fe.n_quadrature_points);
       
       for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
         {
	   const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	   const double dx = factor * fe.JxW(q);
	   const dealii::Tensor<1,dim> n = fe.normal_vector(q);
	   for (unsigned int i=0; i<n_dofs; ++i)
             {
	       const double u= input[q];
               const double g= boundary_data[q];
               const double dnv = fe.shape_grad(i,q) * n;
               const double dnu = Dinput[q] * n;
               const double v= fe.shape_value(i,q);
	       
               result(i) += dx*scalar_coeff.value(quad_point)*(2.*penalty*(u-g)*v - (u-g)*dnv - dnu*v);
             }
         }
     }

    template <int dim, typename CoefficientTYPE>
    inline void
    nitsche_residualvs(dealii::Vector<double> &result,
		     const dealii::FEValuesBase<dim> &fe,
		       const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
		       const dealii::VectorSlice<const std::vector<std::vector<dealii::Tensor<1,dim> > > > &Dinput,
		       const dealii::VectorSlice<const std::vector<std::vector<double> > > &boundary_data,
		     double penalty,
		     double factor = 1.)
    {
       const CoefficientTYPE scalar_coeff;
       const unsigned int n_dofs = fe.dofs_per_cell;
       const unsigned int n_comp = fe.get_fe().n_components();       

       AssertVectorVectorDimension(input, n_comp, fe.n_quadrature_points);
       AssertVectorVectorDimension(Dinput, n_comp, fe.n_quadrature_points);
       AssertVectorVectorDimension(boundary_data, n_comp, fe.n_quadrature_points);       

       for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
         {
	   const dealii::Point<dim> quad_point = fe.quadrature_point(q);
	   const double dx = factor * fe.JxW(q);
	   const dealii::Tensor<1,dim> n = fe.normal_vector(q);
	   for (unsigned int i=0; i<n_dofs; ++i)
	     for (unsigned int d=0; d<n_comp; ++d)
             {
	       const double u= input[d][q];
               const double g= boundary_data[d][q];
               const double dnv = fe.shape_grad_component(i,q,d) * n;
               const double dnu = Dinput[d][q] * n;
               const double v= fe.shape_value_component(i,q,d);
	       
               result(i) += dx*scalar_coeff.value(quad_point)*(2.*penalty*(u-g)*v - (u-g)*dnv - dnu*v);
             }
         }
     }

  } // end NAMESPACE = Diffusion
} // end NAMESPACE = LocalIntegrators

#endif // DIFFUSION_H
