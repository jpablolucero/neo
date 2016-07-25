#ifndef EQUATIONDATA_CC
#define EQUATIONDATA_CC

#include <EquationData.h>

template <int dim>
Coefficient<dim>::Coefficient()
  : dealii::Function<dim>()
{}

template <int dim>
double
Coefficient<dim>::value (const dealii::Point<dim> &p,
                         const unsigned int d) const
{
  return (d == 0) ? 1.:1.;//1./(0.05 + 2.*p.square()) : 1.;
}

template <int dim>
void Coefficient<dim>::value_list (const std::vector<dealii::Point<dim> > &points,
                                   std::vector<double>                    &values,
                                   const unsigned int                     component) const
{
  Assert (values.size() == points.size(),
          dealii::ExcDimensionMismatch (values.size(), points.size()));

  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],component);
}

template <int dim>
dealii::Tensor<1,dim>
Coefficient<dim>::gradient (const dealii::Point<dim> &p,
                            const unsigned int  d) const
{
  dealii::Tensor<1,dim> return_grad;
  if (d == 0)
    for (unsigned int i=0; i<dim; ++i)
      return_grad[i]=0.;/*-p(i)/((p.square()+0.025)*(p.square()+0.025));*/

  return return_grad;
}

template <int dim>
ReferenceFunction<dim>::ReferenceFunction(unsigned int n_comp_)
  : dealii::Function<dim> (n_comp_)
{}

template <int dim>
double
ReferenceFunction<dim>::value(const dealii::Point<dim> &p,
                              const unsigned int /*component = 0*/) const
{
  const double pi2 = dealii::numbers::PI;
  return std::sin(pi2*p(0))*std::sin(pi2*p(1));
}

template <int dim>
dealii::Tensor<1,dim>
ReferenceFunction<dim>::gradient (const dealii::Point<dim> &p,
                                  const unsigned int /*d*/) const
{
  dealii::Tensor<1,dim> return_grad;
  const double pi2 = dealii::numbers::PI*4/3;
  return_grad[0]=pi2*std::cos(pi2*p(0))*std::sin(pi2*p(1));
  return_grad[1]=pi2*std::sin(pi2*p(0))*std::cos(pi2*p(1));

  return return_grad;
}

template <int dim>
double
ReferenceFunction<dim>::laplacian(const dealii::Point<dim> &p,
                                  const unsigned int /*component = 0*/) const
{
  const double pi2 = dealii::numbers::PI*3/2;
  const double return_value = -2*pi2*pi2*std::sin(pi2*p(0))*std::sin(pi2*p(1));
  return return_value;
}

template <int dim>
Angle<dim>::Angle(const std::string &filename)
{
  std::ifstream file(filename.c_str());
  AssertThrow (file.is_open(), dealii::ExcIO());

  unsigned int filedim, n_points, n_moments;
  file >> filedim;
  file >> n_points;
  file >> n_moments;

  AssertThrow(filedim == dim, dealii::ExcDimensionMismatch(filedim,dim));

  std::vector<dealii::Point<dim> > points(n_points);
  std::vector<double> weights(n_points, 0.);

  for (unsigned int i=0; i<points.size(); ++i)
    {
      for (unsigned int d=0; d<filedim; ++d)
        file >> points[i](d);
      file >> weights[i];
    }

  AssertThrow(weights[n_points-1] != 0., dealii::ExcIO());
  dealii::Quadrature<dim>::initialize(points, weights);
}

//---------------------------------------------------------------------------
//    $Id: solution.h 67 2015-03-03 11:34:17Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2014 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------
template <>
const dealii::Point<1>
SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers]
= { dealii::Point<1>(-1.0 / 3.0),
    dealii::Point<1>(0.0),
    dealii::Point<1>(+1.0 / 3.0)   };

template <>
const dealii::Point<2>
SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers]
= { dealii::Point<2>(-0.5, +0.5),
    dealii::Point<2>(-0.5, -0.5),
    dealii::Point<2>(+0.5, -0.5)   };

template <>
const dealii::Point<3>
SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers]
= { dealii::Point<3>(-0.5, +0.5, 0.25),
    dealii::Point<3>(-0.6, -0.5, -0.125),
    dealii::Point<3>(+0.5, -0.5, 0.5)   };

template <int dim>
const double SolutionBase<dim>::width = 1./3.;


//MFSolution
template <int dim>
double MFSolution<dim>::value (const dealii::Point<dim>   &p,
			       const unsigned int) const
{
  const double pi = dealii::numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const dealii::Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
      return_value += std::exp(-x_minus_xi.norm_square() /
			       (this->width * this->width));
    }

  return return_value /
    dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}

template <int dim>
dealii::Tensor<1,dim> MFSolution<dim>::gradient (const dealii::Point<dim>   &p,
						 const unsigned int) const
{
  const double pi = dealii::numbers::PI;
  dealii::Tensor<1,dim> return_value;

  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const dealii::Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

      return_value += (-2 / (this->width * this->width) *
		       std::exp(-x_minus_xi.norm_square() /
				(this->width * this->width)) *
		       x_minus_xi);
    }

  return return_value / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
							    this->width);
}

//MFRightHandSide
template <int dim>
double MFRightHandSide<dim>::value (const dealii::Point<dim>   &p,
				  const unsigned int) const
{
  const double pi = dealii::numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
      const dealii::Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

      return_value += 
        ( (2*dim - 4*x_minus_xi.norm_square()/(this->width * this->width) )/
	  (this->width * this->width) *
	  std::exp(-x_minus_xi.norm_square() /
		   (this->width * this->width)));
    }

  return return_value / dealii::Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
							    this->width);
}

template class Coefficient<2>;
template class Coefficient<3>;
template class ReferenceFunction<2>;
template class ReferenceFunction<3>;
template class Angle<2>;
template class Angle<3>;

template class SolutionBase<1>;
template class SolutionBase<2>;
template class SolutionBase<3>;
template class MFSolution<1>;
template class MFSolution<2>;
template class MFSolution<3>;
template class MFRightHandSide<1>;
template class MFRightHandSide<2>;
template class MFRightHandSide<3>;

#endif // EQUATIONDATA_CC


