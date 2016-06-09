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
      return_grad[i]=-p(i)/((p.square()+0.025)*(p.square()+0.025));

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
Angle<dim>::Angle(const std::string& filename)
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

template class Coefficient<2>;
template class Coefficient<3>;
template class ReferenceFunction<2>;
template class ReferenceFunction<3>;
template class Angle<2>;
template class Angle<3>;
