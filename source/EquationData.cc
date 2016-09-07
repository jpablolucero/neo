#include <EquationData.h>

template <int dim>
Coefficient<dim>::Coefficient()
  : dealii::Function<dim>()
{}

template <int dim>
double
Coefficient<dim>::value (const dealii::Point<dim> &,
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
  /*  const double pi2 = dealii::numbers::PI;
    return std::sin(pi2*p(0))*std::sin(pi2*p(1));*/
  return 0.;
}

template <int dim>
dealii::Tensor<1,dim>
ReferenceFunction<dim>::gradient (const dealii::Point<dim> &p,
                                  const unsigned int /*d*/) const
{
  dealii::Tensor<1,dim> return_grad;
  /*  const double pi2 = dealii::numbers::PI*4/3;
    return_grad[0]=pi2*std::cos(pi2*p(0))*std::sin(pi2*p(1));
    return_grad[1]=pi2*std::sin(pi2*p(0))*std::cos(pi2*p(1));*/

  return return_grad;
}

template <int dim>
double
ReferenceFunction<dim>::laplacian(const dealii::Point<dim> &p,
                                  const unsigned int /*component = 0*/) const
{
  /*  const double pi2 = dealii::numbers::PI*3/2;
    const double return_value = -2*pi2*pi2*std::sin(pi2*p(0))*std::sin(pi2*p(1));
    return return_value;*/
  return -1.;
}

template <int dim>
XS<dim>::XS()
{}

template <int dim>
std::vector<std::vector<double> > XS<dim>::total(const std::vector<dealii::Point<dim> > &points,
                                                 unsigned int n_angles,
                                                 unsigned int ,
                                                 double factor)
{
  std::vector<std::vector<double> > xs ;
  xs.resize(n_angles) ;
  for (auto &angle : xs)
    {
      angle.resize(points.size());
      for (auto &point : angle)
        point = 1. * factor + 0.01 ;
    }
  return xs ;
}

template <int dim>
std::vector<std::vector<std::vector<double> > > XS<dim>::scattering(const std::vector<dealii::Point<dim> > &points,
    unsigned int n_angles,
    unsigned int,
    unsigned int,
    double factor)
{
  std::vector<std::vector<std::vector<double> > > xs ;
  xs.resize(n_angles);
  for (auto &angle_out : xs)
    {
      angle_out.resize(n_angles);
      for (auto &quads : angle_out)
        {
          quads.resize(points.size());
          for (auto &point : quads)
            point = 1. * factor ;
        }
    }
  return xs ;
}

template <int dim>
std::vector<std::vector<double> > XS<dim>::absorption(const std::vector<dealii::Point<dim> > &points,
                                                      const std::vector<double> &weights,
                                                      unsigned int n_angles,
                                                      unsigned int n_groups,
                                                      unsigned int bin,
                                                      double total_factor,
                                                      double scattering_factor)
{
  auto abs = total(points,n_angles,bin,total_factor);
  for (unsigned int bout = 0; bout < n_groups ; ++bout)
    for (unsigned int cin = 0; cin < n_angles ; ++cin)
      for (unsigned int cout = 0; cout < n_angles ; ++cout)
        for (unsigned int q = 0; q < points.size() ; ++q)
          abs[cin][q] -= weights[cout] * scattering(points,n_angles,bin,bout,scattering_factor)[cin][cout][q];
  return abs ;
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

template class Coefficient<2>;
template class Coefficient<3>;
template class ReferenceFunction<2>;
template class ReferenceFunction<3>;
template class Angle<2>;
template class Angle<3>;
template class XS<2>;
template class XS<3>;
