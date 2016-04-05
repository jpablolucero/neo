#include<EquationData.h>

template <int dim>
double DiffCoefficient<dim>::value (const dealii::Point<dim> &p,
				    const unsigned int b) const
{
  return (b == 0) ? 1./(0.05 + 2.*p.square()) : 1.;
}

template <int dim>
void DiffCoefficient<dim>::value_list (const std::vector<dealii::Point<dim> > &points,
				       std::vector<double>                    &values,
				       const unsigned int                     block) const
{
  Assert (values.size() == points.size(),
	  dealii::ExcDimensionMismatch (values.size(), points.size()));
  
  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],block);
}

template <int dim>
double TotalCoefficient<dim>::value (const dealii::Point<dim> &p,
				     const unsigned int b) const
{
  return (b == 0) ? 1. : 1. + 0.0*p.square();
}






































































template <int dim>
void TotalCoefficient<dim>::value_list (const std::vector<dealii::Point<dim> > &points,
					std::vector<double>                    &values,
					const unsigned int                     block) const
{
  Assert (values.size() == points.size(),
	  dealii::ExcDimensionMismatch (values.size(), points.size()));
  
  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],block);
}

template <int dim>
double ReacCoefficient<dim>::value (const dealii::Point<dim> &p,
				    const unsigned int b_m,
				    const unsigned int b_n) const
{
  return (b_n == b_m) ? 1. : 0. + 0.0*p.square();
}

template <int dim>
void ReacCoefficient<dim>::value_list (const std::vector<dealii::Point<dim> > &points,
				       std::vector<double>                    &values,
				       const unsigned int                     block_m,
				       const unsigned int                     block_n) const
{
  Assert (values.size() == points.size(),
	  dealii::ExcDimensionMismatch (values.size(), points.size()));
  
  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],block_m,block_n);
}

template class ReacCoefficient<2>;
template class ReacCoefficient<3>;
template class TotalCoefficient<2>;
template class TotalCoefficient<3>;
template class DiffCoefficient<2>;
template class DiffCoefficient<3>;
