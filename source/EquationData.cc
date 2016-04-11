#include<EquationData.h>

// DiffCoefficient
template <int dim>
DiffCoefficient<dim>::DiffCoefficient()
  : dealii::Function<dim>(3),
    v_data(3, 0.)
{
  v_data[0] = 1./9.;
  v_data[1] = 1./18.;
  v_data[2] = 1./27.;
}

template <int dim>
double
DiffCoefficient<dim>::value (const dealii::Point<dim> &,
			     const unsigned int b) const
{
  return v_data[b];
}

// TotalCoefficient
template <int dim>
TotalCoefficient<dim>::TotalCoefficient()
  : dealii::Function<dim>(3),
    v_data(3, 0.)
{
  v_data[0] = 3.;
  v_data[1] = 6.;
  v_data[2] = 9.;
}

template <int dim>
double TotalCoefficient<dim>::value (const dealii::Point<dim>& /*p*/,
				     const unsigned int b) const
{
  return v_data[b];
}

// ReacCoefficient
template <int dim>
ReacCoefficient<dim>::ReacCoefficient()
  : M_data(3, std::vector<double>(3, 0.))
{
  M_data[0].resize(3);
  M_data[0][0] = 1.; M_data[0][1] = 0.; M_data[0][2] = 3.; 
  M_data[1].resize(3);
  M_data[1][0] = 1.; M_data[1][1] = 3.; M_data[1][2] = 0.; 
  M_data[2].resize(3);
  M_data[2][0] = 1.; M_data[2][1] = 3.; M_data[2][2] = 6.; 
}

template <int dim>
double ReacCoefficient<dim>::value (const dealii::Point<dim>&,
				    const unsigned int b_m,
				    const unsigned int b_n) const
{
  return M_data[b_m][b_n] ;
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

template class DiffCoefficient<2>;
template class DiffCoefficient<3>;
template class ReferenceFunction<2>;
template class ReferenceFunction<3>;
template class ReacCoefficient<2>;
template class ReacCoefficient<3>;
template class TotalCoefficient<2>;
template class TotalCoefficient<3>;
