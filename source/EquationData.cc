#include <EquationData.h>

MaterialParameter::MaterialParameter()
  : viscosity(1.)
{}

template <int dim>
Boundaries<dim>::Boundaries()
  : dealii::Function<dim>(dim+1)
{
  dirichlet.insert(0);
  dirichlet.insert(2);
  dirichlet.insert(3);
}

template <int dim>
void
Boundaries<dim>::vector_value_list (const std::vector<dealii::Point<dim> > &points,
				    std::vector<dealii::Vector<double> > &values) const
{}

template <int dim>
void
Boundaries<dim>::vector_values (const std::vector<dealii::Point<dim>>& points,
				std::vector<std::vector<double>>& values) const
{
  AssertVectorVectorDimension(values, this->n_components, points.size());

  for (unsigned int k = 0; k < points.size(); ++k)
    {
      const dealii::Point<dim>& p = points[k];
      values[0][k] = p(1) - p(1)*p(1);
    }
}

template <int dim>
ReferenceFunction<dim>::ReferenceFunction(unsigned int n_comp_)
  : dealii::Function<dim> (n_comp_)
{}

template <int dim>
void
ReferenceFunction<dim>::vector_value_list(const std::vector<dealii::Point<dim> > &points,
					  std::vector<dealii::Vector<double> > &values) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
    {
      const dealii::Point<dim>& p = points[k];
      values[k](0) = p(1) - p(1)*p(1); // velocity in x direction
      values[k](dim) = 2. * material_param.viscosity * (1. - p(0)); // pressure
    }
}


template class Boundaries<2>;
template class Boundaries<3>;
template class ReferenceFunction<2>;
template class ReferenceFunction<3>;
