#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/vector.h>

#include <set>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>

class MaterialParameter
{
public:
  MaterialParameter();
  const double viscosity;
};

template <int dim>
class Boundaries final : public dealii::Function<dim>
{
public:
  Boundaries();
  Boundaries (const Boundaries &) = delete;
  Boundaries &operator = (const Boundaries &) = delete;

  std::set<unsigned int> dirichlet;

  void vector_value_list (const std::vector<dealii::Point<dim> > &points,
  			  std::vector<dealii::Vector<double> > &values) const override;
  void vector_values (const std::vector<dealii::Point<dim> > &points,
		      std::vector<std::vector<double> > &values) const override;
};

template <int dim>
class ReferenceFunction final : public dealii::Function<dim>
{
public:
  ReferenceFunction(unsigned int n_comp_);
  ReferenceFunction (const ReferenceFunction &) = delete ;
  ReferenceFunction &operator = (const ReferenceFunction &) = delete;

  void vector_value_list (const std::vector<dealii::Point<dim> > &points,
  			  std::vector<dealii::Vector<double> > &values) const override;

private:
  MaterialParameter material_param;
};


#ifdef HEADER_IMPLEMENTATION
#include <EquationData.cc>
#endif

#endif // EQUATIONDATA_H
