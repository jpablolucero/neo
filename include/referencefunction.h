#ifndef REFERENCEFUNCTION_H
#define REFERENCEFUNCTION_H

#include <deal.II/base/function.h>

template <int dim>
class ReferenceFunction : public dealii::Function<dim>
{
public:
  /** ctor */
  ReferenceFunction()
    : dealii::Function<dim>(1)
  {}

  /** Returns the function value for a single component at a given point. */
  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int /*component = 0*/) const
  {
    const double pi2 = dealii::numbers::PI;
    return std::sin(pi2*p(0))*std::sin(pi2*p(1));
  }

  /** Returns the function value for a single component at a given point. */
  virtual double laplacian(const dealii::Point<dim> &p,
                           const unsigned int /*component = 0*/) const
  {
    const double pi2 = dealii::numbers::PI;
    const double return_value = -2*pi2*pi2*std::sin(pi2*p(0))*std::sin(pi2*p(1));
    return return_value;
  }
};


#endif // REFERENCEFUNCTION_H
