#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>

#include <vector>

template <int dim>
class Coefficient final : public dealii::Function<dim>
{
public:
  Coefficient() : dealii::Function<dim>() {}
  double value (const dealii::Point<dim>  &p,
                const unsigned int        component = 0) const override;
  void value_list (const std::vector<dealii::Point<dim> > &points,
                           std::vector<double>                    &values,
                           const unsigned int                     component = 0) const override;
};

#endif // EQUATIONDATA_H
