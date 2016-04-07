#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>

#include <vector>

template <int dim>
class Coefficient final : public dealii::Function<dim>
{
public:
  Coefficient();

  double value (const dealii::Point<dim>  &p,
                const unsigned int        component = 0) const override;

  void value_list (const std::vector<dealii::Point<dim> > &points,
                   std::vector<double>                    &values,
                   const unsigned int                     component = 0) const override;

  dealii::Tensor<1,dim> gradient (const dealii::Point<dim>  &p,
                                  const unsigned int        component = 0) const override;
};

template <int dim>
class ReferenceFunction : public dealii::Function<dim>
{
public:
  ReferenceFunction(unsigned int n_comp_);

  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int /*component = 0*/) const;

  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim> &p,
                                          const unsigned int /*d*/) const;

  virtual double laplacian(const dealii::Point<dim> &p,
                           const unsigned int /*component = 0*/) const;
};

#endif // EQUATIONDATA_H
