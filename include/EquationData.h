#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/full_matrix.h>

#include <string>
#include <fstream>
#include <vector>

template <int dim>
class Coefficient final : public dealii::Function<dim>
{
public:
  Coefficient();
  Coefficient (const Coefficient &) = delete ;
  Coefficient &operator = (const Coefficient &) = delete;

  double value (const dealii::Point<dim>  &p,
                const unsigned int        component = 0) const override;

  void value_list (const std::vector<dealii::Point<dim> > &points,
                   std::vector<double>                    &values,
                   const unsigned int                     component = 0) const override;

  dealii::Tensor<1,dim> gradient (const dealii::Point<dim>  &p,
                                  const unsigned int        component = 0) const override;
};

template <int dim>
class ReferenceFunction final : public dealii::Function<dim>
{
public:
  ReferenceFunction(unsigned int n_comp_);
  ReferenceFunction (const ReferenceFunction &) = delete ;
  ReferenceFunction &operator = (const ReferenceFunction &) = delete;

  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int /*component = 0*/) const;

  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim> &p,
                                          const unsigned int /*d*/) const;

  virtual double laplacian(const dealii::Point<dim> &p,
                           const unsigned int /*component = 0*/) const;
};

template <int dim>
class Angle :
  public dealii::Quadrature<dim>
{
public:
  Angle(const std::string& filename);
};

#ifdef HEADER_IMPLEMENTATION
#include <EquationData.cc>
#endif

#endif // EQUATIONDATA_H
