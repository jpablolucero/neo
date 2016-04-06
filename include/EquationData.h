#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>

#include <vector>

template <int dim>
class DiffCoefficient final : public dealii::Function<dim>
{
 public:
  DiffCoefficient() : dealii::Function<dim>(){}
  virtual double value (const dealii::Point<dim>  &p,
			const unsigned int        block = 0) const override;
  virtual void value_list (const std::vector<dealii::Point<dim> > &points,
			   std::vector<double>                    &values,
			   const unsigned int                     block = 0) const override;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim>  &p,
					  const unsigned int        component = 0) const override;
};

template <int dim>
class ReferenceFunction final : public dealii::Function<dim>
{
public:
  ReferenceFunction() : dealii::Function<dim>(){}
  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int /*component = 0*/) const override;
  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim> &p,
                                          const unsigned int /*d*/) const override;
  virtual double laplacian(const dealii::Point<dim> &p,
                           const unsigned int /*component = 0*/) const override;
};

template <int dim>
class TotalCoefficient final : public dealii::Function<dim>
{
 public:
  TotalCoefficient() : dealii::Function<dim>(){}
  double value (const dealii::Point<dim>  &p,
                const unsigned int        block = 0) const override;
  void value_list (const std::vector<dealii::Point<dim> > &points,
		   std::vector<double>                    &values,
		   const unsigned int                     block = 0) const override;
};

template <int dim>
class ReacCoefficient final
{
 public:
  virtual double value (const dealii::Point<dim>  &p,
			const unsigned int        block_m = 0,
			const unsigned int        block_n = 0) const;
  virtual void value_list (const std::vector<dealii::Point<dim> > &points,
			   std::vector<double>                    &values,
			   const unsigned int                     block_m = 0,
			   const unsigned int                     block_n = 0) const;
};

#endif // EQUATIONDATA_H
