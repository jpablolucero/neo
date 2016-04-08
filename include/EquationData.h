#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>

#include <vector>

template <int dim>
class DiffCoefficient final : public dealii::Function<dim>
{
 public:
  DiffCoefficient();
  virtual double value (const dealii::Point<dim>  &p,
			const unsigned int        block = 0) const override;
 private:
  std::vector<double> data;
};

template <int dim>
class ReferenceFunction final : public dealii::Function<dim>
{
public:
  ReferenceFunction();
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
  TotalCoefficient();
  virtual double value (const dealii::Point<dim>  &p,
			const unsigned int        block = 0) const override;
 private:
  std::vector<double> data;
};

template <int dim>
class ReacCoefficient final
{
 public:
  ReacCoefficient();
  double value (const dealii::Point<dim>  &p,
		const unsigned int        block_m = 0,
		const unsigned int        block_n = 0) const;
  void value_list (const std::vector<dealii::Point<dim> > &points,
		   std::vector<double>                    &values,
		   const unsigned int                     block_m = 0,
		   const unsigned int                     block_n = 0) const;
 private:
  std::vector<std::vector<double> > M_data;
};

#endif // EQUATIONDATA_H
