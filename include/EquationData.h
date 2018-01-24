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

#include <set>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>

class MaterialParameter
{
public:
  MaterialParameter();
  const double lambda;
  const double mu;
};

template <int dim>
class Boundaries final : public dealii::Function<dim>
{
public:
  Boundaries(/*unsigned int n_comp*/);
  Boundaries (const Boundaries &) = delete;
  Boundaries &operator = (const Boundaries &) = delete;

  std::set<unsigned int> dirichlet;

  void vector_value_list (const std::vector<dealii::Point<dim> > &points,
			  std::vector<dealii::Vector<double> > &values) const override;
  void vector_values (const std::vector<dealii::Point<dim> > &points,
		      std::vector<std::vector<double> > &values) const override;
};

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

template<int dim>
class XS final
{
public:
  XS();
  XS (const XS &) = delete ;
  XS &operator = (const XS &) = delete;

  std::vector<double> grid ;

  std::vector<std::vector<double> > total(const std::vector<dealii::Point<dim> > &points,
                                          unsigned int n_angles,
                                          unsigned int group = 0,
                                          double factor = 1.);

  std::vector<std::vector<std::vector<double> > > scattering(const std::vector<dealii::Point<dim> > &points,
                                                             unsigned int n_angles,
                                                             unsigned int group_in = 0,
                                                             unsigned int group_out = 0,
                                                             double factor = 1.);

  std::vector<std::vector<double> > absorption(const std::vector<dealii::Point<dim> > &points,
                                               const std::vector<double> &weights,
                                               unsigned int n_angles,
                                               unsigned int n_groups,
                                               unsigned int bin,
                                               double total_factor,
                                               double scattering_factor);
};

template <int dim>
class Angle :
  public dealii::Quadrature<dim>
{
public:
  Angle(const std::string &filename);
};

//---------------------------------------------------------------------------
//    $Id: solution.h 67 2015-03-03 11:34:17Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2014 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------
template <int dim>
class SolutionBase
{
protected:
  static const unsigned int n_source_centers = 3;
  static const dealii::Point<dim>   source_centers[n_source_centers];
  static const double       width;
};


// MFSolution
template <int dim>
class MFSolution : public dealii::Function<dim>,
  protected SolutionBase<dim>
{
public:
  MFSolution (unsigned int n_comp) : dealii::Function<dim>(n_comp) {}

  virtual double value (const dealii::Point<dim> &p,
                        const unsigned int       component = 0) const;

  virtual dealii::Tensor<1,dim> gradient (const dealii::Point<dim> &p,
                                          const unsigned int       component = 0) const;
};


// MFRightHandside = negative Laplacian of MFSolution
template <int dim>
class MFRightHandSide : public dealii::Function<dim>,
  protected SolutionBase<dim>
{
public:
  MFRightHandSide (unsigned int n_comp) : dealii::Function<dim>(n_comp) {}

  virtual double value (const dealii::Point<dim>   &p,
                        const unsigned int         component = 0) const;
};

// MFDiffCoefficient
template <int dim>
class MFDiffCoefficient : public dealii::Function<dim>,
  protected SolutionBase<dim>
{
public:
  MFDiffCoefficient (unsigned int n_comp) : dealii::Function<dim>(n_comp) {}

  virtual double value (const dealii::Point<dim>   &p,
                        const unsigned int         component = 0) const;
  dealii::VectorizedArray<double> value (const dealii::Point<dim,dealii::VectorizedArray<double> >  &p,
                                         const unsigned int         component = 0) const;
};

inline double planck_integral (double sigma, double temperature)
{
  //  integral of spectral radiance from sigma (cm-1) to infinity.
  //  result is W/m2/sr.
  //  follows Widger and Woodall, Bulletin of the American Meteorological
  //  Society, Vol. 57, No. 10, pp. 1217
  //  constants
  bool linear = false ;
  double old_temperature = temperature ;
  if (temperature < 10.) 
    {
      linear = true ;
      old_temperature = temperature ;
      temperature = 10.;
    }
  double Planck =  6.6260693e-34 ;
  double Boltzmann = 1.380658e-23 ;
  double Speed_of_light = 299792458.0 ;
  double Speed_of_light_sq = Speed_of_light * Speed_of_light ;

  //  compute powers of x, the dimensionless spectral coordinate
  double c1 =  (Planck*Speed_of_light/Boltzmann) ;
  double x =  c1 * 100 * sigma / temperature ;
  double x2 = x *  x  ;
  double x3 = x *  x2 ;

  //  decide how many terms of sum are needed
  double iterations = 2.0 + 20.0/x ;
  iterations = (iterations<512) ? iterations : 512 ;
  int iter = int(iterations) ;

  //  add up terms of sum
  double sum = 0  ;
  for (int n=1;  n<iter; n++)
    {
      double  dn = 1.0/n ;
      sum  += exp(-n*x)*(x3 + 3.*(x2 + 2.*(x+dn)*dn)*dn)*dn;
    }

  //  return result, in units of W/m2/sr
  double c2 =  (2.0*Planck*Speed_of_light_sq) ;
  if (linear) { return c2*std::pow(temperature/c1,4)*sum / 10. * old_temperature ; }
  return c2*std::pow(temperature/c1,4)*sum  ;
}

inline double Dplanck_integral (double sigma, double temperature)
{
  //  integral of spectral radiance from sigma (cm-1) to infinity.
  //  result is W/m2/sr.
  //  follows Widger and Woodall, Bulletin of the American Meteorological
  //  Society, Vol. 57, No. 10, pp. 1217
  //  constants
  double Planck =  6.6260693e-34 ;
  double Boltzmann = 1.380658e-23 ;
  double Speed_of_light = 299792458.0 ;
  double Speed_of_light_sq = Speed_of_light * Speed_of_light ;

  //  compute powers of x, the dimensionless spectral coordinate
  double c1 =  (Planck*Speed_of_light/Boltzmann) ;
  double x =  c1 * 100 * sigma / temperature ;
  double x2 = x *  x  ;
  double x3 = x *  x2 ;
  double x4 = x *  x3 ;

  //  decide how many terms of sum are needed
  double iterations = 2.0 + 20.0/x ;
  iterations = (iterations<512) ? iterations : 512 ;
  int iter = int(iterations) ;

  //  add up terms of sum
  double sum = 0  ;
  for (int n=1;  n<iter; n++)
    {
      double  dn = 1.0/n ;
      sum  += n * exp(-n*x) * (x4 + 4.*(x3 + 3.*(x2 + 2.*(x + dn)*dn)*dn)*dn)*dn ;
    }

  //  return result, in units of W/m2/sr
  double c2 =  (2.0*Planck*Speed_of_light_sq) ;
  if (temperature < 10.) {return planck_integral(sigma,10.)/10.;}
  return c2/c1*std::pow(temperature/c1,3)*sum ;
}

#ifdef HEADER_IMPLEMENTATION
#include <EquationData.cc>
#endif

#endif // EQUATIONDATA_H
