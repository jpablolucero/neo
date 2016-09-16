#ifndef TRANSPORT_H
#define TRANSPORT_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>

#include <functional>

namespace LocalIntegrators
{
  namespace Transport
  {
    template <int dim>
    inline void
    cell_matrix(dealii::FullMatrix<double> &M,
                const dealii::FEValuesBase<dim> &fe,
                const std::vector<dealii::Point<dim> > &angles,
                const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) -= dx * angles[d] * fe.shape_grad_component(i,q,d) * fe.shape_value_component(j,q,d);
        }
    }

    template <int dim>
    inline void
    total_matrix(dealii::FullMatrix<double> &M,
                 const dealii::FEValuesBase<dim> &fe,
                 const std::vector<std::vector<double> > &total,
                 const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) += dx * total[d][q] * fe.shape_value_component(i,q,d) * fe.shape_value_component(j,q,d) ;
        }
    }

    template <int dim>
    inline void
    T_absorption_matrix(dealii::FullMatrix<double> &M,
                        const dealii::FEValuesBase<dim> &feT,
                        const dealii::FEValuesBase<dim> &feV,
                        const std::vector<double> &weights,
                        const std::vector<std::vector<double> >  &abs,
                        const double factor = 1.)
    {
      const unsigned int n_dofsT = feT.dofs_per_cell;
      const unsigned int n_dofsV = feV.dofs_per_cell;
      const unsigned int n_comps = feV.get_fe().n_components();
      const unsigned int n_quads = feT.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = feT.JxW(q)*factor;
          for (unsigned int i=0; i<n_dofsT; ++i)
            for (unsigned int j=0; j<n_dofsV; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) -= dx * weights[d] * abs[d][q] * feT.shape_value(i,q) * feV.shape_value_component(j,q,d) ;
        }
    }

    template <int dim>
    inline void
    T_matrix(dealii::FullMatrix<double> &M,
             const dealii::FEValuesBase<dim> &feT,
             const std::vector<double> &weights,
             const std::vector<std::vector<double> >  &abs,
             std::function<double(double)> planck,
             const double factor = 1.)
    {
      const unsigned int n_dofs = feT.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = feT.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = feT.JxW(q)*factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) += dx * weights[d] * abs[d][q] * feT.shape_value(i,q) * planck(feT.shape_value(j,q)) ;
        }
    }

    template <int dim>
    inline void
    DT_matrix(dealii::FullMatrix<double> &M,
              const dealii::FEValuesBase<dim> &feT,
              const dealii::VectorSlice<const std::vector<std::vector<double> > > &prev,
              const std::vector<double> &weights,
              const std::vector<std::vector<double> >  &abs,
              std::function<double(double)> Dplanck,
              const double factor = 1.)
    {
      const unsigned int n_dofs = feT.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = feT.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = feT.JxW(q)*factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) += dx * weights[d] * abs[d][q] * feT.shape_value(i,q) * Dplanck(prev[0][q]) * feT.shape_value(j,q) ;
        }
    }

    template <int dim>
    inline void
    emission_matrix(dealii::FullMatrix<double> &M,
                    const dealii::FEValuesBase<dim> &feV,
                    const dealii::FEValuesBase<dim> &feT,
                    const std::vector<std::vector<double> >  &abs,
                    std::function<double(double)> planck,
                    const double factor = 1.)
    {
      const unsigned int n_dofsT = feT.dofs_per_cell;
      const unsigned int n_dofsV = feV.dofs_per_cell;
      const unsigned int n_comps = feV.get_fe().n_components();
      const unsigned int n_quads = feT.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = feV.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofsV; ++i)
            for (unsigned int j=0; j<n_dofsT; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) -= dx * abs[d][q] * feV.shape_value_component(i,q,d) * planck(feT.shape_value(j,q)) ;
        }
    }

    template <int dim>
    inline void
    Demission_matrix(dealii::FullMatrix<double> &M,
                     const dealii::FEValuesBase<dim> &feV,
                     const dealii::VectorSlice<const std::vector<std::vector<double> > > &prev,
                     const dealii::FEValuesBase<dim> &feT,
                     const std::vector<std::vector<double> >  &abs,
                     std::function<double(double)> Dplanck,
                     const double factor = 1.)
    {
      const unsigned int n_dofsT = feT.dofs_per_cell;
      const unsigned int n_dofsV = feV.dofs_per_cell;
      const unsigned int n_comps = feV.get_fe().n_components();
      const unsigned int n_quads = feT.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = feV.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofsV; ++i)
            for (unsigned int j=0; j<n_dofsT; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                M(i,j) -= dx * abs[d][q] * feV.shape_value_component(i,q,d) * Dplanck(prev[0][q]) * feT.shape_value(j,q) ;
        }
    }

    template <int dim>
    inline void
    redistribution_matrix(dealii::FullMatrix<double> &M,
                          const dealii::FEValuesBase<dim> &fe,
                          const std::vector<double> &weights,
                          const std::vector<std::vector<std::vector<double> > > &redistribution,
                          const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                for (unsigned int d0=0; d0<n_comps; ++d0)
                  M(i,j) -= dx * weights.at(d0) * redistribution[d0][d][q] *
                            fe.shape_value_component(j,q,d) * fe.shape_value_component(i,q,d0) ;
        }
    }

    template <int dim>
    inline void
    ip_matrix(dealii::FullMatrix<double> &M_INT_INT,
              dealii::FullMatrix<double> &M_INT_EXT,
              dealii::FullMatrix<double> &M_EXT_INT,
              dealii::FullMatrix<double> &M_EXT_EXT,
              const dealii::FEValuesBase<dim> &feINT,
              const dealii::FEValuesBase<dim> &feEXT,
              const std::vector<dealii::Point<dim> > &angles,
              const std::vector<double> &weights,
              const std::vector<std::vector<std::vector<double> > > redistribution1,
              const std::vector<std::vector<std::vector<double> > > redistribution2,
              const double diameter1 = 0.,
              const double diameter2 = 0.,
              const double gamma = 1./2.,
              const double factor = 1.)
    {
      const unsigned int n_dofs = feINT.dofs_per_cell;
      const unsigned int n_comps = feINT.get_fe().n_components();
      const unsigned int n_quads = feINT.n_quadrature_points;

      AssertDimension(M_INT_INT.n(), n_dofs);
      AssertDimension(M_INT_INT.m(), n_dofs);
      AssertDimension(M_INT_EXT.n(), n_dofs);
      AssertDimension(M_INT_EXT.m(), n_dofs);
      AssertDimension(M_EXT_INT.n(), n_dofs);
      AssertDimension(M_EXT_INT.m(), n_dofs);
      AssertDimension(M_EXT_EXT.n(), n_dofs);
      AssertDimension(M_EXT_EXT.m(), n_dofs);

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = 0.5*feINT.JxW(q)*factor;
          for (unsigned int d=0; d<n_comps; ++d)
            {
              double sigma1 = 0.;
              double sigma2 = 0.;
              for (unsigned int d0=0; d0<n_comps; ++d0)
                {
                  sigma1 += redistribution1[d][d0][q]*weights.at(d0) ;
                  sigma2 += redistribution2[d][d0][q]*weights.at(d0) ;
                }
              double gamma1 = 1.0 / std::max(1.0, gamma * sigma1 * diameter1);
              double gamma2 = 1.0 / std::max(1.0, gamma * sigma2 * diameter2);
              double w2 = 1.0 + .5*(gamma1-gamma2);
              double w1 = 1.0 + .5*(gamma2-gamma1);
              double gammaa = .5*(gamma1+gamma2) ;
              for (unsigned int i=0; i<n_dofs; ++i)
                for (unsigned int j=0; j<n_dofs; ++j)
                  {
                    const double nbeta = feINT.normal_vector(q) * angles.at(d);
                    const double anbeta = std::fabs(nbeta);
                    if (std::fabs(sigma1-sigma2) > 1.E-10)
                      {
                        if ((sigma1-sigma2) > 0.)
                          {
                            if (nbeta > 0.)
                              gammaa = gamma2/std::sqrt(2.) ;
                            else
                              gammaa = gamma2;
                          }
                        else
                          {
                            if (nbeta > 0.)
                              gammaa = gamma1;
                            else
                              gammaa = gamma1/std::sqrt(2.) ;
                          }
                      }
                    else
                      gammaa = .5*(gamma1+gamma2) ;
                    const double vINT = feINT.shape_value_component(i,q,d);
                    const double vEXT = feEXT.shape_value_component(i,q,d);
                    const double uINT = feINT.shape_value_component(j,q,d);
                    const double uEXT = feEXT.shape_value_component(j,q,d);
                    M_INT_INT(i,j) += dx * w1 * nbeta * (uINT * vINT);
                    M_EXT_INT(i,j) -= dx * w1 * nbeta * (uINT * vEXT);
                    M_EXT_EXT(i,j) -= dx * w2 * nbeta * (uEXT * vEXT);
                    M_INT_EXT(i,j) += dx * w2 * nbeta * (uEXT * vINT);
                    //Jump
                    M_INT_INT(i,j) += dx * gammaa * anbeta * (uINT * vINT);
                    M_EXT_INT(i,j) -= dx * gammaa * anbeta * (uINT * vEXT);
                    M_EXT_EXT(i,j) += dx * gammaa * anbeta * (uEXT * vEXT);
                    M_INT_EXT(i,j) -= dx * gammaa * anbeta * (uEXT * vINT);
                  }
            }
        }
    }

    template <int dim>
    inline void
    boundary (dealii::FullMatrix<double> &M,
              const dealii::FEValuesBase<dim> &fe,
              const std::vector<dealii::Point<dim> > &angles,
              const std::vector<double> &weights,
              const std::vector<std::vector<std::vector<double> > > redistribution,
              const double diameter = 0.,
              const double gamma = 1./16.,
              const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      Assert (M.m() == n_dofs, dealii::ExcDimensionMismatch(M.m(), n_dofs));
      Assert (M.n() == n_dofs, dealii::ExcDimensionMismatch(M.n(), n_dofs));

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q)*factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int j=0; j<n_dofs; ++j)
              for (unsigned int d=0; d<n_comps; ++d)
                {
                  double sigma = 0.;
                  for (unsigned int d0=0; d0<n_comps; ++d0)
                    sigma += redistribution[d][d0][q] * weights.at(d0);
                  const double gammaa = 1.0 / std::max(1.0, gamma * sigma * diameter);
                  const double nbeta = fe.normal_vector(q) * angles.at(d);
                  if (nbeta > 0)
                    M(i,j) += dx * gammaa * nbeta * fe.shape_value_component(i,q,d) * fe.shape_value_component(j,q,d) ;
                }
        }
    }

    template <int dim>
    inline void
    cell_residual(dealii::Vector<double> &result,
                  const dealii::FEValuesBase<dim> &fe,
                  const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                  const std::vector<dealii::Point<dim> > &angles,
                  const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) -= dx * (angles[d] * fe.shape_grad_component(i,q,d)) * input[d][q] ;
        }
    }

    template <int dim>
    inline void
    total_residual(dealii::Vector<double> &result,
                   const dealii::FEValuesBase<dim> &fe,
                   const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                   const std::vector<std::vector<double> >  &total,
                   const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) += dx * total[d][q] * fe.shape_value_component(i,q,d) * input[d][q];
        }
    }

    template <int dim>
    inline void
    T_absorption_residual(dealii::Vector<double> &result,
                          const dealii::FEValuesBase<dim> &fe,
                          const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                          const std::vector<double> &weights,
                          const std::vector<std::vector<double> >  &abs,
                          const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = input.size();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor ;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) -= dx * weights[d] * abs[d][q] * fe.shape_value(i,q) * input[d][q] ;
        }
    }

    template <int dim>
    inline void
    T_residual(dealii::Vector<double> &result,
               const dealii::FEValuesBase<dim> &fe,
               const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
               const std::vector<double> &weights,
               const std::vector<std::vector<double> >  &abs,
               std::function<double(double)> planck,
               const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor ;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) += dx * weights[d] * abs[d][q] * fe.shape_value(i,q) * planck(input[0][q]) ;
        }
    }

    template <int dim>
    inline void
    DT_residual(dealii::Vector<double> &result,
                const dealii::FEValuesBase<dim> &fe,
                const dealii::VectorSlice<const std::vector<std::vector<double> > > &prev,
                const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                const std::vector<double> &weights,
                const std::vector<std::vector<double> >  &abs,
                std::function<double(double)> Dplanck,
                const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor ;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) += dx * weights[d] * abs[d][q] * fe.shape_value(i,q) * Dplanck(prev[0][q]) * input[0][q] ;
        }
    }

    template <int dim>
    inline void
    emission_residual(dealii::Vector<double> &result,
                      const dealii::FEValuesBase<dim> &fe,
                      const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                      const std::vector<std::vector<double> >  &abs,
                      std::function<double(double)> planck,
                      const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor ;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) -= dx * abs[d][q] * fe.shape_value_component(i,q,d) * planck(input[0][q]) ;
        }
    }

    template <int dim>
    inline void
    Demission_residual(dealii::Vector<double> &result,
                       const dealii::FEValuesBase<dim> &fe,
                       const dealii::VectorSlice<const std::vector<std::vector<double> > > &prev,
                       const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                       const std::vector<std::vector<double> >  &abs,
                       std::function<double(double)> Dplanck,
                       const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = abs.size();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor ;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              result(i) -= dx * abs[d][q] * fe.shape_value_component(i,q,d) * Dplanck(prev[0][q]) * input[0][q] ;
        }
    }

    template <int dim>
    inline void
    redistribution_residual(dealii::Vector<double> &result,
                            const dealii::FEValuesBase<dim> &fe,
                            const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
                            const std::vector<double> &weights,
                            const std::vector<std::vector<std::vector<double> > > &redistribution,
                            const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q) * factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              for (unsigned int d0=0; d0<n_comps; ++d0)
                result(i) -= dx * weights.at(d0) * redistribution[d0][d][q] * input[d][q] * fe.shape_value_component(i,q,d0) ;
        }
    }

    template <int dim>
    inline void
    ip_residual(dealii::Vector<double> &resultINT,
                dealii::Vector<double> &resultEXT,
                const dealii::FEValuesBase<dim> &feINT,
                const dealii::FEValuesBase<dim> &feEXT,
                const dealii::VectorSlice<const std::vector<std::vector<double> > > &inputINT,
                const dealii::VectorSlice<const std::vector<std::vector<double> > > &inputEXT,
                const std::vector<dealii::Point<dim> > &angles,
                const std::vector<double> &weights,
                const std::vector<std::vector<std::vector<double> > > redistribution1,
                const std::vector<std::vector<std::vector<double> > > redistribution2,
                const double diameter1 = 0.,
                const double diameter2 = 0.,
                const double gamma = 1./2.,
                const double factor = 1.)
    {
      const unsigned int n_dofs = feINT.dofs_per_cell;
      const unsigned int n_comps = feINT.get_fe().n_components();
      const unsigned int n_quads = feINT.n_quadrature_points;

      AssertDimension(resultINT.size(), n_dofs);
      AssertDimension(resultEXT.size(), n_dofs);

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = 0.5*feINT.JxW(q)*factor;
          for (unsigned int d=0; d<n_comps; ++d)
            {
              double sigma1 = 0.;
              double sigma2 = 0.;
              for (unsigned int d0=0; d0<n_comps; ++d0)
                {
                  sigma1 += redistribution1[d][d0][q]*weights.at(d0) ;
                  sigma2 += redistribution2[d][d0][q]*weights.at(d0) ;
                }
              double gamma1 = 1.0 / std::max(1.0, gamma * sigma1 * diameter1);
              double gamma2 = 1.0 / std::max(1.0, gamma * sigma2 * diameter2);
              double w2 = 1.0 + .5*(gamma1-gamma2);
              double w1 = 1.0 + .5*(gamma2-gamma1);
              double gammaa = .5*(gamma1+gamma2) ;
              for (unsigned int i=0; i<n_dofs; ++i)
                {
                  const double nbeta = feINT.normal_vector(q) * angles.at(d);
                  const double anbeta = std::fabs(nbeta);
                  if (std::fabs(sigma1-sigma2) > 1.E-10)
                    {
                      if ((sigma1-sigma2) > 0.)
                        {
                          if (nbeta > 0.)
                            gammaa = gamma2/std::sqrt(2.) ;
                          else
                            gammaa = gamma2;
                        }
                      else
                        {
                          if (nbeta > 0.)
                            gammaa = gamma1;
                          else
                            gammaa = gamma1/std::sqrt(2.) ;
                        }
                    }
                  else
                    gammaa = .5*(gamma1+gamma2) ;
                  const double vINT = feINT.shape_value_component(i,q,d);
                  const double vEXT = feEXT.shape_value_component(i,q,d);
                  const double uINT = inputINT[d][q];
                  const double uEXT = inputEXT[d][q];
                  resultINT(i) += dx * w1 * nbeta * (uINT * vINT);
                  resultEXT(i) -= dx * w1 * nbeta * (uINT * vEXT);
                  resultEXT(i) -= dx * w2 * nbeta * (uEXT * vEXT);
                  resultINT(i) += dx * w2 * nbeta * (uEXT * vINT);
                  //Jump
                  resultINT(i) += dx * gammaa * anbeta * (uINT * vINT);
                  resultEXT(i) -= dx * gammaa * anbeta * (uINT * vEXT);
                  resultEXT(i) += dx * gammaa * anbeta * (uEXT * vEXT);
                  resultINT(i) -= dx * gammaa * anbeta * (uEXT * vINT);
                }
            }
        }
    }

    template <int dim>
    inline void
    boundary(dealii::Vector<double> &result,
             const dealii::FEValuesBase<dim> &fe,
             const dealii::VectorSlice<const std::vector<std::vector<double> > > &input,
             const std::vector<dealii::Point<dim> > &angles,
             const std::vector<double> &weights,
             const std::vector<std::vector<std::vector<double> > > redistribution,
             const double diameter = 0.,
             const double gamma = 1./16.,
             const double factor = 1.)
    {
      const unsigned int n_dofs = fe.dofs_per_cell;
      const unsigned int n_comps = fe.get_fe().n_components();
      const unsigned int n_quads = fe.n_quadrature_points;

      for (unsigned int q=0; q<n_quads; ++q)
        {
          const double dx = fe.JxW(q)*factor;
          for (unsigned int i=0; i<n_dofs; ++i)
            for (unsigned int d=0; d<n_comps; ++d)
              {
                double sigma {};
                for (unsigned int d0=0; d0<n_comps; ++d0)
                  sigma += redistribution[d0][d][q] * weights.at(d0);
                const double gammaa = 1.0 / std::max(1.0, gamma * sigma * diameter);
                const double nbeta = fe.normal_vector(q) * angles.at(d);
                if (nbeta > 0)
                  result(i) += dx * gammaa * nbeta
                               * fe.shape_value_component(i,q,d)
                               * input[d][q];
              }
        }
    }

  } // end NAMESPACE = Transport
} // end NAMESPACE = LocalIntegrators

#endif // TRANSPORT_H
