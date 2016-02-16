#ifndef LAPLACEOPERATOR_H
#define LAPLACEOPERATOR_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <MatrixIntegrator.h>

template <int dim, int fe_degree, typename number>
class LaplaceOperator : public dealii::Subscriptor
{
public:
  LaplaceOperator (const dealii::Triangulation<dim>& triangulation_,
		   const dealii::MappingQ1<dim>&  mapping_,
		   const dealii::FE_DGQ<dim>&  fe_,
		   const dealii::DoFHandler<dim>&  dof_handler_); 

  void vmult (dealii::Vector<number> &dst,
              const dealii::Vector<number> &src) const;
  void Tvmult (dealii::Vector<number> &dst,
               const dealii::Vector<number> &src) const;
  void vmult_add (dealii::Vector<number> &dst,
                  const dealii::Vector<number> &src) const;
  void Tvmult_add (dealii::Vector<number> &dst,
                   const dealii::Vector<number> &src) const;

 private:
  const dealii::Triangulation<dim>& triangulation;
  const dealii::MappingQ1<dim>&  mapping;
  const dealii::FE_DGQ<dim>&  fe;
  const dealii::DoFHandler<dim>&  dof_handler; 
};

#endif // LAPLACEOPERATOR_H
