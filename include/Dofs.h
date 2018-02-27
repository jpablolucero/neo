#ifndef DOFS_H
#define DOFS_H

#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <Mesh.h>
#include <FiniteElement.h>
#include <EquationData.h>

template <int dim>
class Dofs final
{
public:
  Dofs (Mesh<dim> & mesh_, FiniteElement<dim> & fe_) ;
  void setup();
  
  Mesh<dim> & mesh ;
  FiniteElement<dim> & fe ;
  dealii::DoFHandler<dim>  dof_handler;
  dealii::IndexSet         locally_owned_dofs;
  dealii::IndexSet         locally_relevant_dofs;

  dealii::ConstraintMatrix constraints;
  ReferenceFunction<dim>   reference_function;
  Boundaries<dim>          boundaries;
};

#ifdef HEADER_IMPLEMENTATION
#include <Dofs.cc>
#endif

#endif // DOFS_H
