#ifndef DOFS_H
#define DOFS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

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
  dealii::DoFHandler<dim>    dof_handler;
  dealii::IndexSet           locally_owned_dofs;
  dealii::IndexSet           locally_relevant_dofs;
  dealii::ConstraintMatrix                   constraints;
  std::shared_ptr<dealii::MGConstrainedDoFs> mg_constrained_dofs;  
  ReferenceFunction<dim>                     reference_function;
};

#ifdef HEADER_IMPLEMENTATION
#include <Dofs.cc>
#endif

#endif // DOFS_H
