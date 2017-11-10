#ifndef RHS_H
#define RHS_H

#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include <GenericLinearAlgebra.h>
#include <FiniteElement.h>
#include <Dofs.h>
#include <ResidualSimpleConstraints.h>
#include <Integrators.h>

template <int dim>
class RHS final
{
 public:
  RHS (FiniteElement<dim> & fe_,Dofs<dim> & dofs_) ;
  
  void assemble(const LA::MPI::Vector & solution);
  
  FiniteElement<dim> & fe;
  Dofs<dim> & dofs;
  
  LA::MPI::Vector            right_hand_side;
};

#ifdef HEADER_IMPLEMENTATION
#include <RHS.cc>
#endif

#endif // RHS_H

