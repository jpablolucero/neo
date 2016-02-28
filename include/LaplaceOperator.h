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
  LaplaceOperator () ; 

  ~LaplaceOperator () ;
  
  void reinit (dealii::DoFHandler<dim> * dof_handler_,
	       dealii::FE_DGQ<dim> * fe_,
	       dealii::Triangulation<dim> * triangulation_,
	       const dealii::MappingQ1<dim> * mapping_);

  void vmult (dealii::Vector<number> &dst,
	      const dealii::Vector<number> &src) const ;
  void Tvmult (dealii::Vector<number> &dst,
	       const dealii::Vector<number> &src) const ;
  void vmult_add (dealii::Vector<number> &dst,
		  const dealii::Vector<number> &src) const ;
  void Tvmult_add (dealii::Vector<number> &dst,
		   const dealii::Vector<number> &src) const ;

 private:
  dealii::DoFHandler<dim> * dof_handler; 
  dealii::FE_DGQ<dim> * fe;
  dealii::Triangulation<dim> * triangulation;
  const dealii::MappingQ1<dim> *  mapping;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  MatrixIntegrator<dim> matrix_integrator ;
};

#endif // LAPLACEOPERATOR_H
