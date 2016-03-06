#ifndef LAPLACEOPERATOR_H
#define LAPLACEOPERATOR_H

#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <MatrixIntegrator.h>
#include <MatrixIntegratorMG.h>

template <int dim, int fe_degree>
class LaplaceOperator : public dealii::Subscriptor
{
public:
  LaplaceOperator () ; 

  ~LaplaceOperator () ;
  
  void reinit (dealii::DoFHandler<dim> * dof_handler_,
	       dealii::FE_DGQ<dim> * fe_,
	       dealii::Triangulation<dim> * triangulation_,
	       const dealii::MappingQ1<dim> * mapping_,
	       unsigned int level_=0,
	       bool level_matrix_=false);

  void build_matrix () ;

  void vmult (dealii::Vector<double> &dst,
	      const dealii::Vector<double> &src) const ;
  void Tvmult (dealii::Vector<double> &dst,
	       const dealii::Vector<double> &src) const ;
  void vmult_add (dealii::Vector<double> &dst,
		  const dealii::Vector<double> &src) const ;
  void Tvmult_add (dealii::Vector<double> &dst,
		   const dealii::Vector<double> &src) const ;

  typedef double value_type ;
  unsigned int m() const {return dof_handler->n_dofs(level);};
  unsigned int n() const {return dof_handler->n_dofs(level);};
  typedef typename dealii::FullMatrix<double>::const_iterator const_iterator ;
  typedef typename dealii::FullMatrix<double>::size_type size_type ;
  const_iterator begin (const size_type r) const {return matrix.begin(r) ;};
  const_iterator end (const size_type r) const {return matrix.end(r) ;};
  double operator()(const size_type i,const size_type j) const
  { return matrix(i,j);};
 private:
  unsigned int level ;
  bool level_matrix ;
  dealii::DoFHandler<dim> * dof_handler; 
  dealii::FE_DGQ<dim> * fe;
  dealii::Triangulation<dim> * triangulation;
  const dealii::MappingQ1<dim> *  mapping;
  dealii::MeshWorker::DoFInfo<dim> * dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  dealii::FullMatrix<double> matrix ;
  MatrixIntegrator<dim> matrix_integrator ;
  MatrixIntegratorMG<dim> matrix_integrator_mg ;
};

#endif // LAPLACEOPERATOR_H
