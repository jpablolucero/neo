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

#include <BlockIntegrators.h>

template <int dim, int fe_degree, bool same_diagonal>
class LaplaceOperator : public dealii::Subscriptor
{
public:
  LaplaceOperator () ; 

  ~LaplaceOperator () ;
  
  void reinit (
      dealii::DoFHandler<dim>* dof_handler_,
      const dealii::MappingQ1<dim>* mapping_ =
      dealii::StaticMappingQ1<dim>::mapping,
      const unsigned int level_ = dealii::numbers::invalid_unsigned_int);
  
  void build_matrix () ;

  void clear () ;

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
  typedef typename dealii::SparseMatrix<double>::const_iterator const_iterator ;
  typedef typename dealii::SparseMatrix<double>::size_type size_type ;
  const_iterator begin (const size_type r) const {return matrix.begin(r) ;};
  const_iterator end (const size_type r) const {return matrix.end(r) ;};
  double operator()(const size_type i,const size_type j) const
  { return matrix(i,j);};
 private:
  unsigned int level ;
  dealii::DoFHandler<dim> * dof_handler; 
  const dealii::MappingQ1<dim> *  mapping;
  dealii::MeshWorker::DoFInfo<dim> * dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  dealii::SparsityPattern sparsity ;
  dealii::SparseMatrix<double> matrix ;
  BMatrixIntegrator<dim,same_diagonal> matrix_integrator ;
  BResidualIntegrator<dim> residual_integrator ;
};

#endif // LAPLACEOPERATOR_H
