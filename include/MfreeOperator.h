#ifndef MFREEOPERATOR_H
#define MFREEOPERATOR_H

#ifdef MATRIXFREE

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <GenericLinearAlgebra.h>
#include <Integrators.h>



template <int dim, int fe_degree, int n_q_points_1d=fe_degree+1, typename number=double>
class MfreeOperator final: public dealii::Subscriptor
{
public:
  typedef double value_type ;
  typedef LA::MPI::SparseMatrixSizeType                         size_type ;

  /*
   *  Construction & Initialization
   */
  MfreeOperator () ;

  ~MfreeOperator () ;

  MfreeOperator (const MfreeOperator &operator_) = delete ;

  MfreeOperator &operator = (const MfreeOperator &) = delete ;

  void
  reinit (const dealii::DoFHandler<dim> *dof_handler_,
          const dealii::Mapping<dim> *mapping_,
          const dealii::ConstraintMatrix *constraints,
          const MPI_Comm &mpi_communicator_,
          const unsigned int level_ = dealii::numbers::invalid_unsigned_int) ;

  /*
   *  Vector multiplication
   */
  void
  vmult (LA::MPI::Vector &dst,
         const LA::MPI::Vector &src) const ;
  void
  vmult_add (LA::MPI::Vector &dst,
             const LA::MPI::Vector &src) const ;
  void
  Tvmult (LA::MPI::Vector &dst,
          const LA::MPI::Vector &src) const ;
  void
  Tvmult_add (LA::MPI::Vector &dst,
              const LA::MPI::Vector &src) const ;

  /*
   *  Utilities
   */
  void
  initialize_dof_vector (LA::MPI::Vector &vector) const;

  void
  set_timer (dealii::TimerOutput &timer_) ;

  /*
   *  General information
   */
  unsigned int
  m() const
  {
    return dof_handler->n_dofs(level);
  }

  unsigned int
  n() const
  {
    return dof_handler->n_dofs(level);
  }

private:
  unsigned int                                        level;
  const dealii::DoFHandler<dim>                       *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::Mapping<dim>                          *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  MPI_Comm                                            mpi_communicator;
  dealii::TimerOutput                                 *timer;

  dealii::MatrixFree<dim,double>                      data;
  MFIntegrator<dim,fe_degree,n_q_points_1d,1,double>  mf_integrator;
};

#ifdef HEADER_IMPLEMENTATION
#include <MFOperator.cc>
#endif

#endif // MATRIXFREE
#endif // MFREEOPERATOR_H
