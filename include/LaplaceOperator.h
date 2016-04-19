#ifndef LAPLACEOPERATOR_H
#define LAPLACEOPERATOR_H

#include <deal.II/base/timer.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <generic_linear_algebra.h>
#include <Integrators.h>


template <int dim, int fe_degree, bool same_diagonal>
class LaplaceOperator : public dealii::Subscriptor
{
public:
  LaplaceOperator () ;

  ~LaplaceOperator () ;

  void reinit (dealii::DoFHandler<dim> *dof_handler_,
               const dealii::MappingQ1<dim> *mapping_,
               const dealii::ConstraintMatrix *constraints,
               const MPI_Comm &mpi_communicator_,
               const unsigned int level_ = dealii::numbers::invalid_unsigned_int);

  void set_timer (dealii::TimerOutput &timer_);

  void build_matrix () ;

  void clear () ;

  void vmult (LA::MPI::Vector &dst,
              const LA::MPI::Vector &src) const ;
  void Tvmult (LA::MPI::Vector &dst,
               const LA::MPI::Vector &src) const ;
  void vmult_add (LA::MPI::Vector &dst,
                  const LA::MPI::Vector &src) const ;
  void Tvmult_add (LA::MPI::Vector &dst,
                   const LA::MPI::Vector &src) const ;

  typedef double value_type ;

  const LA::MPI::SparseMatrix &get_coarse_matrix() const
  {
    return matrix;
  }

  unsigned int m() const
  {
    return dof_handler->n_dofs(level);
  }

  unsigned int n() const
  {
    return dof_handler->n_dofs(level);
  }

  typedef LA::MPI::SparseMatrixSizeType      size_type ;

  double operator()(const size_type i,const size_type j) const
  {
    return matrix(i,j);
  }

private:
  unsigned int                                        level;
  dealii::DoFHandler<dim>                             *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::MappingQ1<dim>                        *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
#if PARALLEL_LA == 0
  dealii::SparsityPattern                             sp;
#endif
  LA::MPI::SparseMatrix                               matrix;
  MatrixIntegrator<dim,same_diagonal>                 matrix_integrator;
  ResidualIntegrator<dim>                             residual_integrator;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_src;
  MPI_Comm                                            mpi_communicator;
  dealii::TimerOutput                                 *timer;
};

#endif // LAPLACEOPERATOR_H
