#ifndef MFOPERATOR_H
#define MFOPERATOR_H

#ifndef MATRIXFREE

#include <deal.II/base/timer.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <GenericLinearAlgebra.h>
#include <Integrators.h>
#include <integration_loop.h>
#include <MGMatrixSimpleMapped.h>

#include <functional>


template <int dim, int fe_degree, typename number=double>
class MFOperator final: public dealii::Subscriptor
{
public:
  typedef double value_type ;
  typedef LA::MPI::SparseMatrixSizeType                         size_type ;
  typedef typename dealii::DoFHandler<dim>::level_cell_iterator level_cell_iterator ;
  typedef typename dealii::LAPACKFullMatrix<double>             LAPACKMatrix ;

  MFOperator () ;
  ~MFOperator () ;
  MFOperator (const MFOperator &operator_);
  MFOperator &operator = (const MFOperator &) = delete;

  void reinit (const dealii::DoFHandler<dim>  *dof_handler_,
               const dealii::Mapping<dim>     *mapping_,
               const dealii::ConstraintMatrix *constraints,
               const MPI_Comm                 &mpi_communicator_,
               const unsigned                 int level_ = dealii::numbers::invalid_unsigned_int,
               LA::MPI::Vector                solution_ = LA::MPI::Vector {});

  void set_cell_range (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_);

  void set_timer (dealii::TimerOutput &timer_);

  // TODO parallel::distributed case
#if PARALLEL_LA < 3
  void build_coarse_matrix();
#endif

  void clear () ;

  void vmult (LA::MPI::Vector &dst,
              const LA::MPI::Vector &src) const ;
  void Tvmult (LA::MPI::Vector &dst,
               const LA::MPI::Vector &src) const ;
  void vmult_add (LA::MPI::Vector &dst,
                  const LA::MPI::Vector &src) const ;
  void Tvmult_add (LA::MPI::Vector &dst,
                   const LA::MPI::Vector &src) const ;

  // TODO parallel::distributed case
#if PARALLEL_LA < 3
  const LA::MPI::SparseMatrix &get_coarse_matrix() const
  {
    return coarse_matrix;
  }
#endif

  unsigned int m() const
  {
    return dof_handler->n_dofs(level);
  }

  unsigned int n() const
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
  const dealii::MGConstrainedDoFs                     *mg_constrained_dofs;
  dealii::TimerOutput                                 *timer;

  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_src;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_solution;
  const std::vector<level_cell_iterator>              *cell_range;
  bool                                                use_cell_range;
  std::vector<std::vector<level_cell_iterator> >      colored_iterators;
  ResidualIntegrator<dim>                             residual_integrator;

#if PARALLEL_LA < 3
  dealii::SparsityPattern                             sp;
  LA::MPI::SparseMatrix                               coarse_matrix;
  MatrixIntegrator<dim>                               matrix_integrator;
#endif // PARALLEL_LA
};

#endif // MATRIXFREE OFF

#ifdef HEADER_IMPLEMENTATION
#include <MFOperator.cc>
#endif

#endif // MFOPERATOR_H
