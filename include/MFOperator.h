#ifndef MFOPERATOR_H
#define MFOPERATOR_H

#include <deal.II/base/std_cxx11/function.h>
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

#include <GenericLinearAlgebra.h>
#include <Integrators.h>
#include <integration_loop.h>

#include <MGMatrixSimpleMapped.h>

namespace
{
  template <int dim, bool is_system_matrix>
  struct CellIterator;

  template <>
  struct CellIterator<2, true>
  {
    typedef typename dealii::DoFHandler<2>::active_cell_iterator type;
  };

  template <>
  struct CellIterator<3, true>
  {
    typedef typename dealii::DoFHandler<3>::active_cell_iterator type;
  };

  template <>
  struct CellIterator<2, false>
  {
    typedef typename dealii::DoFHandler<2>::level_cell_iterator type;
  };

  template <>
  struct CellIterator<3, false>
  {
    typedef typename dealii::DoFHandler<3>::level_cell_iterator type;
  };
}


template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix=false>
class MFOperator final: public dealii::Subscriptor
{
public:
  MFOperator () ;
  ~MFOperator () ;
  MFOperator (const MFOperator &operator_);
  MFOperator &operator = (const MFOperator &) = delete;

  void reinit (const dealii::DoFHandler<dim> *dof_handler_,
               const dealii::MappingQ1<dim> *mapping_,
               const dealii::ConstraintMatrix *constraints,
               const dealii::MGConstrainedDoFs *mg_constrained_dofs,
               const MPI_Comm &mpi_communicator_,
               const unsigned int level_ = dealii::numbers::invalid_unsigned_int);

  void set_cell_range (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_);

  void set_timer (dealii::TimerOutput &timer_);

  void build_coarse_matrix();

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
    return coarse_matrix;
  }

  unsigned int m() const
  {
    return dof_handler->n_dofs(level);
  }

  unsigned int n() const
  {
    return dof_handler->n_dofs(level);
  }

  typedef LA::MPI::SparseMatrixSizeType                         size_type ;
  typedef typename CellIterator<dim, is_system_matrix>::type    cell_iterator;

private:
  unsigned int                                        level;
  const dealii::DoFHandler<dim>                       *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::MappingQ1<dim>                        *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  const dealii::MGConstrainedDoFs                     *mg_constrained_dofs;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  dealii::SparsityPattern                             sp;
  LA::MPI::SparseMatrix                               coarse_matrix;
  MatrixIntegrator<dim,same_diagonal>                 matrix_integrator;
  ResidualIntegrator<dim>                             residual_integrator;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_src;
  mutable LA::MPI::Vector                             ghosted_dst;
  MPI_Comm                                            mpi_communicator;
  dealii::TimerOutput                                 *timer;
  std::vector<std::vector<cell_iterator> >            colored_iterators;
  std::vector<cell_iterator>                          cell_range;
  bool                                                use_cell_range;
};

#ifdef HEADER_IMPLEMENTATION
#include <MFOperator.cc>
#endif

#endif // MFOPERATOR_H
