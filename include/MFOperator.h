#ifndef MFOPERATOR_H
#define MFOPERATOR_H

#include <deal.II/base/timer.h>
#include <deal.II/base/std_cxx11/function.h>
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

template <int dim, int fe_degree, bool same_diagonal>
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
               const MPI_Comm &mpi_communicator_,
               const unsigned int level_ = dealii::numbers::invalid_unsigned_int);

  void set_cell_range (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_);

  void set_timer (dealii::TimerOutput &timer_);

  void build_coarse_matrix();

  void build_matrix
  (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range = std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator>(),
   const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain = std::vector<typename dealii::types::global_dof_index>(),
   const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique = std::map<dealii::types::global_dof_index, unsigned int> ());

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

  const dealii::FullMatrix<double> &get_matrix() const
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
    return level==0?coarse_matrix(i,j):matrix(i,j);
  }

  double el(const size_type i,const size_type j) const
  {
    return level==0?coarse_matrix.el(i,j):matrix(i,j);
  }

private:
  unsigned int                                        level;
  const dealii::DoFHandler<dim>                       *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::MappingQ1<dim>                        *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  dealii::SparsityPattern                             sp;
  dealii::FullMatrix<double>                          matrix;
  LA::MPI::SparseMatrix                               coarse_matrix;
  MatrixIntegrator<dim,same_diagonal>                 matrix_integrator;
  ResidualIntegrator<dim>                             residual_integrator;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_src;
  MPI_Comm                                            mpi_communicator;
  dealii::TimerOutput                                 *timer;
  const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> *cell_range;
  bool                                                use_cell_range;
};

#ifdef HEADER_IMPLEMENTATION
#include <MFOperator.cc>
#endif

#endif // MFOPERATOR_H
