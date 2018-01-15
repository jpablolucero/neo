#ifndef MFOPERATOR_H
#define MFOPERATOR_H

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

#include <Integrators.h>
#include <integration_loop.h>
#include <MGMatrixSimpleMapped.h>
#include <DDHandler.h>

#include <functional>


template <int dim, int fe_degree, typename number=double, typename VectorType=dealii::parallel::distributed::Vector<double> >
class MFOperator final: public dealii::Subscriptor
{
public:
  typedef double value_type ;
  typedef dealii::SparseMatrix<double>::size_type               size_type ;
  typedef typename dealii::DoFHandler<dim>::level_cell_iterator level_cell_iterator ;
  typedef typename dealii::LAPACKFullMatrix<double>             LAPACKMatrix ;

  MFOperator () ;
  ~MFOperator () ;
  MFOperator (const MFOperator &operator_);
  MFOperator &operator = (const MFOperator &) = delete;

  void reinit (const dealii::DoFHandler<dim>  *dof_handler_,
               const dealii::Mapping<dim>     *mapping_,
               const dealii::ConstraintMatrix *constraints,
               const unsigned                 int level_,
               VectorType                     &solution_);

  void set_cell_range (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_);
  void unset_cell_range ();
  void set_subdomain (unsigned int subdomain_idx_);
  void build_coarse_matrix();

  void clear ()
  {
    ghosted_src[level] = 0.;
  }

  void vmult (VectorType &dst, const VectorType &src) const ;
  void Tvmult (VectorType &dst, const VectorType &src) const ;
  void vmult_add (VectorType &dst, const VectorType &src) const ;
  void Tvmult_add (VectorType &dst, const VectorType &src) const ;
  void vmult (dealii::Vector<double> &dst, const dealii::Vector<double> &src) const ;
  void vmult_add (dealii::Vector<double> &dst, const dealii::Vector<double> &src) const ;

  const dealii::SparseMatrix<double> &get_coarse_matrix() const
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

private:
  unsigned int                                        level;
  unsigned int                                        subdomain_idx;
  const dealii::DoFHandler<dim>                       *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::Mapping<dim>                          *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  const dealii::MGConstrainedDoFs                     *mg_constrained_dofs;
  mutable VectorType                                  *solution;
  DGDDHandlerCell<dim>                                ddh;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> zero_info_box;
  mutable dealii::MGLevelObject<VectorType>           ghosted_src;
  mutable dealii::MGLevelObject<VectorType>           zero_src;
  mutable dealii::MGLevelObject<VectorType>           zero_solution;
  mutable dealii::MGLevelObject<VectorType>           ghosted_solution;
  const std::vector<level_cell_iterator>              *cell_range;
  bool                                                use_cell_range;
  std::vector<std::vector<level_cell_iterator> >      colored_iterators;
  const std::vector<level_cell_iterator> *            selected_iterators;
  ResidualIntegrator<dim>                             residual_integrator;
  dealii::SparsityPattern                             sp;
  dealii::SparseMatrix<double>                        coarse_matrix;
  MatrixIntegrator<dim>                               matrix_integrator;
};

#include <MFOperator.templates.h>

#endif // MFOPERATOR_H
