#ifndef NLPSCPRECONDITIONER_H
#define NLPSCPRECONDITIONER_H

#include <deal.II/base/timer.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/meshworker/vector_selector.h>
#include <deal.II/meshworker/vector_selector.templates.h>

#include <functional>
#include <algorithm>

#include <DDHandler.h>
#include <MFOperator.h>
#include <Integrators.h>
#include <integration_loop.h>
#include <MGMatrixSimpleMapped.h>

template <int dim, typename SystemMatrixType, typename VectorType=dealii::LinearAlgebra::distributed::Vector<double>,
	  typename number=double, bool same_diagonal=false>
class NLPSCPreconditioner final
{
public:
  typedef typename dealii::LAPACKFullMatrix<double> LAPACKMatrix;
  typedef typename dealii::FullMatrix<double> Matrix;
  class AdditionalData;

  NLPSCPreconditioner();
  ~NLPSCPreconditioner();
  NLPSCPreconditioner (const NLPSCPreconditioner &) = delete ;
  NLPSCPreconditioner &operator = (const NLPSCPreconditioner &) = delete;

  void initialize(const SystemMatrixType &system_matrix_,const AdditionalData &data);
  void vmult(VectorType &dst, const VectorType &src) const;
  void Tvmult(VectorType &dst, const VectorType &src) const;
  void vmult_add(VectorType &dst, const VectorType &src) const;
  void Tvmult_add(VectorType &dst, const VectorType &src) const;

protected:
  AdditionalData data;

private:
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MGLevelObject<VectorType>           ghosted_solution;
  MatrixIntegrator<dim>                               matrix_integrator;
  unsigned int level;
  std::shared_ptr<DDHandlerBase<dim> > ddh;
  const SystemMatrixType *system_matrix;
};

template <int dim, typename SystemMatrixType, typename VectorType, class number, bool same_diagonal>
class NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::AdditionalData
{
public:
  AdditionalData() : dof_handler(0),
    level(-1),
    relaxation(1.0),
    tol(0.),
    mapping(0),
    use_dictionary(false),
    patch_type(cell_patches)
  {}
  dealii::DoFHandler<dim> *dof_handler;
  unsigned int level;
  unsigned int n_levels;
  double relaxation;
  double tol;
  const dealii::Mapping<dim> *mapping;
  const dealii::SparseMatrix<number> *coarse_matrix;
  bool use_dictionary;
  enum PatchType
  {
    cell_patches,
    vertex_patches
  };
  PatchType patch_type;
  enum SmootherType
  {
    additive,
  };
  SmootherType smoother_type;
  // dealii::MGConstrainedDoFs  mg_constrained_dofs;
};

#include <NLPSCPreconditioner.templates.h>

#endif // NLPSCPRECONDITIONER_H
