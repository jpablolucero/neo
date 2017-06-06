#ifndef PSCPRECONDITIONER_H
#define PSCPRECONDITIONER_H

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

#include <functional>

#include <GenericLinearAlgebra.h>
#include <DDHandler.h>
#include <MFOperator.h>
#include <MfreeOperator.h>
#include <PSCIntegrators.h>
#include <integration_loop.h>
#include <MGMatrixSimpleMapped.h>

template <int dim, typename SystemMatrixType, typename VectorType=LA::MPI::Vector, typename number=double, bool same_diagonal=false>
class PSCPreconditioner final
{
public:
  typedef typename dealii::LAPACKFullMatrix<double> LAPACKMatrix;
  typedef typename dealii::FullMatrix<double> Matrix;
  class AdditionalData;

  PSCPreconditioner();
  ~PSCPreconditioner();
  PSCPreconditioner (const PSCPreconditioner &) = delete ;
  PSCPreconditioner &operator = (const PSCPreconditioner &) = delete;

  void initialize(const SystemMatrixType &system_matrix_,
                  const AdditionalData &data);
  void clear();

  void vmult(VectorType &dst, const VectorType &src) const;

  void Tvmult(VectorType &dst, const VectorType &src) const;

  void vmult_add(VectorType &dst, const VectorType &src) const;

  void Tvmult_add(VectorType &dst, const VectorType &src) const;

protected:
  AdditionalData data;

private:
  void build_matrix
  (const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
   const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
   const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
   dealii::LAPACKFullMatrix<double> &matrix);

  std::vector<std::shared_ptr<LAPACKMatrix> > patch_inverses;

  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;

  dealii::MGLevelObject<LA::MPI::Vector>              ghosted_solution;
  PSCMatrixIntegrator<dim>                            matrix_integrator;
  mutable LA::MPI::Vector                             ghosted_src;
#if PARALLEL_LA==3
  mutable LA::MPI::Vector                             ghosted_dst;
#endif

  unsigned int level;
  std::shared_ptr<DDHandlerBase<dim> > ddh;
  const SystemMatrixType *system_matrix;
};

template <int dim, typename SystemMatrixType, typename VectorType, class number, bool same_diagonal>
class PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::AdditionalData
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
  double relaxation;
  double tol;
  const dealii::Mapping<dim> *mapping;
  LA::MPI::Vector *solution;

  bool use_dictionary;
  enum PatchType
  {
    cell_patches,
    vertex_patches
  };
  PatchType patch_type;

  dealii::MGConstrainedDoFs  mg_constrained_dofs;
};

#include <PSCPreconditioner.h.templates>

#endif // PSCPRECONDITIONER_H
