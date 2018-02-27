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
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <functional>

#include <DDHandler.h>
#include <MFOperator.h>
#include <integration_loop.h>
#include <MGMatrixSimpleMapped.h>

template <int dim,typename SystemMatrixType,typename LocalMatrixType=SystemMatrixType,
	  typename VectorType=dealii::parallel::distributed::Vector<double>,typename number=double,bool same_diagonal=false>
class PSCPreconditioner final
{
public:
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

  template <typename M=LocalMatrixType>
    typename std::enable_if<std::is_same<M,dealii::LAPACKFullMatrix<number> >::value >::type
    configure_local_matrices ();
  template <typename M=LocalMatrixType>
    typename std::enable_if<std::is_same<M,dealii::TrilinosWrappers::SparseMatrix>::value >::type
    configure_local_matrices ();
  template <typename M=LocalMatrixType>
    typename std::enable_if<std::is_same<M,SystemMatrixType>::value >::type
    configure_local_matrices ();
  
  void build_matrix(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
		    const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
		    const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
		    dealii::LAPACKFullMatrix<number> &matrix);
  
  void build_matrix(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
		    const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
		    const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
		    dealii::TrilinosWrappers::SparseMatrix &matrix);

  void add_cell_ordering(dealii::Tensor<1,dim> dir) ;
  std::vector<std::shared_ptr<LocalMatrixType> > patch_matrices;
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  dealii::MGLevelObject<VectorType >                  ghosted_solution;
  MatrixIntegrator<dim>                               matrix_integrator;
  mutable VectorType                                  ghosted_src;
  mutable VectorType                                  ghosted_dst;
  unsigned int level;
  std::shared_ptr<DDHandlerBase<dim> > ddh;
  const SystemMatrixType *system_matrix;
  typedef std::vector<unsigned int>::const_iterator iterator;
  std::vector<std::vector<std::vector<std::vector<iterator> > > > ordered_iterators ;
  std::vector<std::vector<int> > ordered_gens ;
  std::vector<std::vector<unsigned int> > downstream_outbox ;
  unsigned int global_last_gen = 0 ;
};

template <int dim,typename SystemMatrixType,typename LocalMatrixType,typename VectorType,typename number,bool same_diagonal>
class PSCPreconditioner<dim,SystemMatrixType,LocalMatrixType,VectorType,number,same_diagonal>::AdditionalData
{
public:
  AdditionalData() : dof_handler(0),
    level(-1),
    relaxation(1.0),
    tol(0.),
    mapping(0),
    use_dictionary(false),
    patch_type(cell_patches)
  {
    dirs.resize(1);
    if (dim == 2)
      {
	dirs[0][0] =  1. ; dirs[0][1] =  1. ; 
      }
    else if (dim == 3)
      {
	dirs[0][0] =  1. ; dirs[0][1] =  1. ; dirs[0][2] =  1. ; 
      }
  }
  void set_fullsweep()
  {
    dirs.resize(std::pow(2,dim));
    if (dim == 2)
      {
	dirs[0][0] =  1. ; dirs[0][1] =  1. ; 
	dirs[1][0] = -1. ; dirs[1][1] =  1. ; 
	dirs[2][0] =  1. ; dirs[2][1] = -1. ; 
	dirs[3][0] = -1. ; dirs[3][1] = -1. ; 
      }
    else if (dim == 3)
      {
	dirs[0][0] =  1. ; dirs[0][1] =  1. ; dirs[0][2] =  1. ; 
	dirs[1][0] = -1. ; dirs[1][1] =  1. ; dirs[1][2] =  1. ; 
	dirs[2][0] =  1. ; dirs[2][1] = -1. ; dirs[2][2] =  1. ; 
	dirs[3][0] =  1. ; dirs[3][1] =  1. ; dirs[3][2] = -1. ; 
	dirs[4][0] =  1. ; dirs[4][1] = -1. ; dirs[4][2] = -1. ; 
	dirs[5][0] = -1. ; dirs[5][1] = -1. ; dirs[5][2] =  1. ; 
	dirs[6][0] = -1. ; dirs[6][1] =  1. ; dirs[6][2] = -1. ; 
	dirs[7][0] = -1. ; dirs[7][1] = -1. ; dirs[7][2] = -1. ;
      }
  }

  dealii::DoFHandler<dim> *dof_handler;
  unsigned int level;
  unsigned int n_levels;
  double relaxation;
  double tol;
  const dealii::Mapping<dim> *mapping;
  VectorType *solution;
  const dealii::SparseMatrix<double> *coarse_matrix;

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
    additive_with_coarse,
    hybrid,
    multiplicative
  };
  
  SmootherType smoother_type;

  // dealii::MGConstrainedDoFs  mg_constrained_dofs;

  std::vector<dealii::Tensor<1,dim> > dirs;  

};

#include <PSCPreconditioner.templates.h>

#endif // PSCPRECONDITIONER_H
