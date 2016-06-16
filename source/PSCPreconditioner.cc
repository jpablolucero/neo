#include <PSCPreconditioner.h>

namespace implementation
{
  namespace WorkStream
  {
    template <int dim, typename VectorType, class number, bool same_diagonal>
    class Copy
    {
    public:
      dealii::Vector<number>  local_solution;
      unsigned int subdomain_idx;
      VectorType *dst;

      std::shared_ptr<DDHandlerBase<dim> > ddh;
    };

    template <int dim, typename VectorType, class number, bool same_diagonal>
    class Scratch
    {
    public:
      const std::vector<dealii::FullMatrix<double>* > *local_inverses;

      dealii::Vector<number>  local_src;
      const VectorType *src;
    };

    template <int dim, typename VectorType, class number, bool same_diagonal>
    void assemble(const Copy<dim, VectorType, number, same_diagonal> &copy)
    {
      // write back to global vector
      copy.ddh->prolongate_add(*(copy.dst),
                               copy.local_solution,
                               copy.subdomain_idx);
    }

    template <int dim, typename VectorType, class number, bool same_diagonal>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, VectorType, number, same_diagonal> &scratch, Copy<dim, VectorType, number, same_diagonal> &copy)
    {
      const unsigned int subdomain_idx = *iterator;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);

      copy.subdomain_idx = subdomain_idx;
      ddh.reinit(scratch.local_src, subdomain_idx);
      ddh.reinit(copy.local_solution, subdomain_idx);
      // get local contributions and copy to dealii vector
      ddh.restrict_add(scratch.local_src, *(scratch.src), subdomain_idx);
      (*(scratch.local_inverses))[subdomain_idx]->vmult(copy.local_solution,
                                                        scratch.local_src);
    }
  }
}

template <int dim, typename VectorType, class number, bool same_diagonal>
PSCPreconditioner<dim, VectorType, number, same_diagonal>::PSCPreconditioner()
{}

// PSCPreconditioner<...>::initialize(...) is directly implemented in the header

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::clear()
{}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::vmult (VectorType &dst,
								       const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::Tvmult (VectorType &/*dst*/,
									const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::vmult_add (VectorType &dst,
									   const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);

  {
    implementation::WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
    copy_sample.dst = &dst;
    copy_sample.ddh = ddh;

    implementation::WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
    scratch_sample.src = &src;
    scratch_sample.local_inverses = &patch_inverses;

    dealii::WorkStream::run(ddh->colorized_iterators(),
                            implementation::WorkStream::work<dim, VectorType, number, same_diagonal>,
                            implementation::WorkStream::assemble<dim, VectorType, number, same_diagonal>,
                            scratch_sample, copy_sample);
  }

  dst *= data.weight;
  timer->leave_subsection();
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::Tvmult_add (VectorType &/*dst*/,
									    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::build_matrix
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
 const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
 const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
 dealii::FullMatrix<double> &matrix)
{

  dealii::MGLevelObject<dealii::FullMatrix<double> > mg_matrix ;
  mg_matrix.resize(level,level);

  mg_matrix[level] = std::move(dealii::FullMatrix<double>(global_dofs_on_subdomain.size()));

  Assembler::MGMatrixSimpleMapped<dealii::FullMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(constraints);
#endif
  assembler.initialize(all_to_unique);

  //now assemble everything
  dealii::MeshWorker::LoopControl lctrl;
  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
  lctrl.ghost_cells = true;
  //TODO possibly colorize iterators, assume thread-safety for the moment
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > colored_iterators(1, cell_range);


  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler,lctrl, colored_iterators[0]);

  matrix.copy_from(mg_matrix[level]);
}

#include "PSCPreconditioner.inst"
