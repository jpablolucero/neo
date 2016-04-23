#include <PSCPreconditioner.h>

namespace implementation
{
  namespace WorkStream
  {
    template <int dim, class IT>
    std::vector<dealii::types::global_dof_index>
    dd_conflict_indices(const DDHandlerBase<dim> &ddh,
                        const IT &iterator)
    {
      const unsigned int subdomain_idx = *iterator;
      return ddh.global_dof_indices_on_subdomain(subdomain_idx);
    }

    template <int dim, typename VectorType, class number>
    class Copy
    {
    public:
      dealii::Vector<number> local_solution;
      unsigned int subdomain_idx;
      VectorType *dst;

      const DDHandlerBase<dim> *ddh;
    };

    template <int dim, typename VectorType, class number>
    class Scratch
    {
    public:
      dealii::Vector<number> local_src;

      const VectorType *src;
      const std::vector<const dealii::FullMatrix<double>* > *local_inverses;
    };

    template <int dim, typename VectorType, class number>
    void assemble(const Copy<dim, VectorType, number> &copy)
    {
      // write back to global vector
      copy.ddh->prolongate_add(*(copy.dst),
                               copy.local_solution,
                               copy.subdomain_idx);
    }

    template <int dim, typename VectorType, class number>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, VectorType, number> &scratch, Copy<dim, VectorType, number> &copy)
    {
      unsigned int subdomain_idx = *iterator;
      copy.subdomain_idx = subdomain_idx;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);
      ddh.reinit(scratch.local_src, subdomain_idx);
      ddh.reinit(copy.local_solution, subdomain_idx);
      // get local contributions and copy to dealii vector
      ddh.restrict_add(scratch.local_src, *(scratch.src), subdomain_idx);
      (*scratch.local_inverses)[subdomain_idx]->vmult(copy.local_solution,
                                                      scratch.local_src);
    }

    template <int dim, typename VectorType, class number>
    void parallel_loop(VectorType *dst,
                       const VectorType *src,
                       const DDHandlerBase<dim> *ddh,
                       const std::vector<const dealii::FullMatrix<double>* > *
                       local_inverses)
    {
      Copy<dim, VectorType, number> copy_sample;
      copy_sample.dst = dst;
      copy_sample.ddh = ddh;

      Scratch<dim, VectorType, number> scratch_sample;
      scratch_sample.src = src;
      scratch_sample.local_inverses = local_inverses;

      const unsigned int queue = 2 * dealii::MultithreadInfo::n_threads();
      const unsigned int chunk_size = 1;

      dealii::WorkStream::run(ddh->colorized_iterators(),
                              work<dim, VectorType, number>,
                              assemble<dim, VectorType, number>,
                              scratch_sample, copy_sample,
                              queue,
                              chunk_size);
    }
  }
}

template <int dim, typename VectorType, class number>
PSCPreconditioner<dim, VectorType, number>::PSCPreconditioner()
{}


template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::vmult (VectorType &dst,
                                                        const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::Tvmult (VectorType &dst,
                                                         const VectorType &src) const
{
  dst = 0;
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::vmult_add (VectorType &dst,
                                                            const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(data.ddh->get_level());
  timer->enter_subsection(section);

  implementation::WorkStream::parallel_loop<dim, VectorType, number>
  (&dst, &src, data.ddh, &(data.local_inverses));
  dst *= data.weight;
  timer->leave_subsection();
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::Tvmult_add (VectorType &dst,
    const VectorType &src) const
{
  // TODO use transpose of local inverses
  vmult_add(dst, src);
}

#include "PSCPreconditioner.inst"
