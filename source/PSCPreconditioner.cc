#include <PSCPreconditioner.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>

#include <functional>

#include <GlobalTimer.h>

namespace implementation
{
  namespace WorkStream
  {
    template <int dim, class number, class IT>
      std::vector<dealii::types::global_dof_index>
      dd_conflict_indices(const DDHandlerBase<dim, number>& ddh,
                          const IT& iterator)
      {
        const unsigned int subdomain_idx = *iterator;
        return ddh.global_dof_indices_on_subdomain(subdomain_idx);
      }

    template <int dim, class number>
      class Copy
      {
        public:
          dealii::Vector<number> local_solution;
          unsigned int subdomain_idx;
          dealii::Vector<number>* dst;

          const DDHandlerBase<dim, number>* ddh;
      };

    template <int dim, class number>
      class Scratch
      {
        public:
          dealii::Vector<number> local_src;

          const dealii::Vector<number>* src;
          const std::vector<const dealii::FullMatrix<number>* >* local_inverses;
      };

    template <int dim, class number>
      void assemble(const Copy<dim, number>& copy)
      {
        copy.ddh->prolongate_add(*(copy.dst),
                                 copy.local_solution,
                                 copy.subdomain_idx);
      }

    template <int dim, class number>
      void work(const std::vector<unsigned int>::const_iterator& iterator,
                Scratch<dim, number>& scratch, Copy<dim, number>& copy)
      {
        unsigned int subdomain_idx = *iterator;
        copy.subdomain_idx = subdomain_idx;
        const DDHandlerBase<dim, number>& ddh = *(copy.ddh);
        ddh.reinit(scratch.local_src, subdomain_idx);
        ddh.reinit(copy.local_solution, subdomain_idx);
        ddh.restrict_add(scratch.local_src, *(scratch.src), subdomain_idx);
        (*scratch.local_inverses)[subdomain_idx]->vmult(copy.local_solution,
                                                        scratch.local_src);
      }

    template <int dim, class number>
      void parallel_loop(dealii::Vector<number>* dst,
                         const dealii::Vector<number>* src,
                         const DDHandlerBase<dim, number>* ddh,
                         const std::vector<const dealii::FullMatrix<number>* >*
                         local_inverses)
      {
        Copy<dim, number> copy_sample;
        copy_sample.dst = dst;
        copy_sample.ddh = ddh;

        Scratch<dim, number> scratch_sample;
        scratch_sample.src = src;
        scratch_sample.local_inverses = local_inverses;

        const unsigned int queue = 2 * dealii::MultithreadInfo::n_threads();
        const unsigned int chunk_size = 1;

        dealii::WorkStream::run(ddh->colorized_iterators(),
                                work<dim, number>, assemble<dim, number>,
                                scratch_sample, copy_sample,
                                queue,
                                chunk_size);
      }
  }
}

  template <int dim, class number>
PSCPreconditioner<dim, number>::PSCPreconditioner()
{}

template <int dim, class number>
void PSCPreconditioner<dim, number>::vmult (dealii::Vector<number> &dst,
                                            const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, class number>
void PSCPreconditioner<dim, number>::Tvmult (dealii::Vector<number> &dst,
                                             const dealii::Vector<number> &src) const
{
  dst = 0;
  Tvmult_add(dst, src);
}

template <int dim, class number>
void PSCPreconditioner<dim, number>::vmult_add (dealii::Vector<number> &dst,
                                                const dealii::Vector<number> &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(data.ddh->get_level());
  global_timer.enter_subsection(section);

  implementation::WorkStream::parallel_loop<dim, double>(&dst,
                                                         &src,
                                                         data.ddh,
                                                         &(data.local_inverses));
  dst *= data.weight;
  global_timer.leave_subsection();
}

template <int dim, class number>
void PSCPreconditioner<dim, number>::Tvmult_add (dealii::Vector<number> &dst,
                                                 const dealii::Vector<number> &src) const
{
  // TODO use transpose of local inverses
  vmult_add(dst, src);
}

template class PSCPreconditioner<2, double>;
template class PSCPreconditioner<3, double>;
