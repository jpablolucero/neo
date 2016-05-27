#include <PSCPreconditioner.h>

//#define MATRIX_FREE

namespace implementation
{
  namespace WorkStream
  {
    template <int dim, typename VectorType, class number>
    class Copy
    {
    public:
#ifndef MATRIX_FREE
      dealii::Vector<number>  local_solution;
      unsigned int subdomain_idx;
#endif
      VectorType *dst;

      const DDHandlerBase<dim> *ddh;
    };

    template <int dim, typename VectorType, class number>
    class Scratch
    {
    public:
#ifdef MATRIX_FREE
      Scratch()
        : solver_control(100, 1.e-20, 1.e-10,true, true),
          solver (solver_control)
      {}

      Scratch(const Scratch &scratch_)
        : solver_control(100, 1.e-20, 1.e-10,true, true),
          solver (solver_control),
          system_matrix (scratch_.system_matrix),
          src(scratch_.src)
      {}

      MFOperator<dim,1,false>           system_matrix;
      dealii::ReductionControl          solver_control;
      dealii::SolverCG<LA::MPI::Vector> solver;
#else
      const std::vector<dealii::FullMatrix<double> > *local_inverses;
#endif
      dealii::Vector<number>  local_src;
      const VectorType *src;
    };

#ifndef MATRIX_FREE
    template <int dim, typename VectorType, class number>
    void assemble(const Copy<dim, VectorType, number> &copy)
    {
      // write back to global vector
      copy.ddh->prolongate_add(*(copy.dst),
                               copy.local_solution,
                               copy.subdomain_idx);
    }
#endif

    template <int dim, typename VectorType, class number>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, VectorType, number> &scratch, Copy<dim, VectorType, number> &copy)
    {
      const unsigned int subdomain_idx = *iterator;
      std::cout << "Smoothening using patch: " << subdomain_idx << std::endl;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);

      for (unsigned int i=0; i<ddh.subdomain_to_global_map[subdomain_idx].size(); ++i)
        std::cout << "Cell index: " << ddh.subdomain_to_global_map[subdomain_idx][i]->index() << std::endl;

#ifdef MATRIX_FREE
      scratch.system_matrix.set_cell_range(ddh.subdomain_to_global_map[subdomain_idx]);
      scratch.solver.solve(scratch.system_matrix,*(copy.dst),*(scratch.src),dealii::PreconditionIdentity());
#else
      copy.subdomain_idx = subdomain_idx;
      ddh.reinit(scratch.local_src, subdomain_idx);
      ddh.reinit(copy.local_solution, subdomain_idx);
      // get local contributions and copy to dealii vector
      ddh.restrict_add(scratch.local_src, *(scratch.src), subdomain_idx);
      (*(scratch.local_inverses))[subdomain_idx].print(std::cout);
      (*(scratch.local_inverses))[subdomain_idx].vmult(copy.local_solution,
                                                       scratch.local_src);
#endif
    }

    /*    template <int dim, typename VectorType, class number>
        void parallel_loop(VectorType *dst,
                           const VectorType *src,
                           const DDHandlerBase<dim> *ddh)
        {
          Copy<dim, VectorType, number> copy_sample;
          copy_sample.dst = dst;
          copy_sample.ddh = ddh;

          Scratch<dim, VectorType, number> scratch_sample;
          scratch_sample.src = src;
          scratch_sample.local_inverses = &patch_inverses;
    #ifdef MATRIXFREE
          dealii::ConstraintMatrix dummy_constraints;
          dealii::MappingQ1<dim> dummy_mapping;
          const dealii::DoFHandler<dim> &dof_handler      = ddh->get_dofh();
          const dealii::parallel::distributed::Triangulation<dim> *distributed_tria
            = dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>* > (&(dof_handler.get_triangulation()));
          Assert(distributed_tria, dealii::ExcInternalError());
          const MPI_Comm &mpi_communicator = distributed_tria->get_communicator();

          scratch_sample.system_matrix.set_timer(*PSCPreconditioner<dim, VectorType, number>::timer);
          scratch_sample.system_matrix.reinit (&dof_handler,&dummy_mapping, &dummy_constraints,
                                               mpi_communicator, ddh->get_level());
    #endif
          const unsigned int queue = 2 * dealii::MultithreadInfo::n_threads();
          const unsigned int chunk_size = 1;

          dealii::WorkStream::run(ddh->colorized_iterators(),
                                  work<dim, VectorType, number>,
                                  assemble<dim, VectorType, number>,
                                  scratch_sample, copy_sample,
                                  queue,
                                  chunk_size);
        }*/
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
void PSCPreconditioner<dim, VectorType, number>::Tvmult (VectorType &/*dst*/,
                                                         const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::vmult_add (VectorType &dst,
                                                            const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(data.ddh->get_level());
  std::cout << section << std::endl;
  timer->enter_subsection(section);

  /*  implementation::WorkStream::parallel_loop<dim, VectorType, number>
    (&dst, &src, data.ddh);*/
  {
    implementation::WorkStream::Copy<dim, VectorType, number> copy_sample;
    copy_sample.dst = &dst;
    copy_sample.ddh = data.ddh;

    implementation::WorkStream::Scratch<dim, VectorType, number> scratch_sample;
    scratch_sample.src = &src;
    scratch_sample.local_inverses = &patch_inverses;
#ifdef MATRIXFREE
    dealii::ConstraintMatrix dummy_constraints;
    dealii::MappingQ1<dim> dummy_mapping;
    const dealii::DoFHandler<dim> &dof_handler      = data.ddh->get_dofh();
    const dealii::parallel::distributed::Triangulation<dim> *distributed_tria
      = dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>* > (&(dof_handler.get_triangulation()));
    Assert(distributed_tria, dealii::ExcInternalError());
    const MPI_Comm &mpi_communicator = distributed_tria->get_communicator();

    scratch_sample.system_matrix.set_timer(*PSCPreconditioner<dim, VectorType, number>::timer);
    scratch_sample.system_matrix.reinit (&dof_handler,&dummy_mapping, &dummy_constraints,
                                         mpi_communicator, ddh->get_level());
#endif
    const unsigned int queue = 2 * dealii::MultithreadInfo::n_threads();
    const unsigned int chunk_size = 1;

    dealii::WorkStream::run(data.ddh->colorized_iterators(),
                            implementation::WorkStream::work<dim, VectorType, number>,
                            implementation::WorkStream::assemble<dim, VectorType, number>,
                            scratch_sample, copy_sample,
                            queue,
                            chunk_size);

  }



  dst *= data.weight;
  timer->leave_subsection();
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::Tvmult_add (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

#include "PSCPreconditioner.inst"
