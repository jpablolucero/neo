#include <MFPSCPreconditioner.h>

#ifndef MATRIXFREE
namespace implementation
{
  namespace WorkStream
  {
    template <int dim, typename VectorType, class number>
    class Copy
    {
    public:
      VectorType *dst;

      std::shared_ptr<DDHandlerBase<dim> > ddh;
    };

    template <int dim, typename VectorType, class number>
    class Scratch
    {
    public:
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

      dealii::ReductionControl          solver_control;
      dealii::SolverCG<LA::MPI::Vector> solver;
      MFOperator<dim,1>           system_matrix;

      dealii::Vector<number>  local_src;
      const VectorType *src;
    };

    template <int dim, typename VectorType, class number>
    void assemble(const Copy<dim, VectorType, number> &/*copy*/)
    {}

    template <int dim, typename VectorType, class number>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, VectorType, number> &scratch, Copy<dim, VectorType, number> &copy)
    {
      const unsigned int subdomain_idx = *iterator;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);


      scratch.system_matrix.set_cell_range(ddh.subdomain_to_global_map[subdomain_idx]);
      scratch.solver.solve(scratch.system_matrix,*(copy.dst),*(scratch.src),dealii::PreconditionIdentity());
    }
  }
}

template <int dim, typename VectorType, class number>
MFPSCPreconditioner<dim, VectorType, number>::MFPSCPreconditioner()
{}

template <int dim, typename VectorType, class number>
void MFPSCPreconditioner<dim, VectorType, number>::vmult (VectorType &dst,
                                                          const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename VectorType, class number>
void MFPSCPreconditioner<dim, VectorType, number>::Tvmult (VectorType &/*dst*/,
                                                           const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number>
void MFPSCPreconditioner<dim, VectorType, number>::vmult_add (VectorType &dst,
    const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);

  {
    implementation::WorkStream::Copy<dim, VectorType, number> copy_sample;
    copy_sample.dst = &dst;
    copy_sample.ddh = ddh;

    implementation::WorkStream::Scratch<dim, VectorType, number> scratch_sample;
    scratch_sample.src = &src;
    dealii::ConstraintMatrix dummy_constraints;
    dealii::MappingQ1<dim> dummy_mapping;
    const dealii::DoFHandler<dim> &dof_handler      = ddh->get_dofh();
    const dealii::parallel::distributed::Triangulation<dim> *distributed_tria
      = dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>* > (&(dof_handler.get_triangulation()));
    Assert(distributed_tria, dealii::ExcInternalError());
    const MPI_Comm &mpi_communicator = distributed_tria->get_communicator();

    scratch_sample.system_matrix.set_timer(*MFPSCPreconditioner<dim, VectorType, number>::timer);
    scratch_sample.system_matrix.reinit (&dof_handler,&dummy_mapping, &dummy_constraints,
                                         mpi_communicator, level);

    const unsigned int queue = 2 * dealii::MultithreadInfo::n_threads();
    const unsigned int chunk_size = 1;

    dealii::WorkStream::run(ddh->colorized_iterators(),
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
void MFPSCPreconditioner<dim, VectorType, number>::Tvmult_add (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

#include "MFPSCPreconditioner.inst"
#endif // MATRIXFREE
