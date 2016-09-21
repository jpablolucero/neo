#ifdef MATRIXFREE

#include <MfreeOperator.h>

/*
 *  Construction & Initialization
 */
template <int dim, int fe_degree, int n_q_points_1d, typename number>
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::MfreeOperator()
{
  level = 0;
  dof_handler = nullptr;
  fe = nullptr;
  mapping = nullptr;
  constraints = nullptr;
  timer = nullptr;
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::~MfreeOperator()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::reinit (const dealii::DoFHandler<dim>  *dof_handler_,
                                                           const dealii::Mapping<dim>     *mapping_,
                                                           const dealii::ConstraintMatrix *constraints_,
                                                           const MPI_Comm                 &mpi_communicator_,
                                                           const unsigned                 int level_)
{
  timer->enter_subsection("MfreeOperator::reinit");
  // Initialize member variables
  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=level_;
  constraints = constraints_;
  mpi_communicator = mpi_communicator_;

  // Setup MatrixFree object
  const dealii::QGauss<1> quad (n_q_points_1d);
  typename dealii::MatrixFree<dim,double>::AdditionalData addit_data;
  addit_data.tasks_parallel_scheme = dealii::MatrixFree<dim,double>::AdditionalData::none;
  //  addit_data.mapping_update_flags = dealii::update_quadrature_points;
  addit_data.tasks_block_size = 3;
  addit_data.level_mg_handler = level;
#ifndef CG
  addit_data.build_face_info = true;
#endif // CG 
  addit_data.mpi_communicator = mpi_communicator;
  // TODO use constraints given by Simulator --> ERROR in Simulator::setup_multigrid()
  dealii::ConstraintMatrix dummy_constraints;
  dummy_constraints.close();
  data.reinit (*mapping, *dof_handler, dummy_constraints, quad, addit_data);
  timer->leave_subsection();
}

/*
 *  Vector multiplication
 */
template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::vmult (LA::MPI::Vector       &dst,
                                                          const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::vmult_add (LA::MPI::Vector       &dst,
    const LA::MPI::Vector &src) const
{
  Assert(dst.partitioners_are_globally_compatible(*data.get_dof_info(0).vector_partitioner), dealii::ExcInternalError());
  Assert(src.partitioners_are_globally_compatible(*data.get_dof_info(0).vector_partitioner), dealii::ExcInternalError());

  if (level != dealii::numbers::invalid_unsigned_int)
    timer->enter_subsection("MfreeOperator::loop ("+ dealii::Utilities::int_to_string(level)+ ")");
  else
    timer->enter_subsection("MfreeOperator::loop (global)");
  data.loop
  (&MFIntegrator<dim,fe_degree,n_q_points_1d,1,double>::cell,
   &MFIntegrator<dim,fe_degree,n_q_points_1d,1,double>::face,
   &MFIntegrator<dim,fe_degree,n_q_points_1d,1,double>::boundary,
   &mf_integrator,
   dst,
   src);
  timer->leave_subsection();
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult (LA::MPI::Vector       &dst,
                                                           const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult_add (LA::MPI::Vector       &dst,
    const LA::MPI::Vector &src) const
{
  vmult_add(dst, src);
}

/*
 *  Utilities
 */
template <int dim, int fe_degree, int n_q_points_1d, typename number>
const dealii::MatrixFree<dim,number> &
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::get_matrixfree_data () const
{
  return this->data;
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::initialize_dof_vector (LA::MPI::Vector &vector) const
{
  if (!vector.partitioners_are_compatible(*data.get_dof_info(0).vector_partitioner))
    data.initialize_dof_vector(vector);
}

template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
MfreeOperator<dim,fe_degree,n_q_points_1d,number>::set_timer (dealii::TimerOutput &timer_)
{
  timer = &timer_;
}

#include "MfreeOperator.inst"

#endif // MATRIXFREE
