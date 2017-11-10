#include <RHS.h>

extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

template <int dim>
RHS<dim>::RHS (FiniteElement<dim> & fe_,Dofs<dim> & dofs_):
  fe(fe_),
  dofs(dofs_)
{}

template <int dim>
void RHS<dim>::assemble(const LA::MPI::Vector & solution)
{
#ifdef MATRIXFREE
#if PARALLEL_LA == 3
  right_hand_side.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
#elif PARALLEL_LA == 0
  right_hand_side.reinit (dofs.locally_owned_dofs.n_elements());
#else // PARALLEL_LA == 1,2
  AssertThrow(false, dealii::ExcNotImplemented());
#endif // PARALLEL_LA == 3

#else // MATRIXFREE OFF
#if PARALLEL_LA == 0
  right_hand_side.reinit (dofs.locally_owned_dofs.n_elements());
#elif PARALLEL_LA == 3
  right_hand_side.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
#else
  right_hand_side.reinit (dofs.locally_owned_dofs, *mpi_communicator);
#endif // PARALLEL_LA == 0
#endif // MATRIXFREE
  
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  const unsigned int n_gauss_points = fe.fe.degree+1;
#ifdef CG
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
#else
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
#endif // CG
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
                                     dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);

  dealii::AnyData src_data;
#ifdef MATRIXFREE
  info_box.initialize(fe.fe, fe.mapping);
  dealii::MeshWorker::DoFInfo<dim> dof_info(dofs.dof_handler);
#else
  src_data.add<const LA::MPI::Vector *>(&solution,"Newton iterate");
  info_box.initialize(fe.fe,fe.mapping,src_data,LA::MPI::Vector {},&(dofs.dof_handler.block_info()));
  dealii::MeshWorker::DoFInfo<dim> dof_info(dofs.dof_handler.block_info());
#endif // MATRIXFREE

  dealii::AnyData data;
  data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");

  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
//  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
  rhs_assembler.initialize(data);
#ifdef CG
  rhs_assembler.initialize(dofs.constraints);
#endif
  RHSIntegrator<dim> rhs_integrator(fe.fe.n_components());

  dealii::MeshWorker::integration_loop<dim, dim>(dofs.dof_handler.begin_active(),
                                                 dofs.dof_handler.end(),
                                                 dof_info,
                                                 info_box,
                                                 rhs_integrator,
                                                 rhs_assembler);

  right_hand_side.compress(dealii::VectorOperation::add);
}

template class RHS<2>;
template class RHS<3>;

