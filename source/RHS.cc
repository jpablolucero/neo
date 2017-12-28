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
  unsigned int level = dofs.mesh.triangulation.n_levels()-1;
  right_hand_side.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
#else
  right_hand_side.reinit (dofs.locally_owned_dofs, *mpi_communicator);
#endif // PARALLEL_LA == 0
#endif // MATRIXFREE

#if PARALLEL_LA == 3
  dealii::MGLevelObject<LA::MPI::Vector>  ghosted_solution ;
  ghosted_solution.resize(level,level);
  ghosted_solution[level] = std::move(solution) ;
  ghosted_solution[level].update_ghost_values();
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >
    colored_iterators(static_cast<unsigned int>(std::pow(2,dim)));
  auto i = 1 ;
  for (auto p=dofs.dof_handler.begin_mg(level); p!=dofs.dof_handler.end_mg(level); ++p)
    {
      const dealii::types::subdomain_id csid = (p->is_level_cell())
	? p->level_subdomain_id()
	: p->subdomain_id();
      if (csid == p->get_triangulation().locally_owned_subdomain())
        {
	  colored_iterators[i-1].push_back(p);
          i = i % static_cast<unsigned int>(std::pow(2,dim)) ;
          ++i;
        }
    }
#endif 
  
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
#if PARALLEL_LA == 3
  src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(&ghosted_solution,"Newton iterate");
#else
  src_data.add<const LA::MPI::Vector *>(&solution,"Newton iterate");
#endif
  info_box.initialize(fe.fe,fe.mapping,src_data,LA::MPI::Vector {},&(dofs.dof_handler.block_info()));
  dealii::MeshWorker::DoFInfo<dim> dof_info(dofs.dof_handler.block_info());
#endif // MATRIXFREE

  dealii::AnyData data;
  data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");

#if PARALLEL_LA == 3
  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
#else
  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
#endif
  rhs_assembler.initialize(data);
#ifdef CG
  rhs_assembler.initialize(dofs.constraints);
#endif
  RHSIntegrator<dim> rhs_integrator(fe.fe.n_components());

#if PARALLEL_LA == 3
  dealii::MeshWorker::LoopControl lctrl;
  dealii::colored_loop<dim, dim> (colored_iterators,
				  dof_info,
				  info_box,
				  rhs_integrator,
				  rhs_assembler,
				  lctrl) ;
#else
  dealii::MeshWorker::integration_loop<dim, dim>(dofs.dof_handler.begin_active(),
                                                 dofs.dof_handler.end(),
                                                 dof_info,
                                                 info_box,
                                                 rhs_integrator,
                                                 rhs_assembler);
#endif
  right_hand_side.compress(dealii::VectorOperation::add);
}

template class RHS<2>;
template class RHS<3>;

