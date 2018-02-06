#include <RHS.h>

extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

template <int dim>
RHS<dim>::RHS (FiniteElement<dim> & fe_,Dofs<dim> & dofs_):
  fe(fe_),
  dofs(dofs_)
{}

template <int dim>
void RHS<dim>::assemble(const dealii::parallel::distributed::Vector<double> & solution)
{
  unsigned int level = dofs.mesh.triangulation.n_levels()-1;
  right_hand_side.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);

  dealii::MGLevelObject<dealii::parallel::distributed::Vector<double>>  ghosted_solution ;
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
  
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  const unsigned int n_gauss_points = fe.fe.degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
                                     dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);

  dealii::AnyData src_data;
  src_data.add<const dealii::MGLevelObject<dealii::parallel::distributed::Vector<double> >*>(&ghosted_solution,"Newton iterate");
  info_box.initialize(fe.fe,fe.mapping,src_data,dealii::parallel::distributed::Vector<double> {},&(dofs.dof_handler.block_info()));
  dealii::MeshWorker::DoFInfo<dim> dof_info(dofs.dof_handler.block_info());

  dealii::AnyData data;
  data.add<dealii::parallel::distributed::Vector<double> *>(&right_hand_side, "RHS");

  dealii::MeshWorker::Assembler::ResidualSimple<dealii::parallel::distributed::Vector<double> > rhs_assembler;
  rhs_assembler.initialize(data);
#ifdef CG
  rhs_assembler.initialize(dofs.constraints);
#endif
  RHSIntegrator<dim> rhs_integrator(fe.fe.n_components());

  dealii::MeshWorker::LoopControl lctrl;
  dealii::colored_loop<dim, dim> (colored_iterators,dof_info,info_box,rhs_integrator,rhs_assembler,lctrl) ;
  right_hand_side.compress(dealii::VectorOperation::add);
}

template class RHS<2>;
template class RHS<3>;

