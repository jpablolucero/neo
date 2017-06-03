#include <Dofs.h>

template <int dim>
Dofs<dim>::Dofs(Mesh<dim> & mesh_,FiniteElement<dim> & fe_):
  mesh(mesh_),
  fe(fe_),
  dof_handler(mesh.triangulation),
  reference_function(fe.fe.n_components())
{}

template <int dim>
void Dofs<dim>::setup()
{
  dof_handler.distribute_dofs (fe.fe);
  dof_handler.distribute_mg_dofs(fe.fe);
  dof_handler.initialize_local_block_info();

  locally_owned_dofs = dof_handler.locally_owned_dofs();
//   /*std::cout << "locally owned dofs on process "
//             << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
//             << std::endl;
//   for (unsigned int l=0; l<mesh.triangulation.n_global_levels(); ++l)
//     {
//       std::cout << "level: " << l << " n_elements(): "
//                 << dof_handler.locally_owned_mg_dofs(l).n_elements()
//                 << " index set: ";
//       dof_handler.locally_owned_mg_dofs(l).print(std::cout);
//     }
//   std::cout << "n_elements(): "
//             << dof_handler.locally_owned_dofs().n_elements()
//             <<std::endl;
//   dof_handler.locally_owned_dofs().print(dealii::deallog);*/
//   /*  std::cout << "locally relevant dofs on process "
//               << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
//     locally_relevant_dofs.print(std::cout);*/

  dealii::DoFTools::extract_locally_relevant_dofs
  (dof_handler, locally_relevant_dofs);
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
#ifdef CG
#ifdef PERIODIC
  //Periodic boundary conditions
  std::vector<dealii::GridTools::PeriodicFacePair
	      <typename dealii::DoFHandler<dim>::cell_iterator> >
    periodic_faces;

  const unsigned int b_id1 = 2*dim-2;
  const unsigned int b_id2 = 2*dim-1;
  const unsigned int direction = dim-1;

  dealii::GridTools::collect_periodic_faces (dof_handler,
                                             b_id1, b_id2, direction,
                                             periodic_faces);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<dim> >
    (periodic_faces, constraints);
  for (unsigned int i=0; i<2*dim-2; ++i)
    dealii::VectorTools::interpolate_boundary_values(dof_handler, i,
                                                     reference_function,
                                                     constraints);
#else
  for (unsigned int i=0; i<2*dim; ++i)
    dealii::VectorTools::interpolate_boundary_values(dof_handler, i,
                                                     reference_function,
                                                     constraints);
#endif

  dealii::DoFTools::make_hanging_node_constraints
    (dof_handler, constraints);

#endif
  constraints.close();
}

template class Dofs<2>;
template class Dofs<3>;