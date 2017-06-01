#include <Mesh.h>

template <int dim>
Mesh<dim>::Mesh(MPI_Comm & mpi_communicator_):
  triangulation(mpi_communicator_,dealii::Triangulation<dim>::
		limit_level_difference_at_vertices,
		dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
{
  dealii::GridGenerator::hyper_cube(triangulation,0.,1.,true);

#ifdef PERIODIC
  //add periodicity
  typedef typename dealii::Triangulation<dim>::cell_iterator CellIteratorTria;
  std::vector<dealii::GridTools::PeriodicFacePair<CellIteratorTria> > periodic_faces;
  const unsigned int b_id1 = 2;
  const unsigned int b_id2 = 3;
  const unsigned int direction = 1;

  dealii::GridTools::collect_periodic_faces (triangulation, b_id1, b_id2,
                                             direction, periodic_faces, dealii::Tensor<1,dim>());
  triangulation.add_periodicity(periodic_faces);
#endif
}

template class Mesh<2>;
template class Mesh<3>;
