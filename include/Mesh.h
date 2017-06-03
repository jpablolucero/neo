#ifndef MESH_H
#define MESH_H

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

template <int dim>
class Mesh final
{
public:
  Mesh () ;
  dealii::parallel::distributed::Triangulation<dim>   triangulation;
};

#ifdef HEADER_IMPLEMENTATION
#include <Mesh.cc>
#endif

#endif // MESH_H
