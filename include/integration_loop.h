#ifndef INTEGRATION_LOOP_H
#define INTEGRATION_LOOP_H

#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

namespace dealii
{
  template<class INFOBOX, class DOFINFO, int dim, int spacedim, class ITERATOR>
  void restricted_cell_action(
    const ITERATOR &cell,
    const std::vector<ITERATOR> &cell_range,
    MeshWorker::DoFInfoBox<dim, DOFINFO> &dof_info,
    INFOBOX &info,
    const std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)> &cell_worker,
    const std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)> &boundary_worker,
    const std_cxx11::function<void (DOFINFO &, DOFINFO &,
                                    typename INFOBOX::CellInfo &,
                                    typename INFOBOX::CellInfo &)> &face_worker,
    const MeshWorker::LoopControl &loop_control)
  {
    const bool ignore_subdomain = (cell->get_triangulation().locally_owned_subdomain()
                                   == numbers::invalid_subdomain_id);

    types::subdomain_id csid = (cell->is_level_cell())
                               ? cell->level_subdomain_id()
                               : cell->subdomain_id();

    const bool own_cell = ignore_subdomain || (csid == cell->get_triangulation().locally_owned_subdomain());

    dof_info.reset();

    if ((!ignore_subdomain) && (csid == numbers::artificial_subdomain_id))
      return;

    dof_info.cell.reinit(cell);
    dof_info.cell_valid = true;

    const bool integrate_cell          = (cell_worker != 0);
    const bool integrate_boundary      = (boundary_worker != 0);
    const bool integrate_interior_face = (face_worker != 0);

    if (integrate_cell)
      info.cell.reinit(dof_info.cell);
    // Execute this, if cells
    // have to be dealt with
    // before faces
    if (integrate_cell && loop_control.cells_first &&
        ((loop_control.own_cells && own_cell) || (loop_control.ghost_cells && !own_cell)))
      cell_worker(dof_info.cell, info.cell);

    // Call the callback function in
    // the info box to do
    // computations between cell and
    // face action.
    info.post_cell(dof_info);

    if (integrate_interior_face || integrate_boundary)
      for (unsigned int face_no=0; face_no < GeometryInfo<ITERATOR::AccessorType::Container::dimension>::faces_per_cell; ++face_no)
        {
          typename ITERATOR::AccessorType::Container::face_iterator face = cell->face(face_no);
          if (cell->at_boundary(face_no) && !cell->has_periodic_neighbor(face_no))
            {
              // only integrate boundary faces of own cells
              if (integrate_boundary && (own_cell || loop_control.ghost_cells))
                {
                  dof_info.interior_face_available[face_no] = true;
                  dof_info.interior[face_no].reinit(cell, face, face_no);
                  info.boundary.reinit(dof_info.interior[face_no]);
                  boundary_worker(dof_info.interior[face_no], info.boundary);
                }
            }
          else if (integrate_interior_face)
            {
              // Interior face
              TriaIterator<typename ITERATOR::AccessorType> neighbor = cell->neighbor_or_periodic_neighbor(face_no);

              types::subdomain_id neighbid = numbers::artificial_subdomain_id;
              if (neighbor->is_level_cell())
                neighbid = neighbor->level_subdomain_id();
              //subdomain id is only valid for active cells
              else if (neighbor->active())
                neighbid = neighbor->subdomain_id();

              const bool own_neighbor = ignore_subdomain ||
                                        (neighbid == cell->get_triangulation().locally_owned_subdomain());

              // skip if the user doesn't want faces between own cells
              if (own_cell && own_neighbor && loop_control.own_faces==MeshWorker::LoopControl::never)
                continue;

              // skip face to ghost
              if (own_cell != own_neighbor && loop_control.faces_to_ghost==MeshWorker::LoopControl::never)
                continue;

              // Is this face also interior for the local patch?
              // If not, ignore all contributions to other cells.
              bool exterior_in_patch = false;
              for (unsigned int i=0; i<cell_range.size(); ++i)
                if (cell_range[i]->index() == neighbor->index())
                  {
                    exterior_in_patch = true;
                    break;
                  }

              // Deal with refinement edges from the refined side. Assuming one-irregular
              // meshes, this situation should only occur if both cells are active.
              const bool periodic_neighbor = cell->has_periodic_neighbor(face_no);

              if ((!periodic_neighbor && cell->neighbor_is_coarser(face_no))
                  || (periodic_neighbor && cell->periodic_neighbor_is_coarser(face_no)))
                {
                  Assert(false/*!cell->has_children()*/, ExcInternalError());
                  Assert(!neighbor->has_children(), ExcInternalError());

                  // skip if only one processor needs to assemble the face
                  // to a ghost cell and the fine cell is not ours.
                  if (!own_cell
                      && loop_control.faces_to_ghost == MeshWorker::LoopControl::one)
                    continue;

                  const std::pair<unsigned int, unsigned int> neighbor_face_no
                    = periodic_neighbor?
                      cell->periodic_neighbor_of_coarser_periodic_neighbor(face_no):
                      cell->neighbor_of_coarser_neighbor(face_no);
                  const typename ITERATOR::AccessorType::Container::face_iterator nface
                    = neighbor->face(neighbor_face_no.first);

                  dof_info.interior_face_available[face_no] = true;
                  dof_info.exterior_face_available[face_no] = exterior_in_patch;
                  dof_info.interior[face_no].reinit(cell, face, face_no);
                  info.face.reinit(dof_info.interior[face_no]);
                  dof_info.exterior[face_no].reinit(
                    neighbor, nface, neighbor_face_no.first, neighbor_face_no.second);
                  info.subface.reinit(dof_info.exterior[face_no]);

                  face_worker(dof_info.interior[face_no], dof_info.exterior[face_no],
                              info.face, info.subface);
                }
              else
                {
                  // If iterator is active and neighbor is refined, skip
                  // internal face.
                  if (internal::is_active_iterator(cell) && neighbor->has_children())
                    {
                      Assert(loop_control.own_faces != MeshWorker::LoopControl::both, ExcMessage(
                               "Assembling from both sides for own_faces is not "
                               "supported with hanging nodes!"));
                      //continue;
                    }

                  // Now neighbor is on same level, double-check this:
                  Assert(cell->level()==neighbor->level(), ExcInternalError());

                  // If we own both cells only do faces from one side (unless
                  // MeshWorker::LoopControl says otherwise). Here, we rely on cell comparison
                  // that will look at cell->index().
                  if (neighbor < cell && exterior_in_patch)
                    continue;
                  if (own_cell && own_neighbor
                      && loop_control.own_faces == MeshWorker::LoopControl::one
                      && (neighbor < cell)
                      && exterior_in_patch)
                    continue;

                  // independent of loop_control.faces_to_ghost,
                  // we only look at faces to ghost on the same level once
                  // (only where own_cell=true and own_neighbor=false)
                  if (!own_cell && !loop_control.ghost_cells)
                    continue;

                  // now only one processor assembles faces_to_ghost. We let the
                  // processor with the smaller (level-)subdomain id assemble the
                  // face.
                  if (own_cell && !own_neighbor
                      && loop_control.faces_to_ghost == MeshWorker::LoopControl::one
                      && !loop_control.ghost_cells)
                    continue;

                  const unsigned int neighbor_face_no = periodic_neighbor?
                                                        cell->periodic_neighbor_face_no(face_no):
                                                        cell->neighbor_face_no(face_no);
                  Assert (periodic_neighbor || neighbor->face(neighbor_face_no) == face, ExcInternalError());
                  // Regular interior face
                  dof_info.interior_face_available[face_no] = true;
                  dof_info.exterior_face_available[face_no] = exterior_in_patch;
                  dof_info.interior[face_no].reinit(cell, face, face_no);
                  info.face.reinit(dof_info.interior[face_no]);
                  dof_info.exterior[face_no].reinit(
                    neighbor, neighbor->face(neighbor_face_no), neighbor_face_no);
                  info.neighbor.reinit(dof_info.exterior[face_no]);

                  face_worker(dof_info.interior[face_no], dof_info.exterior[face_no],
                              info.face, info.neighbor);
                }
            }
        } // faces
    // Call the callback function in
    // the info box to do
    // computations between face and
    // cell action.
    info.post_faces(dof_info);

    // Execute this, if faces
    // have to be handled first
    if (integrate_cell && !loop_control.cells_first &&
        ((loop_control.own_cells && own_cell) || (loop_control.ghost_cells && !own_cell)))
      cell_worker(dof_info.cell, info.cell);
  }


  template<int dim, int spacedim, typename ITERATOR, typename DOFINFO,
           typename INFOBOX, typename INTEGRATOR, typename ASSEMBLER>
  void colored_loop(const std::vector<std::vector<ITERATOR> > &colored_iterators,
                    DOFINFO  &dof_info,
                    INFOBOX  &info,
                    const INTEGRATOR &integrator,
                    ASSEMBLER  &assembler,
                    const dealii::MeshWorker::LoopControl &lctrl = dealii::MeshWorker::LoopControl(),
                    const std::vector<ITERATOR> &total_cell_range = std::vector<ITERATOR>())
  {
    bool parallel = true;
    bool restrict_to_cell_range = (total_cell_range.size()!=0);
#ifdef DEBUG
    if (restrict_to_cell_range)
      {
        unsigned int sum_colored_ranges = 0;
        for (unsigned int color = 0; color < colored_iterators.size(); ++color)
          sum_colored_ranges += colored_iterators[color].size();
        Assert(sum_colored_ranges == total_cell_range.size(), ExcInternalError());
      }
#endif

    std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)>   cell_worker ;
    std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)>   boundary_worker ;
    std_cxx11::function<void (DOFINFO &, DOFINFO &,
                              typename INFOBOX::CellInfo &, typename INFOBOX::CellInfo &)>   face_worker ;

    // TODO: get rid of 'ifs' here to allow generic INTEGRATORs as it is designed
    if (integrator.use_cell)
      cell_worker = std_cxx11::bind(&INTEGRATOR::cell, &integrator, std_cxx11::_1, std_cxx11::_2);
    if (integrator.use_boundary)
      boundary_worker = std_cxx11::bind(&INTEGRATOR::boundary, &integrator, std_cxx11::_1, std_cxx11::_2);
    if (integrator.use_face)
      face_worker = std_cxx11::bind(&INTEGRATOR::face, &integrator, std_cxx11::_1, std_cxx11::_2,
                                    std_cxx11::_3, std_cxx11::_4);

    std_cxx11::function<void (const ITERATOR &, INFOBOX &, MeshWorker::DoFInfoBox<dim, DOFINFO>&)> cell_action;
    if (restrict_to_cell_range)
      cell_action = std_cxx11::bind(&restricted_cell_action<INFOBOX, DOFINFO, dim, spacedim, ITERATOR>,
                                    std_cxx11::_1, total_cell_range, std_cxx11::_3,
                                    std_cxx11::_2, cell_worker, boundary_worker, face_worker, lctrl);

    else
      cell_action = std_cxx11::bind(&MeshWorker::cell_action<INFOBOX, DOFINFO, dim, spacedim, ITERATOR>,
                                    std_cxx11::_1, std_cxx11::_3, std_cxx11::_2,
                                    cell_worker, boundary_worker, face_worker, lctrl);

    MeshWorker::DoFInfoBox<dim, DOFINFO> dof_info_box(dof_info);
    assembler.initialize_info(dof_info_box.cell, false);
    for (unsigned int i=0; i<GeometryInfo<dim>::faces_per_cell; ++i)
      {
        assembler.initialize_info(dof_info_box.interior[i], true);
        assembler.initialize_info(dof_info_box.exterior[i], true);
      }

    //  Loop over all cells
    if (parallel)
      {
        WorkStream::run(colored_iterators, cell_action,
                        std_cxx11::bind(&internal::assemble<dim,DOFINFO,ASSEMBLER>,
                                        std_cxx11::_1, &assembler),
                        info, dof_info_box,
                        MultithreadInfo::n_threads(),8);
      }
    else
      {
        for (unsigned int color=0; color<colored_iterators.size(); ++color)
          for (typename std::vector<ITERATOR>::const_iterator p = colored_iterators[color].begin();
               p != colored_iterators[color].end(); ++p)
            {
              cell_action(*p, info, dof_info_box);
              dof_info_box.assemble(assembler);
            }
      }
  }
}
#endif

