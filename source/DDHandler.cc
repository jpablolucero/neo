#include <DDHandler.h>

/***
 *** DDHandlerBase
 ***/

template <int dim>
DDHandlerBase<dim>::DDHandlerBase() :
  subdomain_to_global_map(0),
  subdomain_iterators(0),
  colorized_iterators_(0),
  level(dealii::numbers::invalid_unsigned_int),
  dofh(0)
{}


template <int dim>
DDHandlerBase<dim>::DDHandlerBase(const DDHandlerBase<dim> &other) :
  subdomain_to_global_map(other.subdomain_to_global_map),
  subdomain_iterators(other.subdomain_iterators),
  colorized_iterators_(0),
  level(other.level),
  dofh(other.dofh)
{
  // need to make sure that the colored iterators refer to the subdomain
  // iterators in this instance. Calculating it again is unnecessarily
  // complicated but simple to do.
  if (other.subdomain_iterators.size() > 0)
    {
      initialize_colorized_iterators();
    }
}


template <int dim>
void DDHandlerBase<dim>::initialize(const dealii::DoFHandler<dim> &dofh,
                                    unsigned int level)
{
  if (level == dealii::numbers::invalid_unsigned_int)
    {
      Assert(dofh.get_triangulation().n_global_levels() > 0,
             dealii::ExcInternalError());
      level = dofh.get_triangulation().n_global_levels() - 1;
    }
  this->level = level;
  this->dofh = &dofh;

  initialize_subdomain_to_global_map();
  initialize_global_dofs_on_subdomain();

  subdomain_iterators.resize(this->size());
  for (unsigned int i = 0; i < subdomain_iterators.size(); ++i)
    {
      subdomain_iterators[i] = i;
    }

  initialize_colorized_iterators();
  initialize_max_n_overlaps();
}


template <int dim>
unsigned int DDHandlerBase<dim>::size() const
{
  return subdomain_to_global_map.size();
}


template <int dim>
unsigned int DDHandlerBase<dim>::n_subdomain_dofs(
  const unsigned int subdomain_idx) const
{
  return global_dofs_on_subdomain[subdomain_idx].size();
}


template <int dim>
unsigned int DDHandlerBase<dim>::get_level() const
{
  return level;
}


template <int dim>
const dealii::DoFHandler<dim> &DDHandlerBase<dim>::get_dofh() const
{
  return *dofh;
}


template <int dim>
const std::vector<std::vector<std::vector<unsigned int>::const_iterator> > &
DDHandlerBase<dim>::colorized_iterators() const
{
  return colorized_iterators_;
}


template <int dim>
unsigned int DDHandlerBase<dim>::max_n_overlaps() const
{
  return max_n_overlaps_;
}


template <int dim>
template <class number>
void DDHandlerBase<dim>::reinit(dealii::Vector<number> &vec,
                                const unsigned int subdomain_idx) const
{
  vec.reinit(n_subdomain_dofs(subdomain_idx));
}


template <int dim>
template <typename VectorType, class number>
void DDHandlerBase<dim>::restrict_add(dealii::Vector<number> &dst,
                                      const VectorType &src,
                                      const unsigned int subdomain_idx)
const
{
  Assert(dst.size() == n_subdomain_dofs(subdomain_idx),
         dealii::ExcDimensionMismatch(dst.size(), n_subdomain_dofs(subdomain_idx)));

  for (unsigned int i = 0; i < n_subdomain_dofs(subdomain_idx); ++i)
    dst[i] += src[global_dofs_on_subdomain[subdomain_idx][i]];
}


template <int dim>
template <typename VectorType, class number>
void DDHandlerBase<dim>::prolongate_add(VectorType &dst,
                                        const dealii::Vector<number> &src,
                                        const unsigned int subdomain_idx)
const
{
  Assert(dst.size() == dofh->n_dofs(level),
         dealii::ExcDimensionMismatch(dst.size(), dofh->n_dofs(level)));
  unsigned int n_patch_dofs = n_subdomain_dofs(subdomain_idx);
  Assert(n_patch_dofs > 0, dealii::ExcInternalError());
  Assert(src.size() == n_patch_dofs,
         dealii::ExcDimensionMismatch(src.size(), n_patch_dofs));
  for (unsigned int i = 0; i < n_patch_dofs; ++i)
    dst[global_dofs_on_subdomain[subdomain_idx][i]] += src[i];
}

template <int dim>
void DDHandlerBase<dim>::initialize_colorized_iterators()
{
  AssertThrow(false, dealii::ExcInternalError());
//  typedef std::vector<unsigned int>::const_iterator IT;
//  std::function<std::vector<dealii::types::global_dof_index>(const IT &)> conflicts =
//    [this] (const IT& it)
//  {
//    return global_dofs_on_subdomain(*it);
//  };
//  this->colorized_iterators_ =
//    dealii::GraphColoring::make_graph_coloring(
//      subdomain_iterators.cbegin(),
//      subdomain_iterators.cend(),
//      conflicts);
}


template <int dim>
void DDHandlerBase<dim>::initialize_max_n_overlaps()
{
  AssertThrow(false, dealii::ExcInternalError());
//  std::vector<unsigned int> overlaps(this->size());
//  for (unsigned int i = 0; i < this->size(); ++i)
//    {
//      auto dofs_i = this->global_dofs_on_subdomain(i);
//      std::sort(dofs_i.begin(), dofs_i.end());
//      for (unsigned int j = 0; j < i; ++j)
//        {
//          if (i != j)
//            {
//              std::vector<dealii::types::global_dof_index> intersection(dofs_i.size());
//              std::vector<dealii::types::global_dof_index>::iterator it;
//              auto dofs_j = this->global_dofs_on_subdomain(j);
//              std::sort(dofs_j.begin(), dofs_j.end());
//              it = std::set_intersection(dofs_i.begin(), dofs_i.end(),
//                                         dofs_j.begin(), dofs_j.end(),
//                                         intersection.begin());
//              unsigned int overlapping_dofs = std::distance(intersection.begin(), it);
//              if (overlapping_dofs > 0)
//                {
//                  overlaps[i] += 1;
//                  overlaps[j] += 1;
//                }
//            }
//        }
//    }
//  max_n_overlaps_ = *(std::max_element(overlaps.begin(), overlaps.end()));
}

template <int dim>
void DDHandlerBase<dim>::initialize_global_dofs_on_subdomain()
{
  std::vector<dealii::types::global_dof_index> all_dofs;
  const unsigned int n_dofs_per_cell = dofh->get_fe().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> ldi(n_dofs_per_cell);
  all_to_unique.resize(subdomain_to_global_map.size());
  global_dofs_on_subdomain.resize(subdomain_to_global_map.size());
  for (unsigned int i=0; i<subdomain_to_global_map.size(); ++i)
    {
      all_dofs.clear();
      for (unsigned int j=0; j<subdomain_to_global_map[i].size(); ++j)
        {
          subdomain_to_global_map[i][j]->get_active_or_mg_dof_indices(ldi);
          all_dofs.insert(all_dofs.end(), ldi.begin(), ldi.end());
        }

      //store unique dofs
      std::unordered_set<dealii::types::global_dof_index> tmpset(all_dofs.begin(), all_dofs.end());
      global_dofs_on_subdomain[i].assign(tmpset.begin(), tmpset.end());

      //fill all_to_unique
      for (unsigned int k=0; k<global_dofs_on_subdomain[i].size(); ++k)
        all_to_unique[i][global_dofs_on_subdomain[i][k]]=k;
    }
}



/***
 *** DGDDHandlerCell
 ***/

template <int dim>
void DGDDHandlerCell<dim>::initialize_subdomain_to_global_map()
{
  const dealii::DoFHandler<dim> &dof_handler      = this->get_dofh();
  const dealii::Triangulation<dim> &triangulation = dof_handler.get_triangulation();
  if (this->get_level() >= triangulation.n_levels())
    return;
  const unsigned int n_subdomains         = triangulation.n_cells(this->get_level());
  this->subdomain_to_global_map.reserve(n_subdomains);

  //just store information for locally owned cells
  for (auto cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
      {
        this->subdomain_to_global_map.push_back(std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator>(1, cell));
        /*std::cout << " Cell id: " << cell->index()
                  << " Center " << cell->center()
                  << std::endl;*/
      }
}

template <int dim>
void DGDDHandlerCell<dim>::initialize_colorized_iterators()
{
  // Every subdomain contains only one cell and since the finite element is
  // discontinuous, any basis function will have support only on a single
  // cell so there are no write conflicts, hence we just use a single color
  // for all subdomains
  this->colorized_iterators_.assign(1, std::vector<iterator>(0));
  iterator subdomain_it = this->subdomain_iterators.cbegin();
  for (unsigned int i = 0; i < this->subdomain_iterators.size(); ++i)
    {
      this->colorized_iterators_[0].push_back(subdomain_it);
      ++subdomain_it;
    }
}


template <int dim>
void DGDDHandlerCell<dim>::initialize_max_n_overlaps()
{
  // no overlaps
  this->max_n_overlaps_ = 0;
}


/***
 *** DGDDHandlerVertex
 ***/

template <int dim>
void DGDDHandlerVertex<dim>::initialize_subdomain_to_global_map()
{
  const bool only_interior_patches = true;
  const dealii::DoFHandler<dim> &dof_handler      = this->get_dofh();
  const dealii::Triangulation<dim> &triangulation = dof_handler.get_triangulation();
  if (this->get_level() >= triangulation.n_levels())
    return;
  this->subdomain_to_global_map.reserve(dof_handler.locally_owned_mg_dofs(this->get_level()).n_elements());

  std::map<dealii::types::global_dof_index, std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > vertex_to_cell;

  //first store information for all cells and vertices we know about
  for (typename dealii::DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    if (cell->level_subdomain_id()!=dealii::numbers::artificial_subdomain_id)
      {
        for (unsigned int v=0; v<dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            bool boundary_vertex = false;
            const unsigned int vg = cell->vertex_index(v);
            /*std::cout << "Vertex: " << vg
                      << " Cell id: " << cell->index()
                      << " Center " << cell->center()
                      << std::endl;*/

            if (only_interior_patches)
              {
                for (unsigned int d=0; d<dim; ++d)
                  {
                    const unsigned int face = dealii::GeometryInfo<dim>::vertex_to_face[v][d];
                    if (cell->at_boundary(face) || cell->neighbor(face)->level() != (int) this->get_level())
                      {
                        boundary_vertex = true;
                        break;
                      }
                  }

                if (!boundary_vertex)
                  vertex_to_cell[vg].push_back(cell);
              }
            else
              vertex_to_cell[vg].push_back(cell);
          }
      }
  //now erase all vertices that are also part of a ghosted cell whose subdomain_id is lower.
  for (typename dealii::DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    {
      const unsigned int subdomain_id = cell->level_subdomain_id();
      if (subdomain_id != dealii::numbers::artificial_subdomain_id && subdomain_id<triangulation.locally_owned_subdomain())
        {
          for (unsigned int v=0; v<dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
            {
              const unsigned int vg = cell->vertex_index(v);
              vertex_to_cell[vg].clear();
            }
        }
    }
  //erase all vertices that do not have at least one locally owned cell
  {
    for (auto it = vertex_to_cell.begin(); it!=vertex_to_cell.end(); ++it)
      {
        bool has_locally_owned_cell = false;
        for (unsigned int i=0; i<it->second.size(); ++i)
          if (it->second[i]->level_subdomain_id() == triangulation.locally_owned_subdomain())
            {
              has_locally_owned_cell = true;
              break;
            }
        if (!has_locally_owned_cell)
          it->second.clear();
      }
  }

  //fill subdomain_to_global_map by ignoring all vertices that don't have cells anymore
  for (auto it=vertex_to_cell.begin(); it!=vertex_to_cell.end(); ++it)
    {
      if (it->second.size()>0)
        {
          this->subdomain_to_global_map.push_back(it->second);
          /*std::cout << "\nVertex: " << it->first << " ";
          for (unsigned int i=0; i<it->second.size(); ++i)
            std::cout << "Cell id: " << it->second[i]->index()
                      << " Center " << it->second[i]->center()
                      << std::endl;*/
        }
      // std::cout << std::endl;
    }

}


template <int dim>
void DGDDHandlerVertex<dim>::initialize_colorized_iterators()
{
  typedef std::vector<unsigned int>::const_iterator IT;
  std::function<std::vector<dealii::types::global_dof_index>(const IT &)> conflicts =
    [this] (const IT& it)
  {
    std::vector<dealii::types::global_dof_index> cell_ids(this->subdomain_to_global_map[*it].size());
    for (unsigned int i=0; i<cell_ids.size(); ++i)
      cell_ids[i]=this->subdomain_to_global_map[*it][i]->index();
    return cell_ids;
  };
  if (this->subdomain_iterators.cbegin()!=this->subdomain_iterators.cend())
    this->colorized_iterators_ =
      dealii::GraphColoring::make_graph_coloring(
        this->subdomain_iterators.cbegin(),
        this->subdomain_iterators.cend(),
        conflicts);
}


template <int dim>
void DGDDHandlerVertex<dim>::initialize_max_n_overlaps()
{
  std::vector<unsigned int> overlaps(this->size());
  for (unsigned int i = 0; i < this->size(); ++i)
    {
      std::vector<dealii::types::global_dof_index> cell_ids_i(this->subdomain_to_global_map[i].size());
      for (unsigned int c=0; c<cell_ids_i.size(); ++c)
        cell_ids_i[c]=this->subdomain_to_global_map[i][c]->index();
      std::sort(cell_ids_i.begin(), cell_ids_i.end());

      for (unsigned int j = 0; j < i; ++j)
        {

          std::vector<dealii::types::global_dof_index> cell_ids_j(this->subdomain_to_global_map[j].size());
          for (unsigned int c=0; c<cell_ids_j.size(); ++c)
            cell_ids_j[c]=this->subdomain_to_global_map[j][c]->index();

          std::vector<dealii::types::global_dof_index> intersection(cell_ids_i.size());
          std::vector<dealii::types::global_dof_index>::iterator it;
          std::sort(cell_ids_j.begin(), cell_ids_j.end());
          it = std::set_intersection(cell_ids_i.begin(), cell_ids_i.end(),
                                     cell_ids_j.begin(), cell_ids_j.end(),
                                     intersection.begin());
          const unsigned int overlapping_cells = std::distance(intersection.begin(), it);
          if (overlapping_cells > 0)
            {
              overlaps[i] += 1;
              overlaps[j] += 1;
            }
        }
    }

  this->max_n_overlaps_ = this->size()!=0?*(std::max_element(overlaps.begin(), overlaps.end())):0;
}

#include "DDHandler.inst"
