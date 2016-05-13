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
  return subdomain_to_global_map[subdomain_idx].size();
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
  assert(dst.size() == n_subdomain_dofs(subdomain_idx));
  for (unsigned int i = 0; i < n_subdomain_dofs(subdomain_idx); ++i)
    {
      dst[i] += src[subdomain_to_global_map[subdomain_idx][i]];
    }
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
  unsigned int n_block_dofs = n_subdomain_dofs(subdomain_idx);
  Assert(n_block_dofs > 0, dealii::ExcInternalError());
  Assert(src.size() == n_block_dofs,
         dealii::ExcDimensionMismatch(src.size(), n_block_dofs));
  for (unsigned int i = 0; i < n_block_dofs; ++i)
    {
      dst[subdomain_to_global_map[subdomain_idx][i]] += src[i];
    }
}


template <int dim>
std::vector<dealii::types::global_dof_index> DDHandlerBase<dim>::global_dofs_on_subdomain(
  const unsigned int subdomain_idx) const
{
  return subdomain_to_global_map[subdomain_idx];
}


template <int dim>
void DDHandlerBase<dim>::initialize_colorized_iterators()
{
  typedef std::vector<unsigned int>::const_iterator IT;
  std::function<std::vector<dealii::types::global_dof_index>(const IT &)> conflicts =
    [this] (const IT& it)
  {
    return this->global_dofs_on_subdomain(*it);
  };
  this->colorized_iterators_ =
    dealii::GraphColoring::make_graph_coloring(
      subdomain_iterators.cbegin(),
      subdomain_iterators.cend(),
      conflicts);
}


template <int dim>
void DDHandlerBase<dim>::initialize_max_n_overlaps()
{
  std::vector<unsigned int> overlaps(this->size());
  for (unsigned int i = 0; i < this->size(); ++i)
    {
      auto dofs_i = this->global_dofs_on_subdomain(i);
      std::sort(dofs_i.begin(), dofs_i.end());
      for (unsigned int j = 0; j < i; ++j)
        {
          if (i != j)
            {
              std::vector<dealii::types::global_dof_index> intersection(dofs_i.size());
              std::vector<dealii::types::global_dof_index>::iterator it;
              auto dofs_j = this->global_dofs_on_subdomain(j);
              std::sort(dofs_j.begin(), dofs_j.end());
              it = std::set_intersection(dofs_i.begin(), dofs_i.end(),
                                         dofs_j.begin(), dofs_j.end(),
                                         intersection.begin());
              unsigned int overlapping_dofs = std::distance(intersection.begin(), it);
              if (overlapping_dofs > 0)
                {
                  overlaps[i] += 1;
                  overlaps[j] += 1;
                }
            }
        }
    }
  max_n_overlaps_ = *(std::max_element(overlaps.begin(), overlaps.end()));
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
  const unsigned int n_subdomain_dofs     = dof_handler.get_fe().dofs_per_cell;
  this->subdomain_to_global_map.reserve(n_subdomains);

  //just store information for locally owned cells
  unsigned int subdomain_idx = 0;
  for (auto cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
      {
        this->subdomain_to_global_map.push_back(std::vector<dealii::types::global_dof_index>(n_subdomain_dofs));
        cell->get_active_or_mg_dof_indices(this->subdomain_to_global_map[subdomain_idx]);
        ++subdomain_idx;
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
  const dealii::DoFHandler<dim> &dof_handler      = this->get_dofh();
  const dealii::Triangulation<dim> &triangulation = dof_handler.get_triangulation();
  if (this->get_level() >= triangulation.n_levels())
    return;
  const unsigned int n_subdomains         = triangulation.n_cells(this->get_level());
  const unsigned int n_subdomain_dofs     = dof_handler.get_fe().dofs_per_cell;
  this->subdomain_to_global_map.reserve(n_subdomains);

  //just store information for locally owned cells
  unsigned int subdomain_idx = 0;
  for (DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
      {
        this->subdomain_to_global_map.push_back(std::vector<dealii::types::global_dof_index>(n_subdomain_dofs));
        cell->get_active_or_mg_dof_indices(this->subdomain_to_global_map[subdomain_idx]);
        local_cell_ids.push_back(cell);
        ++subdomain_idx;
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const unsigned int vg = cell->vertex_index(v);
            vertex_to_cell[vg].push_back(cell);
        }
      }
  //now erase all vertices that are also part of a ghosted cell whose subdomain_id is lower.
  for (DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
  {
      const unsigned int subdomain_id = cell->level_subdomain_id();
    if (subdomain_id != dealii::numbers::artificial_subdomain_id && subdomain_id<triangulation.locally_owned_subdomain())
      {
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const unsigned int vg = cell->vertex_index(v);
            vertex_to_cell[vg].clear();
        }
      }
}


template <int dim>
void DGDDHandlerVertex<dim>::initialize_vertex_to_cell_map()
{
    // store for each vertex on a locally owned cell the cells that contain that vertex
    // for vertices that have neighbors on different subdomains only store on the cell
    // with the lowest subdomain_id
    // we are only considering internal vertices

    const dealii::DoFHandler<dim> &dof_handler      = this->get_dofh();
    const dealii::Triangulation<dim> &triangulation = dof_handler.get_triangulation();
    if (this->get_level() >= triangulation.n_levels())
      return;

    typename dealii::DoFHandler<dim>::level_cell_iterator cell;
    typename dealii::DoFHandler<dim>::level_cell_iterator endc = dof_handler.end(level);

    // Vector mapping from vertex index in the triangulation to consecutive
    2190     // block indices on this level The number of cells at a vertex
    2191     std::vector<unsigned int> vertex_cell_count(dof_handler.get_triangulation().n_vertices(), 0);
    2192
    2193     // Is a vertex at the boundary?
    2194     std::vector<bool> vertex_boundary(dof_handler.get_triangulation().n_vertices(), false);
    2195
    2196     std::vector<unsigned int> vertex_mapping(dof_handler.get_triangulation().n_vertices(),
    2197                                              numbers::invalid_unsigned_int);
    2198
    2199     // Estimate for the number of dofs at this point
    2200     std::vector<unsigned int> vertex_dof_count(dof_handler.get_triangulation().n_vertices(), 0);
    2201
    2202     // Identify all vertices active on this level and remember some data
    2203     // about them
    2204     for (cell=dof_handler.begin(level); cell != endc; ++cell)
    2205       for (unsigned int v=0; v<GeometryInfo<DoFHandlerType::dimension>::vertices_per_cell; ++v)
    2206         {
    2207           const unsigned int vg = cell->vertex_index(v);
    2208           vertex_dof_count[vg] += cell->get_fe().dofs_per_cell;
    2209           ++vertex_cell_count[vg];
    2210           for (unsigned int d=0; d<DoFHandlerType::dimension; ++d)
    2211             {
    2212               const unsigned int face = GeometryInfo<DoFHandlerType::dimension>::vertex_to_face[v][d];
    2213               if (cell->at_boundary(face))
    2214                 vertex_boundary[vg] = true;
    2215               else if ((!level_boundary_patches)
    2216                        && (cell->neighbor(face)->level() != (int) level))
    2217                 vertex_boundary[vg] = true;
    2218             }
    2219         }
    2220     // From now on, only vertices with positive dof count are "in".
    2221
    2222     // Remove vertices at boundaries or in corners
    2223     for (unsigned int vg=0; vg<vertex_dof_count.size(); ++vg)
    2224       if ((!single_cell_patches && vertex_cell_count[vg] < 2)
    2225           ||
    2226           (!boundary_patches && vertex_boundary[vg]))
    2227         vertex_dof_count[vg] = 0;
    2228
    2229     // Create a mapping from all vertices to the ones used here
    2230     unsigned int n_vertex_count=0;
    2231     for (unsigned int vg=0; vg<vertex_mapping.size(); ++vg)
    2232       if (vertex_dof_count[vg] != 0)
    2233         vertex_mapping[vg] = n_vertex_count++;
    2234
    2235     // Compactify dof count
    2236     for (unsigned int vg=0; vg<vertex_mapping.size(); ++vg)
    2237       if (vertex_dof_count[vg] != 0)
    2238         vertex_dof_count[vertex_mapping[vg]] = vertex_dof_count[vg];
    2239
    2240     // Now that we have all the data, we reduce it to the part we actually
    2241     // want
    2242     vertex_dof_count.resize(n_vertex_count);
    2243
    2244     // At this point, the list of patches is ready. Now we enter the dofs
    2245     // into the sparsity pattern.
    2246     block_list.reinit(vertex_dof_count.size(), dof_handler.n_dofs(level), vertex_dof_count);
    2247
    2248     std::vector<types::global_dof_index> indices;
    2249     std::vector<bool> exclude;
    2250
    2251     for (cell=dof_handler.begin(level); cell != endc; ++cell)
    2252       {
    2253         const FiniteElement<DoFHandlerType::dimension> &fe = cell->get_fe();
    2254         indices.resize(fe.dofs_per_cell);
    2255         cell->get_mg_dof_indices(indices);
    2256
    2257         for (unsigned int v=0; v<GeometryInfo<DoFHandlerType::dimension>::vertices_per_cell; ++v)
    2258           {
    2259             const unsigned int vg = cell->vertex_index(v);
    2260             const unsigned int block = vertex_mapping[vg];
    2261             if (block == numbers::invalid_unsigned_int)
    2262               continue;
    2263
    2264             if (interior_only)
    2265               {
    2266                 // Exclude degrees of freedom on faces opposite to the
    2267                 // vertex
    2268                 exclude.resize(fe.dofs_per_cell);
    2269                 std::fill(exclude.begin(), exclude.end(), false);
    2270                 const unsigned int dpf = fe.dofs_per_face;
    2271
    2272                 for (unsigned int d=0; d<DoFHandlerType::dimension; ++d)
    2273                   {
    2274                     const unsigned int a_face = GeometryInfo<DoFHandlerType::dimension>::vertex_to_face[v][d];
    2275                     const unsigned int face = GeometryInfo<DoFHandlerType::dimension>::opposite_face[a_face];
    2276                     for (unsigned int i=0; i<dpf; ++i)
    2277                       exclude[fe.face_to_cell_index(i,face)] = true;
    2278                   }
    2279                 for (unsigned int j=0; j<indices.size(); ++j)
    2280                   if (!exclude[j])
    2281                     block_list.add(block, indices[j]);
    2282               }
    2283             else
    2284               {
    2285                 for (unsigned int j=0; j<indices.size(); ++j)
    2286                   block_list.add(block, indices[j]);
    2287               }
    2288           }
    2289       }
    2290   }
    2291




  const dealii::DoFHandler<dim> &dof_handler      = this->get_dofh();
  const dealii::Triangulation<dim> &triangulation = dof_handler.get_triangulation();
  if (this->get_level() >= triangulation.n_levels())
    return;
  const unsigned int n_subdomains         = triangulation.n_cells(this->get_level());
  const unsigned int n_subdomain_dofs     = dof_handler.get_fe().dofs_per_cell;
  this->subdomain_to_global_map.reserve(n_subdomains);

  //just store information for locally owned cells
  unsigned int subdomain_idx = 0;
  for (auto cell = dof_handler.begin_mg(this->get_level());
       cell != dof_handler.end_mg(this->get_level());
       ++cell)
    if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
      {
        this->subdomain_to_global_map.push_back(std::vector<dealii::types::global_dof_index>(n_subdomain_dofs));
        cell->get_active_or_mg_dof_indices(this->subdomain_to_global_map[subdomain_idx]);
        ++subdomain_idx;
      }
}

template <int dim>
void DGDDHandlerVertex<dim>::initialize_colorized_iterators()
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
void DGDDHandlerVertex<dim>::initialize_max_n_overlaps()
{
  // no overlaps
  this->max_n_overlaps_ = 0;
}

#include "DDHandler.inst"
