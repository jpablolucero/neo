#include <DDHandler.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/graph_coloring.h>



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
 *** DGDDHandler
 ***/

template <int dim>
void DGDDHandler<dim>::initialize_subdomain_to_global_map()
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
void DGDDHandler<dim>::initialize_colorized_iterators()
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
void DGDDHandler<dim>::initialize_max_n_overlaps()
{
  // no overlaps
  this->max_n_overlaps_ = 0;
}

#include "DDHandler.inst"
