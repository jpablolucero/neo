#ifndef DDHANDLER_H
#define DDHANDLER_H

#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>



/* Base class for domain decompositions used in the Parallel Subspace
 * Correction Preconditioner.
 */
template<int dim, class number>
class DDHandlerBase
{
  public:
    typedef std::vector<unsigned int>::const_iterator iterator;
    DDHandlerBase();
    DDHandlerBase(const DDHandlerBase<dim, number>& ddh);
    virtual ~DDHandlerBase() = default;

    void initialize(const dealii::DoFHandler<dim>& dofh,
                    unsigned int level = dealii::numbers::invalid_unsigned_int);

    unsigned int size() const;
    unsigned int n_subdomain_dofs(const unsigned int subdomain_idx) const;
    unsigned int get_level() const;
    const dealii::DoFHandler<dim>&  get_dofh() const;
    const std::vector<std::vector<iterator> >&
      colorized_iterators() const;
    unsigned int max_n_overlaps() const;

    void reinit(dealii::Vector<number>& vec,
                const unsigned int subdomain_idx) const;
    void restrict_add(dealii::Vector<number>& dst,
                      const dealii::Vector<number>& src,
                      const unsigned int subdomain_idx) const;
    void prolongate_add(dealii::Vector<number>& dst,
                        const dealii::Vector<number>& src,
                        const unsigned int subdomain_idx) const;

    std::vector<unsigned int>
      global_dofs_on_subdomain(const unsigned int subdomain_idx) const;

  protected:
    std::vector<std::vector<unsigned int> > subdomain_to_global_map;
    std::vector<unsigned int> subdomain_iterators;
    std::vector<std::vector<iterator> > colorized_iterators_;
    unsigned int max_n_overlaps_;

    // Main work function that has to be implemented. Initialize the
    // subdomain_to_global_map member to map from subdomain index to the
    // global dofs that live there
    virtual void initialize_subdomain_to_global_map() = 0;
    // Initialize colorized_iterators_ such that subdomains of the same color
    // do not share global dofs. By default this calls make_graph_coloring
    // which can be expensive, thus it can be overriden for simpler
    // implementations.
    virtual void initialize_colorized_iterators();
    // Calculate the maximum number of subdomains that share a dof and store
    // it in max_n_overlaps_. Again, the default implementation can be
    // expensive and thus it can be overridden.
    virtual void initialize_max_n_overlaps();

  private:
    unsigned int level;
    const dealii::DoFHandler<dim>* dofh;
};



/* Domain Decomposition Handler implementation for
 *  - Discontinuous Elements
 *  - Subdomains consisting of single cells
 */
template<int dim, class number>
class DGDDHandler : public DDHandlerBase<dim, number>
{
  public:
    using typename DDHandlerBase<dim, number>::iterator;
    
  protected:
    virtual void initialize_subdomain_to_global_map();
    virtual void initialize_colorized_iterators();
    virtual void initialize_max_n_overlaps();
};


#endif
