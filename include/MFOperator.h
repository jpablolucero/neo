#ifndef MFOPERATOR_H
#define MFOPERATOR_H

#include <deal.II/base/std_cxx11/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <GenericLinearAlgebra.h>
#include <Integrators.h>
#include <DDHandler.h>

template <int dim, int fe_degree, bool same_diagonal>
class MFOperator final: public dealii::Subscriptor
{
public:
  MFOperator () ;
  ~MFOperator () ;
  MFOperator (const MFOperator &) = delete ;
  MFOperator &operator = (const MFOperator &) = delete;

  void reinit (dealii::DoFHandler<dim> *dof_handler_,
               const dealii::MappingQ1<dim> *mapping_,
               const dealii::ConstraintMatrix *constraints,
               const MPI_Comm &mpi_communicator_,
               const unsigned int level_ = dealii::numbers::invalid_unsigned_int);

  void set_timer (dealii::TimerOutput &timer_);

  void build_matrix () ;

  void clear () ;

  void vmult (LA::MPI::Vector &dst,
              const LA::MPI::Vector &src) const ;
  void Tvmult (LA::MPI::Vector &dst,
               const LA::MPI::Vector &src) const ;
  void vmult_add (LA::MPI::Vector &dst,
                  const LA::MPI::Vector &src) const ;
  void Tvmult_add (LA::MPI::Vector &dst,
                   const LA::MPI::Vector &src) const ;

  typedef double value_type ;

  const LA::MPI::SparseMatrix &get_coarse_matrix() const
  {
    return matrix;
  }

  unsigned int m() const
  {
    return dof_handler->n_dofs(level);
  }

  unsigned int n() const
  {
    return dof_handler->n_dofs(level);
  }

  typedef LA::MPI::SparseMatrixSizeType      size_type ;

  double operator()(const size_type i,const size_type j) const
  {
    return matrix(i,j);
  }

private:
  typedef dealii::MeshWorker::DoFInfo<dim> DOFINFO ;
  typedef dealii::MeshWorker::IntegrationInfoBox<dim> INFOBOX ;
  typedef dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > ASSEMBLER ;
  typedef typename dealii::DoFHandler< dim, dim >::level_cell_iterator ITERATOR ;
  typedef dealii::MeshWorker::LocalIntegrator<dim, dim> INTEGRATOR ;

  unsigned int                                        level;
  dealii::DoFHandler<dim>                             *dof_handler;
  const dealii::FiniteElement<dim>                    *fe;
  const dealii::MappingQ1<dim>                        *mapping;
  const dealii::ConstraintMatrix                      *constraints;
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> >  dof_info;
  mutable dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
#if PARALLEL_LA == 0
  dealii::SparsityPattern                             sp;
#endif
  LA::MPI::SparseMatrix                               matrix;
  MatrixIntegrator<dim,same_diagonal>                 matrix_integrator;
  ResidualIntegrator<dim>                             residual_integrator;
  mutable dealii::MGLevelObject<LA::MPI::Vector>      ghosted_src;
  MPI_Comm                                            mpi_communicator;
  dealii::TimerOutput                                 *timer;
  std::vector<std::vector<ITERATOR> >                 colored_iterators;
};

#ifdef HEADER_IMPLEMENTATION
#include <MFOperator.cc>
#endif

namespace internal
{
  template<int dim, int spacedim, typename ITERATOR, typename DOFINFO, typename INFOBOX, typename INTEGRATOR, typename ASSEMBLER>
  void colored_loop(const std::vector<std::vector<ITERATOR> > colored_iterators,
		    DOFINFO  &dof_info,
		    INFOBOX  &info,
		    const INTEGRATOR &integrator,
		    ASSEMBLER  &assembler,
		    const dealii::MeshWorker::LoopControl &lctrl = dealii::MeshWorker::LoopControl())
  {
    const dealii::std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)>   cell_worker 
      = dealii::std_cxx11::bind(&INTEGRATOR::cell, &integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2);
    const dealii::std_cxx11::function<void (DOFINFO &, typename INFOBOX::CellInfo &)>   boundary_worker 
      = dealii::std_cxx11::bind(&INTEGRATOR::boundary, &integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2);
    const dealii::std_cxx11::function<void (DOFINFO &, DOFINFO &,
					    typename INFOBOX::CellInfo &,
					    typename INFOBOX::CellInfo &)>   face_worker
      = dealii::std_cxx11::bind(&INTEGRATOR::face, &integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2,
				dealii::std_cxx11::_3, dealii::std_cxx11::_4);

    dealii::MeshWorker::DoFInfoBox<dim, DOFINFO> dinfo_box(dof_info);

    assembler.initialize_info(dinfo_box.cell, false);
    for (unsigned int i=0; i<dealii::GeometryInfo<dim>::faces_per_cell; ++i)
      {
	assembler.initialize_info(dinfo_box.interior[i], true);
	assembler.initialize_info(dinfo_box.exterior[i], true);
      }

    //  Loop over all cells                                                                                                          
#ifdef DEAL_II_MESHWORKER_PARALLEL
    dealii::WorkStream::run(colored_iterators,
			    dealii::std_cxx11::bind(&dealii::MeshWorker::cell_action<INFOBOX, DOFINFO, dim, spacedim, ITERATOR>,
						    dealii::std_cxx11::_1, dealii::std_cxx11::_3, dealii::std_cxx11::_2,
						    cell_worker, boundary_worker, face_worker, lctrl),
			    dealii::std_cxx11::bind(&dealii::internal::assemble<dim,DOFINFO,ASSEMBLER>,
						    dealii::std_cxx11::_1, &assembler),
			    info, dinfo_box,
			    dealii::MultithreadInfo::n_threads(),8);

#else
      for (unsigned int color=0; color<colored_iterators.size(); ++color)
	for (typename std::vector<ITERATOR>::const_iterator p = colored_iterators[color].begin();
	     p != colored_iterators[color].end(); ++p)
	  {
	    dealii::MeshWorker::cell_action<INFOBOX,DOFINFO,dim,spacedim>(*p, dinfo_box, info,
									  cell_worker, boundary_worker, face_worker,
									  lctrl);
	    dinfo_box.assemble(assembler);
	  }
#endif
  }
} // end namespace internal

#endif // MFOPERATOR_H
