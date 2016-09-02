#ifndef MATRIXFREE

#include <MWOperator.h>

template <int dim, int fe_degree, typename number>
MWOperator<dim,fe_degree,number>::MWOperator()
{
  level = 0;
  dof_handler = nullptr;
  fe = nullptr;
  mapping = nullptr;
  constraints = nullptr;
  timer = nullptr;
  use_cell_range = false;
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::set_timer(dealii::TimerOutput &timer_)
{
  timer = &timer_;
}

template <int dim, int fe_degree, typename number>
MWOperator<dim,fe_degree,number>::~MWOperator()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}

template <int dim, int fe_degree, typename number>
MWOperator<dim,fe_degree,number>::MWOperator(const MWOperator &operator_)
  : Subscriptor(operator_)
{
  timer = operator_.timer;
  this->reinit(operator_.dof_handler,
               operator_.mapping,
               operator_.constraints,
               operator_.mpi_communicator,
               operator_.level);
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::reinit
(const dealii::DoFHandler<dim> *dof_handler_,
 const dealii::Mapping<dim> *mapping_,
 const dealii::ConstraintMatrix *constraints_,
 const MPI_Comm &mpi_communicator_,
 const unsigned int level_)
{
  timer->enter_subsection("MWOperator::reinit");
  // Initialize member variables
  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=level_;
  constraints = constraints_;
  mpi_communicator = mpi_communicator_;

  // Setup DoFInfo & IntegrationInfoBox
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> > tmp
  (new dealii::MeshWorker::DoFInfo<dim> {dof_handler->block_info()});
  dof_info = std::move(tmp);
  Assert(fe->degree == fe_degree, dealii::ExcInternalError());
  const unsigned int n_gauss_points = dof_handler->get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
                                       n_gauss_points,
                                       n_gauss_points);
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_values |
                                     dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);
  // Setup "colorized iterators"
  // TODO possibly colorize iterators, assume thread-safety for the moment
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >
  all_iterators(static_cast<unsigned int>(std::pow(2,dim)));
  auto i = 1 ;
  for (auto p=dof_handler->begin_mg(level); p!=dof_handler->end_mg(level); ++p)
    {
      const dealii::types::subdomain_id csid = (p->is_level_cell())
                                               ? p->level_subdomain_id()
                                               : p->subdomain_id();
      if (csid == p->get_triangulation().locally_owned_subdomain())
        {
          all_iterators[i-1].push_back(p);
          i = i % static_cast<unsigned int>(std::pow(2,dim)) ;
          ++i;
        }
    }
  colored_iterators = std::move(all_iterators);

  // Initialize ghosted src
  ghosted_src.resize(level, level);
#if PARALLEL_LA != 0
  dealii::IndexSet locally_owned_level_dofs = dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs
  (*dof_handler, level, locally_relevant_level_dofs);
  ghosted_src[level].reinit(locally_owned_level_dofs,
                            locally_relevant_level_dofs,
                            mpi_communicator_);
#else // PARALLEL_LA == 0
  ghosted_src[level].reinit(locally_owned_level_dofs.n_elements());
#endif // PARALLEL_LA
  timer->leave_subsection();
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::set_cell_range
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_)
{
  use_cell_range = true;
  cell_range = &cell_range_;
  colored_iterators[0] = *cell_range;
}

#if PARALLEL_LA < 3
template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::build_coarse_matrix()
{
  Assert(dof_handler != 0, dealii::ExcInternalError());
  info_box.initialize( *fe, *mapping, &(dof_handler->block_info()));
  dealii::MGLevelObject<LA::MPI::SparseMatrix> mg_matrix ;
  mg_matrix.resize(level,level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler,level,locally_relevant_level_dofs);
  dealii::DynamicSparsityPattern dsp(locally_relevant_level_dofs);
  //for the coarse matrix, we want to assemble always everything
  dealii::MGTools::make_flux_sparsity_pattern(*dof_handler,dsp,level);
#if PARALLEL_LA == 0
  sp.copy_from (dsp);
  mg_matrix[level].reinit(sp);
#else
  mg_matrix[level].reinit(dof_handler->locally_owned_mg_dofs(level),
                          dof_handler->locally_owned_mg_dofs(level),
                          dsp,mpi_communicator);
#endif // PARALLEL_LA
  dealii::MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(constraints);
#endif // CG
  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler);
  mg_matrix[level].compress(dealii::VectorOperation::add);
#if PARALLEL_LA==0
  coarse_matrix = std::move(mg_matrix[level]);
#else
  coarse_matrix.copy_from(mg_matrix[level]);
#endif // PARALLEL_LA
}

#endif // PARALLEL_LA < 3

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::vmult (LA::MPI::Vector &dst,
                                              const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::Tvmult (LA::MPI::Vector &dst,
                                               const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::vmult_add (LA::MPI::Vector &dst,
                                                  const LA::MPI::Vector &src) const
{
  timer->enter_subsection("MWOperator::initialize ("+ dealii::Utilities::int_to_string(level)+ ")");
  // Initialize MPI vectors
  ghosted_src[level] = std::move(src);
  dealii::IndexSet locally_owned_level_dofs = dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs
  (*dof_handler, level, locally_relevant_level_dofs);
#if PARALLEL_LA == 3
  ghosted_src[level].update_ghost_values();
  dst.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,mpi_communicator);
#elif PARALLEL_LA == 2
  dst.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,mpi_communicator,true);
#endif // PARALLEL_LA
  // Setup AnyData
  dealii::AnyData dst_data;
  dst_data.add<LA::MPI::Vector *>(&dst, "dst");
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(&ghosted_src,"src");
  timer->leave_subsection();

  timer->enter_subsection("MWOperator::assembler_setup ("+ dealii::Utilities::int_to_string(level)+ ")");
  info_box.initialize(*fe, *mapping, src_data, src, &(dof_handler->block_info()));
  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > assembler;
  assembler.initialize(dst_data);
  timer->leave_subsection();

  timer->enter_subsection("MWOperator::loop ("+ dealii::Utilities::int_to_string(level)+ ")");
  {
    dealii::MeshWorker::LoopControl lctrl;
    //TODO possibly colorize iterators, assume thread-safety for the moment
    if (use_cell_range)
      {
        lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
        lctrl.ghost_cells = true;
        dealii::colored_loop<dim, dim> (colored_iterators,
                                        *dof_info,
                                        info_box,
                                        residual_integrator,
                                        assembler,
                                        lctrl,
                                        colored_iterators[0]);
      }
    else
      {
        dealii::colored_loop<dim, dim> (colored_iterators,
                                        *dof_info,
                                        info_box,
                                        residual_integrator,
                                        assembler,
                                        lctrl);
      }
  }
  timer->leave_subsection();
}

template <int dim, int fe_degree, typename number>
void MWOperator<dim,fe_degree,number>::Tvmult_add
(LA::MPI::Vector &dst,
 const LA::MPI::Vector &src) const
{
  vmult_add(dst, src);
}

#include "MWOperator.inst"

#endif // MATRIXFREE