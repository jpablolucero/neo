#include <MFOperator.h>

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::MFOperator()
{
  level = 0;
  dof_handler = nullptr;
  fe = nullptr;
  mapping = nullptr;
  constraints = nullptr;
  timer = nullptr;
  use_cell_range = false;
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::set_timer(dealii::TimerOutput &timer_)
{
  timer = &timer_;
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::~MFOperator()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}

/*template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::clear()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}*/


template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::MFOperator(const MFOperator &operator_)
  : Subscriptor(operator_)
{
  timer = operator_.timer;
  this->reinit(operator_.dof_handler,
               operator_.mapping,
               operator_.constraints,
               operator_.mg_constrained_dofs,
               operator_.mpi_communicator,
               operator_.level);
}


template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::reinit
(const dealii::DoFHandler<dim> *dof_handler_,
 const dealii::MappingQ1<dim> *mapping_,
 const dealii::ConstraintMatrix *constraints_,
 const dealii::MGConstrainedDoFs *mg_constrained_dofs_,
 const MPI_Comm &mpi_communicator_,
 const unsigned int level_)
{
  Assert ((is_system_matrix && level_ == dealii::numbers::invalid_unsigned_int)
          ||(!is_system_matrix && level_ != dealii::numbers::invalid_unsigned_int),
          dealii::ExcMessage("A system matrix is not supposed to has levels."
                             " If not the system matrix represented, a valid level is needed!"));

  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=is_system_matrix?0:level_;
  constraints = constraints_;
  mg_constrained_dofs = mg_constrained_dofs_;
  mpi_communicator = mpi_communicator_;
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

  dealii::IndexSet locally_owned_level_dofs
    = is_system_matrix?dof_handler->locally_owned_dofs():dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  if (is_system_matrix)
    dealii::DoFTools::extract_locally_relevant_dofs
    (*dof_handler, locally_relevant_level_dofs);
  else
    dealii::DoFTools::extract_locally_relevant_level_dofs
    (*dof_handler, level, locally_relevant_level_dofs);

  {
    //Need an additional pseudo-level for adaptivity
    if (!is_system_matrix && level !=0)
      ghosted_src.resize(level-1, level);
    else
      ghosted_src.resize(level, level);

#if PARALLEL_LA == 0
    ghosted_src[level].reinit(locally_owned_level_dofs.n_elements());
    ghosted_dst.reinit(locally_owned_level_dofs.n_elements());
#else
    ghosted_src[level].reinit(locally_owned_level_dofs,
                              locally_relevant_level_dofs,
                              mpi_communicator_);
    ghosted_dst.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,mpi_communicator,true);
#endif

    if (!is_system_matrix && level!=0)
      {
        const dealii::IndexSet locally_owned_lower_level_dofs = dof_handler->locally_owned_mg_dofs(level-1);
        dealii::IndexSet locally_relevant_lower_level_dofs;
        dealii::DoFTools::extract_locally_relevant_level_dofs
        (*dof_handler, level-1, locally_relevant_lower_level_dofs);

#if PARALLEL_LA == 0
        ghosted_src[level-1].reinit(locally_owned_lower_level_dofs);
#else
        ghosted_src[level-1].reinit(locally_owned_lower_level_dofs,
                                    locally_relevant_lower_level_dofs,
                                    mpi_communicator_);
#endif
        ghosted_src[level-1] = 0.;
      }
  }


  //TODO possibly colorize iterators, assume thread-safety for the moment
  std::vector<std::vector<cell_iterator> > all_iterators
  (static_cast<unsigned int>(std::pow(2,dim)));
  if (is_system_matrix)
    {
      auto i = 1 ;
      for (typename dealii::DoFHandler<dim>::active_cell_iterator p=dof_handler->begin_active(level);
           p!=dof_handler->end(); ++p)
        {
          AssertThrow(p->active(), dealii::ExcInternalError());
          const dealii::types::subdomain_id csid = p->subdomain_id();
          if (csid == p->get_triangulation().locally_owned_subdomain())
            {
              all_iterators[i-1].push_back(p);
              i = i % static_cast<unsigned int>(std::pow(2,dim)) ;
              ++i;
            }
        }
    }
  else
    {
      auto i = 1 ;
      for (auto p=dof_handler->begin_mg(level); p!=dof_handler->end_mg(level); ++p)
        {
          const dealii::types::subdomain_id csid = p->level_subdomain_id();
          if (csid == p->get_triangulation().locally_owned_subdomain())
            {
              all_iterators[i-1].push_back(p);
              i = i % static_cast<unsigned int>(std::pow(2,dim)) ;
              ++i;
            }
        }
    }
  colored_iterators = std::move(all_iterators);
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::set_cell_range
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_)
{
  Assert(!is_system_matrix, dealii::ExcMessage("Can't set cell_range for system_matrix!"));
  use_cell_range = true;
  // Do the necessary cast manually
  colored_iterators.clear();
  cell_range.clear();
  for (unsigned int i=0; i<cell_range_.size(); ++i)
    {
      const cell_iterator cell (&(dof_handler->get_triangulation()), level, cell_range_[i]->index(), dof_handler);
      colored_iterators[0].push_back(cell);
      cell_range.push_back(cell);
    }
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::build_coarse_matrix()
{
  Assert(level == 0, dealii::ExcInternalError());
  Assert(dof_handler != 0, dealii::ExcInternalError());

  info_box.initialize(*fe, *mapping, &(dof_handler->block_info()));
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
#endif

  dealii::MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(*mg_constrained_dofs);
#endif

  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler);

  mg_matrix[level].compress(dealii::VectorOperation::add);
#if PARALLEL_LA==0
  coarse_matrix = std::move(mg_matrix[level]);
#else
  coarse_matrix.copy_from(mg_matrix[level]);
#endif
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::vmult (LA::MPI::Vector &dst,
    const LA::MPI::Vector &src) const
{
  vmult_add(dst, src);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::Tvmult (LA::MPI::Vector &/*dst*/,
    const LA::MPI::Vector &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim, fe_degree, same_diagonal, is_system_matrix>::vmult_add (LA::MPI::Vector &dst,
    const LA::MPI::Vector &src) const
{
  if (!use_cell_range)
    timer->enter_subsection("LO::initialize ("+ dealii::Utilities::int_to_string(level)+ ")");
  ghosted_dst = 0.;
  dealii::AnyData dst_data;
  dst_data.add<LA::MPI::Vector *>(&ghosted_dst, "dst");
  ghosted_src[level] = src;
  dealii::AnyData src_data ;
  if (is_system_matrix)
    src_data.add<const LA::MPI::Vector *>(&(ghosted_src[level]),"src");
  else
    src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(&ghosted_src,"src");
  if (!use_cell_range)
    timer->leave_subsection();

  if (!use_cell_range)
    timer->enter_subsection("LO::assembler_setup ("+ dealii::Utilities::int_to_string(level)+ ")");
  if (!is_system_matrix)
    info_box.initialize(*fe, *mapping, src_data, ghosted_src, &(dof_handler->block_info()));
  else
    info_box.initialize(*fe, *mapping, src_data, ghosted_src[level], &(dof_handler->block_info()));

  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > assembler;
  assembler.initialize(dst_data);
  if (!use_cell_range)
    timer->leave_subsection();

  if (!use_cell_range)
    timer->enter_subsection("LO::IntegrationLoop ("+ dealii::Utilities::int_to_string(level)+ ")");
  {
    dealii::MeshWorker::LoopControl lctrl;
    //TODO possibly colorize iterators, assume thread-safety for the moment
    if (use_cell_range)
      {
        lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
        lctrl.ghost_cells = true;
        dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, residual_integrator, assembler,lctrl, cell_range);
      }
    else
      {
        dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, residual_integrator, assembler,lctrl);
      }
  }
  ghosted_dst.compress(dealii::VectorOperation::add);
  dst = ghosted_dst;

  if (!use_cell_range)
    timer->leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal, bool is_system_matrix>
void MFOperator<dim,fe_degree, same_diagonal, is_system_matrix>::Tvmult_add (LA::MPI::Vector &/*dst*/,
    const LA::MPI::Vector &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

#include "MFOperator.inst"
