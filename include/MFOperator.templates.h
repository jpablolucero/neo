#include <MFOperator.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

template <int dim, int fe_degree, typename number, typename VectorType>
MFOperator<dim,fe_degree,number,VectorType>::MFOperator()
{
  level = 0;
  dof_handler = nullptr;
  fe = nullptr;
  mapping = nullptr;
  constraints = nullptr;
  use_cell_range = false;
  selected_iterators = nullptr;
}

template <int dim, int fe_degree, typename number, typename VectorType>
MFOperator<dim,fe_degree,number,VectorType>::~MFOperator()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
  selected_iterators = nullptr;
}

template <int dim, int fe_degree, typename number, typename VectorType>
MFOperator<dim,fe_degree,number,VectorType>::MFOperator(const MFOperator &operator_)
  : Subscriptor(operator_)
{
  this->reinit(operator_.dof_handler,
               operator_.mapping,
               operator_.constraints,
               operator_.level,
	       *(operator_.solution));
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::reinit
(const dealii::DoFHandler<dim> *dof_handler_,
 const dealii::Mapping<dim> *mapping_,
 const dealii::ConstraintMatrix *constraints_,
 const unsigned int level_,
 VectorType& solution_)
{
  timer->enter_subsection("MFOperator::reinit");
  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=level_;
  constraints = constraints_;
  solution = &solution_ ;
  ddh.initialize(*dof_handler, level);
  std::unique_ptr<dealii::MeshWorker::DoFInfo<dim> > tmp
  (new dealii::MeshWorker::DoFInfo<dim> {dof_handler->block_info()});
  dof_info = std::move(tmp);
  Assert(fe->degree == fe_degree, dealii::ExcInternalError());
  const unsigned int n_gauss_points = dof_handler->get_fe().degree+1;
  dealii::UpdateFlags update_flags = dealii::update_JxW_values | dealii::update_quadrature_points | dealii::update_values |
                                     dealii::update_gradients;
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
  info_box.initialize_update_flags();
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);
  zero_info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
  zero_info_box.initialize_update_flags();
  zero_info_box.add_update_flags(update_flags, true, true, true, true);
  zero_info_box.cell_selector.add("src", true, true, false);
  zero_info_box.boundary_selector.add("src", true, true, false);
  zero_info_box.face_selector.add("src", true, true, false);
  zero_info_box.cell_selector.add("Newton iterate", true, true, false);
  zero_info_box.boundary_selector.add("Newton iterate", true, true, false);
  zero_info_box.face_selector.add("Newton iterate", true, true, false);
  dealii::IndexSet locally_owned_level_dofs = dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler, level, locally_relevant_level_dofs);
  ghosted_src.resize(level, level);
  ghosted_src[level].reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  ghosted_src[level].update_ghost_values();
  ghosted_solution.resize(level, level);
  ghosted_solution[level].reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  ghosted_solution[level] = solution_ ;
  ghosted_solution[level].update_ghost_values();
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_src,"src");
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_solution,"Newton iterate");
  info_box.initialize(*fe, *mapping, src_data, VectorType{}, &(dof_handler->block_info()));
  zero_src.resize(level, level);
  zero_src[level].reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  zero_src[level].update_ghost_values();
  dealii::AnyData zero_src_data ;
  zero_src_data.add<const dealii::MGLevelObject<VectorType >*>(&zero_src,"src");
  zero_src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_solution,"Newton iterate");
  zero_info_box.initialize(*fe, *mapping, zero_src_data, VectorType{}, &(dof_handler->block_info()));
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >
    all_iterators(static_cast<unsigned int>(std::pow(2,dim)));
  auto i = 1 ;
  for (auto p=dof_handler->begin_mg(level); p!=dof_handler->end_mg(level); ++p)
    {
      const dealii::types::subdomain_id csid = (p->is_level_cell()) ? p->level_subdomain_id() : p->subdomain_id();
      if (csid == p->get_triangulation().locally_owned_subdomain())
        {
	  all_iterators[i-1].push_back(p);
          i = i % static_cast<unsigned int>(std::pow(2,dim)) ;
          ++i;
        }
    }
  colored_iterators = std::move(all_iterators);
  timer->leave_subsection();
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::set_cell_range
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_)
{
  use_cell_range = true;
  selected_iterators = &cell_range_ ;
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::unset_cell_range()
{
  use_cell_range = false;
  selected_iterators = nullptr ;
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::set_subdomain(unsigned int subdomain_idx_)
{
  subdomain_idx = subdomain_idx_;
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::build_coarse_matrix()
{
  Assert(dof_handler != 0, dealii::ExcInternalError());
  dealii::MGLevelObject<dealii::SparseMatrix<double> > mg_matrix ;
  mg_matrix.resize(level,level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler,level,locally_relevant_level_dofs);
  dealii::DynamicSparsityPattern dsp(locally_relevant_level_dofs);
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_src,"src");
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_solution,"Newton iterate");
  info_box.initialize(*fe, *mapping, src_data, VectorType {}, &(dof_handler->block_info()));
  dealii::MGTools::make_flux_sparsity_pattern(*dof_handler,dsp,level);
  sp.copy_from (dsp);
  mg_matrix[level].reinit(sp);
  dealii::MeshWorker::Assembler::MGMatrixSimple<dealii::SparseMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(*mg_constrained_dofs);
#endif // CG
  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler);
  mg_matrix[level].compress(dealii::VectorOperation::add);
  coarse_matrix = std::move(mg_matrix[level]);
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::vmult (VectorType &dst,
							 const VectorType &src) const
{
  dst = 0.;
  dealii::IndexSet locally_owned_level_dofs = dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs
    (*dof_handler, level, locally_relevant_level_dofs);
  dst.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::Tvmult (VectorType &dst,
							  const VectorType &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::vmult_add (VectorType &dst,
                                                  const VectorType &src) const
{
  std::swap(ghosted_src[level],*(const_cast<VectorType*>(&src)));
  std::swap(ghosted_solution[level],*solution);
  dealii::AnyData dst_data;
  dst_data.add<VectorType *>(&dst, "dst");
  dealii::MeshWorker::Assembler::ResidualSimple<VectorType > assembler;
  assembler.initialize(dst_data);
  dealii::MeshWorker::LoopControl lctrl;
  if (use_cell_range)
    {
      lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
      lctrl.own_faces = dealii::MeshWorker::LoopControl::both ;
      dealii::colored_loop<dim, dim> (colored_iterators,*dof_info,info_box,residual_integrator,assembler,lctrl,*selected_iterators);
    }
  else
    dealii::colored_loop<dim, dim> (colored_iterators,*dof_info,info_box,residual_integrator,assembler,lctrl);
  std::swap(*(const_cast<VectorType*>(&src)),ghosted_src[level]);
  std::swap(*solution,ghosted_solution[level]);
}

template <int dim, int fe_degree, typename number, typename VectorType>
void
MFOperator<dim,fe_degree,number,VectorType>::Tvmult_add (VectorType &dst,
                                              const VectorType &src) const
{
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::vmult (dealii::Vector<double> &dst,
							 const dealii::Vector<double> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number, typename VectorType>
void MFOperator<dim,fe_degree,number,VectorType>::vmult_add (dealii::Vector<double> &local_dst,
							     const dealii::Vector<double> &local_src) const
{
  dealii::Vector<double> zero(local_src.size());
  ddh.prolongate(zero_src[level],local_src,subdomain_idx);
  dealii::AnyData dst_data;
  dst_data.add<dealii::Vector<double> *>(&local_dst, "dst");
  Assembler::ResidualSimpleMapped<dealii::Vector<double> > assembler;
  assembler.initialize(dst_data);
  assembler.initialize(ddh.all_to_unique[subdomain_idx]);
  dealii::MeshWorker::LoopControl lctrl;
  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
  lctrl.own_faces = dealii::MeshWorker::LoopControl::both ;
  dealii::colored_loop<dim, dim> (colored_iterators,*dof_info,zero_info_box,residual_integrator,assembler,lctrl,*selected_iterators);
  ddh.prolongate(zero_src[level],zero,subdomain_idx);
}
