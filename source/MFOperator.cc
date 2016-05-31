#include <MFOperator.h>

template <int dim, int fe_degree, bool same_diagonal>
MFOperator<dim, fe_degree, same_diagonal>::MFOperator()
{
  level = 0;
  dof_handler = nullptr;
  fe = nullptr;
  mapping = nullptr;
  constraints = nullptr;
  timer = nullptr;
  use_cell_range = false;
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim, fe_degree, same_diagonal>::set_timer(dealii::TimerOutput &timer_)
{
  timer = &timer_;
}

template <int dim, int fe_degree, bool same_diagonal>
MFOperator<dim, fe_degree, same_diagonal>::~MFOperator()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}

/*template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim, fe_degree, same_diagonal>::clear()
{
  dof_handler = nullptr ;
  fe = nullptr ;
  mapping = nullptr ;
}*/


template <int dim, int fe_degree, bool same_diagonal>
MFOperator<dim, fe_degree, same_diagonal>::MFOperator(const MFOperator &operator_)
  : Subscriptor(operator_)
{
  timer = operator_.timer;
  this->reinit(operator_.dof_handler,
               operator_.mapping,
               operator_.constraints,
               operator_.mpi_communicator,
               operator_.level);
}


template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim, fe_degree, same_diagonal>::reinit
(const dealii::DoFHandler<dim> *dof_handler_,
 const dealii::MappingQ1<dim> *mapping_,
 const dealii::ConstraintMatrix *constraints_,
 const MPI_Comm &mpi_communicator_,
 const unsigned int level_)
{
  timer->enter_subsection("LO::reinit");
  dof_handler = dof_handler_ ;
  fe = &(dof_handler->get_fe());
  mapping = mapping_ ;
  level=level_;
  constraints = constraints_;
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

  dealii::IndexSet locally_owned_level_dofs = dof_handler->locally_owned_mg_dofs(level);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs
  (*dof_handler, level, locally_relevant_level_dofs);

  ghosted_src.resize(level, level);
#if PARALLEL_LA == 0
  ghosted_src[level].reinit(locally_owned_level_dofs.n_elements());
#else
  ghosted_src[level].reinit(locally_owned_level_dofs,
                            locally_relevant_level_dofs,
                            mpi_communicator_);
#endif
  timer->leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim, fe_degree, same_diagonal>::set_cell_range
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range_)
{
  use_cell_range = true;
  cell_range = &cell_range_;
  residual_integrator.set_cell_range(*cell_range);
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim, fe_degree, same_diagonal>::build_matrix
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range)
{
  Assert(dof_handler != 0, dealii::ExcInternalError());

  info_box.initialize(*fe, *mapping, &(dof_handler->block_info()));
  dealii::MGLevelObject<LA::MPI::SparseMatrix> mg_matrix ;
  mg_matrix.resize(level,level);

  const unsigned int n = dof_handler->get_fe().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> level_dof_indices (n);
  std::vector<dealii::types::global_dof_index> neighbor_dof_indices (n);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler,level,locally_relevant_level_dofs);
  dealii::DynamicSparsityPattern dsp(locally_relevant_level_dofs);

  if (level == 0)
    {
      //for the coarse matrix, we want to assemble always everything
      dealii::MGTools::make_flux_sparsity_pattern(*dof_handler,dsp,level);
    }
  else
    {
      //we create a flux_sparsity_pattern
      for (auto cell = cell_range.begin(); cell != cell_range.end(); ++cell)
        {
          (*cell)->get_active_or_mg_dof_indices (level_dof_indices);
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                const dealii::types::global_dof_index i1 = level_dof_indices [i];
                const dealii::types::global_dof_index i2 = level_dof_indices [j];
                dsp.add(i1, i2);
              }
          // Loop over all interior neighbors
          for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
            if ( (! (*cell)->at_boundary(face)))
              {

                // TODO: normally, we want to allow only entries related to the patch
                const typename dealii::DoFHandler<dim>::level_cell_iterator neighbor = (*cell)->neighbor(face);
//                const int neighbor_index = neighbor->index();

//                bool inner_face = false;

//                for (unsigned int i=0; i<cell_range.size(); ++i)
//                  if (cell_range[i]->index() == neighbor_index)
//                    {
//                      inner_face = true;
//                      break;
//                    }

//                if (inner_face)
                {
                  neighbor->get_active_or_mg_dof_indices (neighbor_dof_indices);
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int j=0; j<n; ++j)
                      {
                        dsp.add (level_dof_indices[i], neighbor_dof_indices[j]);
                        //TODO: Remove the next two
                        dsp.add (neighbor_dof_indices[i], level_dof_indices[j]);
                        dsp.add (neighbor_dof_indices[i], neighbor_dof_indices[j]);
                      }
                }
              }
        }
    }

//  dsp.print(std::cout);

#if PARALLEL_LA == 0
  sp.copy_from (dsp);
  mg_matrix[level].reinit(sp);
#else
  if (level==0)
    mg_matrix[level].reinit(dof_handler->locally_owned_mg_dofs(level),
                            dof_handler->locally_owned_mg_dofs(level),
                            dsp,mpi_communicator);
  else
    {
      dealii::IndexSet relevant_mg_dofs;
      dealii::DoFTools::extract_locally_relevant_level_dofs
      (*dof_handler, level, relevant_mg_dofs);
      mg_matrix[level].reinit(relevant_mg_dofs,
                              dsp,MPI_COMM_SELF);
    }
  matrix_integrator.set_cell_range(cell_range);
#endif

  dealii::MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(constraints);
#endif


  //now assemble everything
  if (cell_range.size()==0)
    {
      dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(level),
                                                      dof_handler->end_mg(level),
                                                      *dof_info, info_box,
                                                      matrix_integrator, assembler);

    }
  else
    {
      dealii::MeshWorker::LoopControl lctrl;
      //assemble faces from both sides
      //lctrl.own_faces = dealii::MeshWorker::LoopControl::both; //crucial for Cell smoother!
      lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
      lctrl.ghost_cells = true;

      dealii::integration_loop<dim, dim> (cell_range, *dof_info, info_box,
                                          matrix_integrator, assembler, lctrl);
    }

  mg_matrix[level].compress(dealii::VectorOperation::add);
#if PARALLEL_LA==0
  matrix = std::move(mg_matrix[level]);
#else
  matrix.copy_from(mg_matrix[level]);
#endif
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim,fe_degree,same_diagonal>::vmult (LA::MPI::Vector &dst,
                                                     const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim,fe_degree,same_diagonal>::Tvmult (LA::MPI::Vector &dst,
                                                      const LA::MPI::Vector &src) const
{
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim,fe_degree,same_diagonal>::vmult_add (LA::MPI::Vector &dst,
                                                         const LA::MPI::Vector &src) const
{
  timer->enter_subsection("LO::initialize ("+ dealii::Utilities::int_to_string(level)+ ")");
  dealii::AnyData dst_data;
  dst_data.add<LA::MPI::Vector *>(&dst, "dst");
  ghosted_src[level] = std::move(src);
  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(&ghosted_src,"src");
  timer->leave_subsection();

  timer->enter_subsection("LO::assembler_setup ("+ dealii::Utilities::int_to_string(level)+ ")");
  info_box.initialize(*fe, *mapping, src_data, ghosted_src, &(dof_handler->block_info()));
  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > assembler;
  assembler.initialize(dst_data);
//  assembler.initialize(*constraints);
  timer->leave_subsection();

  timer->enter_subsection("LO::IntegrationLoop ("+ dealii::Utilities::int_to_string(level)+ ")");
  if (!use_cell_range)
    {
      dealii::MeshWorker::integration_loop<dim, dim>
      (dof_handler->begin_mg(level), dof_handler->end_mg(level),
       *dof_info,info_box,residual_integrator,assembler);
    }
  else
    {
      dealii::MeshWorker::DoFInfoBox<dim, dealii::MeshWorker::DoFInfo<dim> > dof_info_box(*dof_info);
      assembler.initialize_info(dof_info_box.cell, false);
      for (unsigned int i=0; i<dealii::GeometryInfo<dim>::faces_per_cell; ++i)
        {
          assembler.initialize_info(dof_info_box.interior[i], true);
          assembler.initialize_info(dof_info_box.exterior[i], true);
        }

      auto cell_worker  = dealii::std_cxx11::bind(&dealii::MeshWorker::LocalIntegrator<dim>::cell, &residual_integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2);
      auto boundary_worker = dealii::std_cxx11::bind(&dealii::MeshWorker::LocalIntegrator<dim>::boundary, &residual_integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2);
      auto face_worker = dealii::std_cxx11::bind(&dealii::MeshWorker::LocalIntegrator<dim>::face, &residual_integrator, dealii::std_cxx11::_1, dealii::std_cxx11::_2, dealii::std_cxx11::_3, dealii::std_cxx11::_4);

      dealii::MeshWorker::LoopControl lctrl;
      //assemble faces from both sides
      lctrl.own_faces = dealii::MeshWorker::LoopControl::both;

      // Loop over all cells
      for (unsigned int i=0; i<cell_range->size(); ++i)
        {
          dealii::MeshWorker::cell_action<dealii::MeshWorker::IntegrationInfoBox<dim>,
                 dealii::MeshWorker::DoFInfo<dim>, dim, dim >
                 ((*cell_range)[i], dof_info_box, info_box, cell_worker,
                  boundary_worker, face_worker, lctrl);
          dof_info_box.assemble(assembler);
        }
    }
  timer->leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void MFOperator<dim,fe_degree, same_diagonal>::Tvmult_add (LA::MPI::Vector &dst,
                                                           const LA::MPI::Vector &src) const
{
  vmult_add(dst, src);
}

#include "MFOperator.inst"
