#include <LaplaceOperator.h>

template <int dim, int fe_degree, bool same_diagonal>
LaplaceOperator<dim, fe_degree, same_diagonal>::LaplaceOperator()
{
  level = 0;
  dof_handler = NULL;
  fe = NULL;
  mapping = NULL;
  constraints = NULL;
  timer = NULL;
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::set_timer(dealii::TimerOutput &timer_)
{
  timer = &timer_;
}

template <int dim, int fe_degree, bool same_diagonal>
LaplaceOperator<dim, fe_degree, same_diagonal>::~LaplaceOperator()
{
  dof_handler = NULL ;
  fe = NULL ;
  mapping = NULL ;
}

/*template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::clear()
{
  dof_handler = NULL ;
  fe = NULL ;
  mapping = NULL ;
}*/


template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim, fe_degree, same_diagonal>::reinit
(dealii::DoFHandler<dim> *dof_handler_,
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
void LaplaceOperator<dim, fe_degree, same_diagonal>::build_matrix ()
{
  Assert(dof_handler != 0, dealii::ExcInternalError());

  timer->enter_subsection("LO::build_matrix");
  info_box.initialize(*fe, *mapping, &(dof_handler->block_info()));
  dealii::MGLevelObject<LA::MPI::SparseMatrix> mg_matrix ;
  mg_matrix.resize(level,level);

  const unsigned int n = dof_handler->get_fe().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> level_dof_indices (n);
  dealii::IndexSet locally_relevant_level_dofs;
  dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler,level,locally_relevant_level_dofs);
  dealii::DynamicSparsityPattern dsp(locally_relevant_level_dofs);

  bool first_cell_found = false;
  typename dealii::DoFHandler<dim>::level_cell_iterator first_cell;

  if (level == 0)
    {
      //for the coarse matrix, we want to assemble always everything
      dealii::MGTools::make_flux_sparsity_pattern(*dof_handler,dsp,level);
      first_cell_found = true;
      first_cell = dof_handler->begin_mg(0);
    }
  else
    {
      //we create a flux_sparsity_pattern without couplings between cells
      for (auto cell = dof_handler->begin_mg(level); cell != dof_handler->end_mg(level); ++cell)
        if (cell->level_subdomain_id() == dof_handler->get_triangulation().locally_owned_subdomain())
          {
            if (!first_cell_found)
              {
                first_cell_found=true;
                first_cell=cell;
              }

            cell->get_active_or_mg_dof_indices (level_dof_indices);
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                {
                  const dealii::types::global_dof_index i1 = level_dof_indices [i];
                  const dealii::types::global_dof_index i2 = level_dof_indices [j];
                  dsp.add(i1, i2);
                }
            if (same_diagonal)
              {
                //if we use just one cell, we only need to allow for storing one cell
                break;
              }
            else
              {
                // Loop over all interior neighbors
                for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
                  {
                    if ( (! cell->at_boundary(face)) &&
                         (static_cast<unsigned int>(cell->neighbor_level(face)) == level) )
                      {
                        const typename dealii::DoFHandler<dim>::level_cell_iterator neighbor = cell->neighbor(face);
                        neighbor->get_active_or_mg_dof_indices (level_dof_indices);
                        if (neighbor->is_locally_owned_on_level() == false)
                          for (unsigned int i=0; i<n; ++i)
                            for (unsigned int j=0; j<n; ++j)
                              dsp.add (level_dof_indices[i], level_dof_indices[j]);
                      }
                  }
              }
          }
    }
  AssertThrow(first_cell_found || dof_handler->locally_owned_mg_dofs(level).n_elements()==0,
              dealii::ExcInternalError());

#if PARALLEL_LA == 0
  sp.copy_from (dsp);
  mg_matrix[level].reinit(sp);
#else
  mg_matrix[level].reinit(dof_handler->locally_owned_mg_dofs(level),
                          dof_handler->locally_owned_mg_dofs(level),
                          dsp,mpi_communicator);
#endif

  if (first_cell_found)
    {
      dealii::MeshWorker::Assembler::MGMatrixSimple<LA::MPI::SparseMatrix> assembler;
      assembler.initialize(mg_matrix);
#ifdef CG
      assembler.initialize(constraints);
#endif

      typename dealii::DoFHandler<dim>::level_cell_iterator end_cell;
      if (!same_diagonal || level==0)
        end_cell = dof_handler->end_mg(level);
      else
        {
          end_cell = first_cell;
          ++end_cell;
        }
      dealii::MeshWorker::integration_loop<dim, dim> (first_cell,
                                                      end_cell,
                                                      *dof_info, info_box,
                                                      matrix_integrator, assembler);
    }
  mg_matrix[level].compress(dealii::VectorOperation::add);
#if PARALLEL_LA==0
  matrix = std::move(mg_matrix[level]);
#else
  matrix.copy_from(mg_matrix[level]);
#endif

  timer->leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::vmult (LA::MPI::Vector &dst,
                                                          const LA::MPI::Vector &src) const
{
  dst = 0;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::Tvmult (LA::MPI::Vector &dst,
                                                           const LA::MPI::Vector &src) const
{
  dst = 0;
  Tvmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree,same_diagonal>::vmult_add (LA::MPI::Vector &dst,
    const LA::MPI::Vector &src) const
{
  timer->enter_subsection("LO::initialize ("+ dealii::Utilities::int_to_string(level)+ ")");
  dealii::AnyData dst_data;
  dst_data.add<LA::MPI::Vector *>(&dst, "dst");
  ghosted_src[level] = src;
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
  dealii::MeshWorker::integration_loop<dim, dim>
  (dof_handler->begin_mg(level), dof_handler->end_mg(level),
   *dof_info,info_box,residual_integrator,assembler);
  timer->leave_subsection();
}

template <int dim, int fe_degree, bool same_diagonal>
void LaplaceOperator<dim,fe_degree, same_diagonal>::Tvmult_add (LA::MPI::Vector &dst,
    const LA::MPI::Vector &src) const
{
  dst = 0;
  vmult_add(dst, src);
  AssertIsFinite(dst.l2_norm());
}

#include "LaplaceOperator.inst"
