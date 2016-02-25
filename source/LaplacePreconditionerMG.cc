#include <LaplacePreconditionerMG.h>

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::LaplacePreconditionerMG()
{}

template<int dim, int fe_degree, typename number>
LaplacePreconditionerMG<dim, fe_degree, number>::~LaplacePreconditionerMG()
{
  dof_handler = NULL ;
  fe = NULL ;
  triangulation = NULL ;
  mapping = NULL ;
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::clear ()
{}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::reinit (dealii::DoFHandler<dim> * dof_handler_,
							    dealii::FE_DGQ<dim> * fe_,
							    dealii::Triangulation<dim> * triangulation_,
							    const dealii::MappingQ1<dim> * mapping_)
{
  dof_handler = dof_handler_ ;
  fe = fe_ ;
  triangulation = triangulation_ ;
  mapping = mapping_ ;

  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0;l<triangulation->n_levels();++l)
    dealii::deallog << ' ' << dof_handler->n_dofs(l);
  dealii::deallog << std::endl;
  
  mg_transfer.clear();
  mg_transfer.build_matrices(*dof_handler);

  const unsigned int n_levels = triangulation->n_levels();

  mg_smoother.clear();

  smoother_data.resize(0, n_levels-1);

  mg_sparsity.resize(0, n_levels-1);

  mg_matrix.resize(0, n_levels-1);
  mg_matrix.clear();
  mg_matrix_up.resize(0, n_levels-1);
  mg_matrix_up.clear();
  mg_matrix_down.resize(0, n_levels-1);
  mg_matrix_down.clear();

  for (unsigned int level=mg_sparsity.min_level();
       level<=mg_sparsity.max_level();++level)
    {
      dealii::DynamicSparsityPattern c_sparsity(dof_handler->n_dofs(level));
      dealii::MGTools::make_flux_sparsity_pattern(*dof_handler, c_sparsity, level);

      mg_sparsity[level].copy_from(c_sparsity);

      mg_matrix[level].reinit(mg_sparsity[level]);
      mg_matrix_up[level].reinit(mg_sparsity[level]);
      mg_matrix_down[level].reinit(mg_sparsity[level]);
    }

  dealii::deallog << "Assemble MG matrices" << std::endl;

  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_gradients |
    dealii::update_quadrature_points ;

  info_box.add_update_flags_all(update_flags);
  info_box.initialize(*fe, *mapping);

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler->block_info());

  dealii::MeshWorker::Assembler::MGMatrixSimple<dealii::SparseMatrix<number> > assembler;
  assembler.initialize(mg_matrix);
  assembler.initialize_interfaces(mg_matrix_up, mg_matrix_down);

  MatrixIntegratorMG<dim> matrix_integrator ;

  dealii::MeshWorker::integration_loop<dim, dim> (dof_handler->begin_mg(),dof_handler->end_mg(),
                                                  dof_info, info_box, matrix_integrator, assembler);

  coarse_matrix.reinit(0,0);
  coarse_matrix.copy_from (mg_matrix[mg_matrix.min_level()]);
  mg_coarse.initialize(coarse_matrix, 1.e-15);

  for (unsigned int l=smoother_data.min_level()+1;l<=smoother_data.max_level();++l)
    {
      smoother_data[l].block_list.reinit(triangulation->n_cells(l),dof_handler->n_dofs(l), fe->dofs_per_cell);
      dealii::DoFTools::make_cell_patches(smoother_data[l].block_list, *dof_handler, l);
      smoother_data[l].block_list.compress();
      smoother_data[l].relaxation = 1.0 ;
      smoother_data[l].inversion = dealii::PreconditionBlockBase<number>::svd;
      smoother_data[l].threshold = 1.e-12;
    }
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(1);
  mg_smoother.set_variable(false);
}


template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::vmult (dealii::Vector<number> & dst,
	    const dealii::Vector<number> & src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::Tvmult (dealii::Vector<number> & dst,
	     const dealii::Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::vmult_add (dealii::Vector<number> & dst,
							 const dealii::Vector<number> & src) const
{
  dealii::mg::Matrix<dealii::Vector<number> > mgmatrix(mg_matrix);
  dealii::mg::Matrix<dealii::Vector<number> > mgdown(mg_matrix_down);
  dealii::mg::Matrix<dealii::Vector<number> > mgup(mg_matrix_up);

  dealii::Multigrid<dealii::Vector<number> > mg(*dof_handler, mgmatrix,
                                                mg_coarse, mg_transfer,
                                                mg_smoother, mg_smoother);

  mg.set_edge_matrices(mgdown, mgup);
  mg.set_minlevel(mg_matrix.min_level());

  mg_transfer.copy_to_mg(*dof_handler,
			 mg.defect,
			 src);
  mg.cycle();
  mg_transfer.copy_from_mg_add(*dof_handler,
			       dst,
			       mg.solution);

}

template <int dim, int fe_degree, typename number>
void LaplacePreconditionerMG<dim,fe_degree,number>::Tvmult_add (dealii::Vector<number> & dst,
							  const dealii::Vector<number> &src) const
{
  vmult_add(dst, src);
}

template class LaplacePreconditionerMG<2,1,double>;
