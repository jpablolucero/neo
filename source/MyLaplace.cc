#include <MyLaplace.h>

template <int dim,bool same_diagonal>
MyLaplace<dim,same_diagonal>::MyLaplace ()
  :
  mapping (),
  fe {dealii::FE_DGQ<dim>{2}, 1},
  dof_handler (triangulation)
{}

template <int dim,bool same_diagonal>
MyLaplace<dim,same_diagonal>::~MyLaplace ()
{}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();
  system_matrix.reinit (&dof_handler, &mapping, triangulation.n_levels() - 1);
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::setup_multigrid ()
{
  const unsigned int n_levels = triangulation.n_levels();
  mg_matrix.resize(0, n_levels-1);  
  for (unsigned int level=0;level<n_levels;++level)
    {
      mg_matrix[level].reinit(&dof_handler,&mapping,level);
      mg_matrix[level].build_matrix();
    }
  coarse_matrix.reinit(dof_handler.n_dofs(0),dof_handler.n_dofs(0));
  coarse_matrix.copy_from(mg_matrix[0]) ;
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::assemble_rhs()
{
  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
    dealii::update_values |
    dealii::update_quadrature_points ;
  
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box_rhs;
  info_box_rhs.add_update_flags_all(update_flags);
  info_box_rhs.initialize(fe, mapping, &dof_handler.block_info());
  dealii::MeshWorker::DoFInfo<dim> dof_info_rhs(dof_handler.block_info());
  
  dealii::ConstraintMatrix cmatrix_dummy;
  dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > rhs_assembler;
  dealii::AnyData data;
  data.add(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);
  rhs_assembler.initialize(cmatrix_dummy);

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
						 dof_info_rhs, info_box_rhs,
						 rhs_integrator, rhs_assembler);
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve ()
{
  if(use_psc)
  {
    solve_psc();
  } else
  {
    solve_blockjacobi();
  }
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve_psc ()
{
  global_timer.enter_subsection("solve::mg_initialization");
  dealii::MGTransferPrebuilt<dealii::Vector<double> > mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  dealii::MGCoarseGridSVD<double, 
			  dealii::Vector<double> >    mg_coarse;
  mg_coarse.initialize(coarse_matrix, 1.e-15);

  // Smoother setup
  typedef PSCPreconditioner<dim, double> Smoother;

  dealii::MGLevelObject<dealii::FullMatrix<double> > local_level_inverse;
  local_level_inverse.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<DGDDHandler<dim, double> > level_ddh;
  level_ddh.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());

  for(unsigned int level = mg_matrix.min_level();
      level <= mg_matrix.max_level();
      ++level)
  {
    // init ddhandler
    level_ddh[level].initialize(dof_handler, level);

    // init local inverse
    // TODO this assumes that mg_matrix stores the right thing in the first
    // indices....
    const unsigned int n = dof_handler.get_fe().n_dofs_per_cell();
    dealii::FullMatrix<double> local_matrix(n, n);
    for(unsigned int i = 0; i < n; ++i)
    {
      for(unsigned int j = 0; j < n; ++j)
      {
        local_matrix(i, j) = mg_matrix[level](i, j);
      }
    }
    local_level_inverse[level].reinit(n, n);
    local_level_inverse[level].invert(local_matrix);

    // setup smoother data
    smoother_data[level].ddh = &(level_ddh[level]);
    smoother_data[level].local_inverses =
      std::vector<const dealii::FullMatrix<double>* >(
          level_ddh[level].size(),
          &(local_level_inverse[level]));
    smoother_data[level].weight = 1.0;
  }
  // /SmootherSetup
  
  dealii::MGSmootherPrecondition<
    SystemMatrixType,
    Smoother,
    dealii::Vector<double> > mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(6);
  dealii::mg::Matrix<dealii::Vector<double> >         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::Multigrid<dealii::Vector<double> > mg(dof_handler, mgmatrix,
						mg_coarse, mg_transfer,
						mg_smoother, mg_smoother);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, dealii::Vector<double>,
			 dealii::MGTransferPrebuilt<dealii::Vector<double> > >
  preconditioner(dof_handler, mg, mg_transfer);  
  dealii::ReductionControl        solver_control (1000, 1.E-20, 1.E-10);
  dealii::SolverCG<>              solver (solver_control);
  global_timer.leave_subsection();
  global_timer.enter_subsection("solve::solve");
  solver.solve(system_matrix,solution,right_hand_side,preconditioner);
  global_timer.leave_subsection();
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve_blockjacobi ()
{
  global_timer.enter_subsection("solve::mg_initialization");
  dealii::MGTransferPrebuilt<dealii::Vector<double> > mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  dealii::MGCoarseGridSVD<double, 
			  dealii::Vector<double> >    mg_coarse;
  mg_coarse.initialize(coarse_matrix, 1.e-15);
  typename dealii::PreconditionBlockJacobi<SystemMatrixType >::AdditionalData 
    smoother_data(dof_handler.block_info().local().block_size(0),
                  1.0, true, same_diagonal);
  dealii::MGSmootherPrecondition<SystemMatrixType,
				 dealii::PreconditionBlockJacobi<SystemMatrixType >,
				 dealii::Vector<double> > mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(6);
  dealii::mg::Matrix<dealii::Vector<double> >         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::Multigrid<dealii::Vector<double> > mg(dof_handler, mgmatrix,
						mg_coarse, mg_transfer,
						mg_smoother, mg_smoother);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, dealii::Vector<double>,
			 dealii::MGTransferPrebuilt<dealii::Vector<double> > >
  preconditioner(dof_handler, mg, mg_transfer);  
  dealii::ReductionControl        solver_control (1000, 1.E-20, 1.E-10);
  dealii::SolverCG<>              solver (solver_control);
  global_timer.leave_subsection();
  global_timer.enter_subsection("solve::solve");
  solver.solve(system_matrix,solution,right_hand_side,preconditioner);
  global_timer.leave_subsection();
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::output_results () const
{
  std::string filename = "solution.vtu";
  std::ofstream vtu_output (filename.c_str());
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");
  data_out.build_patches ();
  data_out.write_vtu(vtu_output);
}


template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::run ()
{
  for (unsigned int cycle=0; cycle<9-dim; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      if (cycle == 0)
	{  
	  dealii::GridGenerator::hyper_cube (triangulation,0.,1.);
	  triangulation.refine_global (3-dim);
	}
      global_timer.reset();
      global_timer.enter_subsection("refine_global");
      triangulation.refine_global (1);
      global_timer.leave_subsection();
      dealii::deallog << "Finite element: " << fe.get_name() << std::endl;
      dealii::deallog << "Number of active cells: " << 
	triangulation.n_active_cells() << std::endl;
      global_timer.enter_subsection("setup_system");
      setup_system ();
      global_timer.leave_subsection();
      dealii::deallog << "DoFHandler levels: ";
      for (unsigned int l=0;l<triangulation.n_levels();++l)
	dealii::deallog << ' ' << dof_handler.n_dofs(l);
      dealii::deallog << std::endl;
      global_timer.enter_subsection("setup_multigrid");
      setup_multigrid ();
      global_timer.leave_subsection();
      global_timer.enter_subsection("assemble_rhs");
      assemble_rhs ();
      global_timer.leave_subsection();
      global_timer.enter_subsection("solve");
      solve ();
      global_timer.leave_subsection();
      global_timer.enter_subsection("output");
      output_results ();
      global_timer.leave_subsection();
      global_timer.print_summary();
      dealii::deallog << std::endl;
      system_matrix.clear();
    }
}

template class MyLaplace<2,true>;
template class MyLaplace<3,true>;
template class MyLaplace<2,false>;
template class MyLaplace<3,false>;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, true>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, false>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, true>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, false>,double >;
