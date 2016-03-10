#include <MyLaplace.h>
#include <GlobalTimer.h>

template <int dim,bool same_diagonal>
MyLaplace<dim,same_diagonal>::MyLaplace ()
  :
  mapping (),
  fe (1),
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
  system_matrix.reinit (&dof_handler,&fe,&mapping,
			triangulation.n_levels()-1) ;
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
  right_hand_side = 1.0/triangulation.n_active_cells();
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::setup_multigrid ()
{
  const unsigned int n_levels = triangulation.n_levels();
  mg_matrix.resize(0, n_levels-1);  
  for (unsigned int level=0;level<n_levels;++level)
    {
      mg_matrix[level].reinit(&dof_handler,&fe,&mapping,level);
      mg_matrix[level].build_matrix();
    }
  coarse_matrix.reinit(dof_handler.n_dofs(0),dof_handler.n_dofs(0));
  coarse_matrix.copy_from(mg_matrix[0]) ;
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve ()
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
  for (unsigned int cycle=0; cycle<7-dim; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      if (cycle == 0)
	{  
	  dealii::GridGenerator::hyper_cube (triangulation,-1.,1.);
	  triangulation.refine_global (3-dim);
	}
      global_timer.reset();
      global_timer.enter_subsection("refine_global");
      triangulation.refine_global (1);
      global_timer.leave_subsection();
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
      global_timer.enter_subsection("solve");
      solve ();
      global_timer.leave_subsection();
      global_timer.enter_subsection("output");
      output_results ();
      global_timer.leave_subsection();
      global_timer.print_summary();
      dealii::deallog << std::endl;
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
