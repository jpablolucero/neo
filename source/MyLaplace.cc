#include <MyLaplace.h>

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mapping (),
  fe (1),
  dof_handler (triangulation)
{}

template <int dim>
MyLaplace<dim>::~MyLaplace ()
{}

template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();
  system_matrix.reinit (&dof_handler,&fe,&mapping,
			triangulation.n_levels()-1) ;
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
  right_hand_side = 1.0/triangulation.n_active_cells() ;
}

template <int dim>
void MyLaplace<dim>::setup_multigrid ()
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

template <int dim>
void MyLaplace<dim>::solve ()
{
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
  solver.solve(system_matrix,solution,right_hand_side,preconditioner);
}

template <int dim>
void MyLaplace<dim>::output_results () const
{
  std::string filename = "solution.vtu";
  std::ofstream vtu_output (filename.c_str());
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");
  data_out.build_patches ();
  data_out.write_vtu(vtu_output);
}


template <int dim>
void MyLaplace<dim>::run ()
{
  for (unsigned int cycle=0; cycle<9-dim; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      if (cycle == 0)
	{  
	  dealii::GridGenerator::hyper_cube (triangulation,-1.,1.);
	  triangulation.refine_global (3-dim);
	}
      triangulation.refine_global (1);
      dealii::deallog << "Number of active cells: " << 
	triangulation.n_active_cells() << std::endl;
      setup_system ();
      dealii::deallog << "DoFHandler levels: ";
      for (unsigned int l=0;l<triangulation.n_levels();++l)
	dealii::deallog << ' ' << dof_handler.n_dofs(l);
      dealii::deallog << std::endl;
      setup_multigrid ();
      solve ();
      output_results ();
      dealii::deallog << std::endl;
    }
}

template class MyLaplace<2>;
template class MyLaplace<3>;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, true>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, false>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, true>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, false>,double >;
