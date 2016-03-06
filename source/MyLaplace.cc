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
  system_matrix.reinit (&dof_handler,&fe,&triangulation,&mapping) ;
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
  right_hand_side = 1.0/triangulation.n_active_cells() ;
}

template <int dim>
void MyLaplace<dim>::setup_multigrid ()
{
  mg_transfer.build_matrices(dof_handler);
  const unsigned int n_levels = triangulation.n_levels();
  mg_matrix_laplace.resize(0, n_levels-1);
  mg_matrix_preconditioner.resize(0, n_levels-1);  
  for (unsigned int level=0;level<n_levels;++level)
    {
      mg_matrix_laplace[level].reinit(&dof_handler,&fe,&triangulation,&mapping,level,true);
      mg_matrix_preconditioner[level].reinit(&dof_handler,&fe,&triangulation,&mapping,level,true);
      mg_matrix_preconditioner[level].build_matrix();
    }
  coarse_matrix.reinit(dof_handler.n_dofs(0),dof_handler.n_dofs(0));
  coarse_matrix.copy_from(mg_matrix_preconditioner[0]) ;
  mg_coarse.initialize(coarse_matrix, 1.e-15);
  typename dealii::PreconditionBlockJacobi<SystemMatrixType >::AdditionalData 
    smoother_data(dof_handler.block_info().local().block_size(0),1.0,true,true);
  mg_smoother.initialize(mg_matrix_preconditioner, smoother_data);
  mgmatrixlaplace.initialize(mg_matrix_laplace);
}

template <int dim>
void MyLaplace<dim>::solve ()
{
  dealii::Multigrid<dealii::Vector<typename SystemMatrixType::value_type> > mglaplace(dof_handler, mgmatrixlaplace,
										     mg_coarse, mg_transfer,
										     mg_smoother, mg_smoother);
  mglaplace.set_minlevel(mg_matrix_laplace.min_level());
  mglaplace.set_maxlevel(mg_matrix_laplace.max_level());
  dealii::PreconditionMG<dim, dealii::Vector<typename SystemMatrixType::value_type>,
			 dealii::MGTransferPrebuilt<dealii::Vector<typename SystemMatrixType::value_type> > >
  preconditioner(dof_handler, mglaplace, mg_transfer);
  
  dealii::ReductionControl        solver_control (1000, 1.E-20, 1.E-10);
  dealii::SolverCG<>              solver (solver_control);
  solver_control.log_history(true);
  solver.solve(system_matrix,solution,right_hand_side,preconditioner);
}

template <int dim>
void MyLaplace<dim>::output_results () const
{
  std::string filename = "solution.gnuplot";
  std::ofstream gnuplot_output (filename.c_str());
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");
  data_out.build_patches ();
  data_out.write_gnuplot(gnuplot_output);
}


template <int dim>
void MyLaplace<dim>::run ()
{
  dealii::GridGenerator::hyper_cube (triangulation,-1.,1.);
  triangulation.refine_global (5);
  dealii::deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  setup_system ();
  setup_multigrid ();
  solve ();
  output_results ();
}

template class MyLaplace<2>;
template class MyLaplace<3>;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<2,1>,double >;
template class dealii::PreconditionBlockJacobi<LaplaceOperator<3,1>,double >;
