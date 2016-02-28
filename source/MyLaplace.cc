#include <MyLaplace.h>

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mapping (),
  fe (1),
  dof_handler (triangulation)
{}

template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();
  system_matrix.reinit (&dof_handler,&fe,&triangulation,&mapping) ;
  system_mg_matrix.reinit (&dof_handler,&fe,&triangulation,&mapping) ;
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
  right_hand_side = 1.0/triangulation.n_active_cells() ;
}

template <int dim>
void MyLaplace<dim>::solve ()
{
  dealii::ReductionControl           solver_control (1000, 1.E-20, 1.E-10);
  dealii::SolverCG<>              solver (solver_control);
  solver_control.log_history(true);
  solver.solve (system_matrix, solution, right_hand_side,system_mg_matrix);
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
  triangulation.refine_global (10);
  dealii::deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  setup_system ();
  solve ();
  output_results ();
}

template class MyLaplace<2>;
