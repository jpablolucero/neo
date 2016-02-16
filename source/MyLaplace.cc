#include <MyLaplace.h>

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mapping (),
  fe (1),
  dof_handler (triangulation),
  system_matrix ( triangulation,
		  mapping,
		  fe,
		  dof_handler) 
{}


template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
}

template <int dim>
void MyLaplace<dim>::assemble_system ()
{
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
    dealii::update_values;            
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping);

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  dealii::MeshWorker::Assembler::ResidualSimple<dealii::Vector<double> > rhs_assembler;
  dealii::AnyData data;
  data.add<dealii::Vector<double>*>(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);

  RHSIntegrator<dim> rhs_integrator;

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
						 dof_info, info_box,
						 rhs_integrator, rhs_assembler);

}

template <int dim>
void MyLaplace<dim>::solve (dealii::Vector<double> &solution)
{
  dealii::SolverControl           solver_control (1000, 1e-12);
  dealii::SolverCG<>              solver (solver_control);

  solver_control.log_history(true);
  solver.solve (system_matrix, solution, right_hand_side,
  		dealii::PreconditionIdentity());
}

template <int dim>
void MyLaplace<dim>::output_results () const
{
  std::string filename = "solution";

  filename += ".gnuplot";
  dealii::deallog << "Writing solution to <" << filename << ">" << std::endl;
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
    
  triangulation.refine_global (3);
  dealii::deallog << "Number of active cells:       "
	  << triangulation.n_active_cells()
	  << std::endl;

  setup_system ();
    
  dealii::deallog << "Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << std::endl;

  assemble_system ();
  solve (solution);
  output_results ();
}

template class MyLaplace<2>;
