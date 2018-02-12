#include <Simulator.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;
extern std::unique_ptr<dealii::ConditionalOStream> pcout ;

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::Simulator ()
  :
  n_levels(2),
  min_level(0),
  smoothing_steps(1),
  fe(degree),
  dofs(mesh,fe),
  rhs(fe,dofs),
  residual(*this),
  inverse(*this),
  newton(residual,inverse)
{}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::~Simulator ()
{}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
template <typename P>
typename std::enable_if<std::is_same<P,dealii::PreconditionIdentity>::value >::type
Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::setup_system ()
{
  dofs.setup();
  ghosted_solution.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
  solution.reinit (dofs.locally_owned_dofs, *mpi_communicator);
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping), &(dofs.constraints), mesh.triangulation.n_global_levels()-1,
			ghosted_solution);
}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
template <typename P>
typename std::enable_if<!std::is_same<P,dealii::PreconditionIdentity>::value >::type
Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::setup_system ()
{
  dofs.setup();
  ghosted_solution.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
  solution.reinit (dofs.locally_owned_dofs, *mpi_communicator);
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping), &(dofs.constraints), mesh.triangulation.n_global_levels()-1,
			ghosted_solution);
  pdata.mesh = &mesh;
  pdata.dofs = &dofs;
  pdata.fe = &fe;
  pdata.solution = &ghosted_solution;
}


template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::solve ()
{
  preconditioner.initialize(system_matrix,pdata);
 
  // Setup Solver
  dealii::ReductionControl                                 solver_control (dofs.dof_handler.n_dofs(), 1.e-20, 1.E-10,true);
  typename dealii::SolverGMRES<VectorType>::AdditionalData data(100,true);
  dealii::SolverGMRES<VectorType>                          solver (solver_control,data);

#ifdef CG
  VectorType constraint_entries(solution);
  dofs.constraints.distribute(constraint_entries);
  VectorType tmp(solution);
  system_matrix.vmult(tmp, constraint_entries);
  rhs.right_hand_side.add(-1., tmp);
#endif

  // Solve the system
  timer->enter_subsection("solve::solve");
  dofs.constraints.set_zero(solution);
  solver.solve(system_matrix,solution,rhs.right_hand_side,preconditioner);
  
#ifdef CG
  solution += constraint_entries;
  dofs.constraints.distribute(solution);
#endif

  ghosted_solution = solution;
  ghosted_solution.update_ghost_values();
  timer->leave_subsection();
}


template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::compute_error () const
{
  dealii::QGauss<dim> quadrature (degree+2);
  dealii::Vector<double> local_errors;

  unsigned int base_component_start = 0;
  const unsigned int n_base_elements = fe.fe.n_base_elements();
  for (unsigned int i = 0; i < n_base_elements; ++i)
    {
      const dealii::FiniteElement<dim>& base = fe.fe.base_element(i);
      const unsigned int base_n_components = base.n_components();
      const std::pair<unsigned int, unsigned int> selected (base_component_start,
							    base_component_start + base_n_components);
      dealii::ComponentSelectFunction<dim>
        block_mask (selected, fe.fe.n_components());

      dealii::VectorTools::integrate_difference (fe.mapping, dofs.dof_handler,
						 ghosted_solution,
						 dofs.reference_function,
						 local_errors, quadrature,
						 dealii::VectorTools::L2_norm,
						 &block_mask);
      const double L2_error_local = local_errors.l2_norm();
      const double L2_error
	= std::sqrt(dealii::Utilities::MPI::sum(L2_error_local * L2_error_local,
						*mpi_communicator));

      if (n_base_elements > 1)
	*pcout << "Block(" << i << ") ";
      *pcout << "L2 error: " << L2_error << std::endl;

      base_component_start += base_n_components;
    }
}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::output_results (const unsigned int cycle) const
{
  std::string filename = "solution-"+dealii::Utilities::int_to_string(cycle,2);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler (dofs.dof_handler);

  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> output_data_types (fe.fe.n_components());
  unsigned int comp = 0;
  for (unsigned int i = 0; i < fe.fe.n_base_elements(); ++i)
    {
      const dealii::FiniteElement<dim>& base = fe.fe.base_element(i);
      dealii::DataComponentInterpretation::DataComponentInterpretation inter =
	dealii::DataComponentInterpretation::component_is_scalar;
      if (base.n_components() == dim)
	inter = dealii::DataComponentInterpretation::component_is_part_of_vector;
      for (unsigned int j = 0; j < fe.fe.element_multiplicity(i); ++j)
	for (unsigned int k = 0; k < base.n_components(); ++k)
	  output_data_types[comp++] = inter;
    }
  data_out.add_data_vector (ghosted_solution, "u",
			    dealii::DataOut_DoFData<dealii::DoFHandler<dim>, dim, dim>::type_dof_data,
			    output_data_types);

  dealii::Vector<float> subdomain (mesh.triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = mesh.triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");

  data_out.build_patches (fe.fe.degree);

  const unsigned int n_proc = dealii::Utilities::MPI::n_mpi_processes(*mpi_communicator);
  if (n_proc>1)
    {
      const int n_digits = dealii::Utilities::needed_digits(n_proc);
      std::ofstream output
	((filename + "."
	  + dealii::Utilities::int_to_string(mesh.triangulation.locally_owned_subdomain(),n_digits)
	  + ".vtu").c_str());
      data_out.write_vtu (output);

      if (dealii::Utilities::MPI::this_mpi_process(*mpi_communicator) == 0)
        {
          std::vector<std::string> filenames;
          for (unsigned int i=0; i<n_proc; i++)
            filenames.push_back (filename + "."
                                 + dealii::Utilities::int_to_string (i,n_digits) + ".vtu");
          std::ofstream master_output ((filename + ".pvtu").c_str());
          data_out.write_pvtu_record (master_output, filenames);
        }
    }
  else
    {
      std::ofstream output ((filename + ".vtk").c_str());
      data_out.write_vtk (output);
    }
}


template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::run ()
{
  timer->reset();
  timer->enter_subsection("refine_global");
  *pcout << "Refine global" << std::endl;
  mesh.triangulation.refine_global (n_levels-1);
  timer->leave_subsection();

  *pcout << "Finite element: " << fe.fe.get_name() << std::endl;
  *pcout << "Number of active cells: "
        << mesh.triangulation.n_global_active_cells()
        << std::endl;

  timer->enter_subsection("setup_system");
  *pcout << "Setup system" << std::endl;
  setup_system ();
  *pcout << "Assemble system" << std::endl;
  rhs.assemble(ghosted_solution);
  timer->leave_subsection();

  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=min_level; l<mesh.triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dofs.dof_handler.n_dofs(l);
  dealii::deallog << std::endl;

  timer->enter_subsection("solve");
  *pcout << "Solve" << std::endl;
  solve ();
  timer->leave_subsection();

  timer->enter_subsection("output");
  *pcout << "Output" << std::endl;
  compute_error();
  output_results(n_levels);
  timer->leave_subsection();

  timer->print_summary();
  *pcout << std::endl;
  // workaround regarding issue #2533
  // GrowingVectorMemory does not destroy the vectors
  // after this instance goes out of scope.
  // Unfortunately, the mpi_communicators given to the
  // remaining vectors might be invalid the next time
  // a vector is requested. Therefore, clean up everything
  // before going out of scope.
  dealii::GrowingVectorMemory<VectorType>::release_unused_memory();
}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::run_non_linear ()
{
  timer->reset();
  timer->enter_subsection("refine_global");
  *pcout << "Refine global" << std::endl;
  mesh.triangulation.refine_global (n_levels-1);
  timer->leave_subsection();

  *pcout << "Finite element: " << fe.fe.get_name() << std::endl;
  *pcout << "Number of active cells: "
        << mesh.triangulation.n_global_active_cells()
        << std::endl;

  timer->enter_subsection("setup_system");
  *pcout << "Setup system" << std::endl;
  setup_system ();
  timer->leave_subsection();

  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0; l<mesh.triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dofs.dof_handler.n_dofs(l);
  dealii::deallog << std::endl;
  auto sol = solution ;
  for (auto &elem : sol) elem = 600. ;
  dealii::AnyData solution_data;
  solution_data.add(&sol, "solution");
  dealii::AnyData data;
  newton.control.set_reduction(1.E-8);

  timer->enter_subsection("solve");
  *pcout << "Solve" << std::endl;
  newton(solution_data, data);
  ghosted_solution = *(solution_data.try_read_ptr<VectorType>("solution"));
  ghosted_solution.update_ghost_values();
  timer->leave_subsection();

  timer->enter_subsection("output");
  output_results(n_levels);
  timer->leave_subsection();

  timer->print_summary();
  *pcout << std::endl;
  // workaround regarding issue #2533
  // GrowingVectorMemory does not destroy the vectors
  // after this instance goes out of scope.
  // Unfortunately, the mpi_communicators given to the
  // remaining vectors might be invalid the next time
  // a vector is requested. Therefore, clean up everything
  // before going out of scope.
  dealii::GrowingVectorMemory<VectorType>::release_unused_memory();
}
