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
#ifdef MG
  preconditioner(mesh,dofs,fe),
#endif // MG
  residual(*this),
  inverse(*this),
  newton(residual,inverse)
{}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::~Simulator ()
{}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::setup_system ()
{
  dofs.setup();
  
#ifdef MATRIXFREE
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),*mpi_communicator,dealii::numbers::invalid_unsigned_int);
#else
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),mesh.triangulation.n_global_levels()-1);
#endif

#ifdef MATRIXFREE
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(solution_tmp);
#else // MATRIXFREE OFF
  solution.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, *mpi_communicator);
  solution_tmp.reinit (dofs.locally_owned_dofs, *mpi_communicator);
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping), &(dofs.constraints), mesh.triangulation.n_global_levels()-1, solution);
#endif // MATRIXFREE
}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::solve ()
{
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping), &(dofs.constraints), mesh.triangulation.n_global_levels()-1, solution);
  
#ifdef MG
  preconditioner.setup(solution,min_level);
#endif // MG
  
  // Setup Solver
  dealii::ReductionControl                                 solver_control (dofs.dof_handler.n_dofs(), 1.e-20, 1.e-10,true);
  typename dealii::SolverGMRES<VectorType>::AdditionalData data(100,true);
  dealii::SolverGMRES<VectorType>                          solver (solver_control,data);
      
  // Solve the system
  timer->enter_subsection("solve::solve");
  dofs.constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution_tmp,rhs.right_hand_side,preconditioner);
  
#ifdef CG
  dofs.constraints.distribute(solution_tmp);
#endif
  solution = solution_tmp;
  timer->leave_subsection();
}


template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::compute_error () const
{
  dealii::QGauss<dim> quadrature (degree+2);
  dealii::Vector<double> local_errors;

  dealii::VectorTools::integrate_difference (fe.mapping, dofs.dof_handler,
                                             solution,
                                             dofs.reference_function,
                                             local_errors, quadrature,
                                             dealii::VectorTools::L2_norm);
  const double L2_error_local = local_errors.l2_norm();
  const double L2_error
    = std::sqrt(dealii::Utilities::MPI::sum(L2_error_local * L2_error_local,
                                            *mpi_communicator));

  *pcout << "L2 error: " << L2_error << std::endl;
}

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim,unsigned int degree>
void Simulator<SystemMatrixType,VectorType,Preconditioner,dim,degree>::output_results (const unsigned int cycle) const
{
  std::string filename = "solution-"+dealii::Utilities::int_to_string(cycle,2);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler (dofs.dof_handler);
  data_out.add_data_vector (solution, "u");
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
  rhs.assemble(solution);
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
  auto sol = solution_tmp ;
  for (auto &elem : sol) elem = 600. ;
  dealii::AnyData solution_data;
  solution_data.add(&sol, "solution");
  dealii::AnyData data;
  newton.control.set_reduction(1.E-8);
  timer->enter_subsection("solve");
  *pcout << "Solve" << std::endl;
  newton(solution_data, data);
  solution = *(solution_data.try_read_ptr<VectorType>("solution"));
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
