#include <Simulator.h>

template <int dim,bool same_diagonal,unsigned int degree>
Simulator<dim,same_diagonal,degree>::Simulator (dealii::TimerOutput &timer_,
                                                MPI_Comm &mpi_communicator_,
                                                dealii::ConditionalOStream &pcout_)
  :
  n_levels(2),
  min_level(0),
  smoothing_steps(1),
  mpi_communicator(mpi_communicator_),
  mesh(mpi_communicator_),
  fe(degree),
  dofs(mesh,fe),
  rhs(fe,dofs,mpi_communicator_),
  pcout (pcout_),
  timer(timer_),
  residual(*this),
  inverse(*this),
  newton(residual,inverse)
{
  // initialize timer
  system_matrix.set_timer(timer);
#if PARALLEL_LA == 0
  pcout<< "Using deal.II (serial) linear algebra" << std::endl;
#elif PARALLEL_LA == 1
  pcout<< "Using PETSc parallel linear algebra" << std::endl;
#elif PARALLEL_LA == 2
  pcout<< "Using Trilinos parallel linear algebra" << std::endl;
#else
  pcout<< "Using deal.II parallel linear algebra" << std::endl;
#endif // PARALLEL_LA
#ifdef CG
  pcout<< "Using FE_Q elements" << std::endl;
#else
  pcout<< "Using FE_DGQ elements" << std::endl;
#endif //CG

#ifdef MATRIXFREE
  pcout << "Using deal.II's MatrixFree objects" << std::endl;
#else
  pcout << "Using MeshWorker-based matrix-free implementation" << std::endl;
#endif // MATRIXFREE
}

template <int dim,bool same_diagonal,unsigned int degree>
Simulator<dim,same_diagonal,degree>::~Simulator ()
{}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::setup_system ()
{
//   /*std::cout << "locally owned dofs on process "
//             << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
//             << std::endl;
//   for (unsigned int l=0; l<mesh.triangulation.n_global_levels(); ++l)
//     {
//       std::cout << "level: " << l << " n_elements(): "
//                 << dofs.dof_handler.locally_owned_mg_dofs(l).n_elements()
//                 << " index set: ";
//       dofs.dof_handler.locally_owned_mg_dofs(l).print(std::cout);
//     }
//   std::cout << "n_elements(): "
//             << dofs.dof_handler.locally_owned_dofs().n_elements()
//             <<std::endl;
//   dofs.dof_handler.locally_owned_dofs().print(dealii::deallog);*/
//   /*  std::cout << "locally relevant dofs on process "
//               << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
//     dofs.locally_relevant_dofs.print(std::cout);*/

  dofs.setup();
  
#ifdef MATRIXFREE
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),mpi_communicator,dealii::numbers::invalid_unsigned_int);
#else
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),mpi_communicator,mesh.triangulation.n_global_levels()-1);
#endif

#ifdef MATRIXFREE
#if PARALLEL_LA == 3
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(solution_tmp);
#elif PARALLEL_LA == 0
  solution.reinit (dofs.locally_owned_dofs.n_elements());
  solution_tmp.reinit (dofs.locally_owned_dofs.n_elements());
#else // PARALLEL_LA == 1,2
  AssertThrow(false, dealii::ExcNotImplemented());
#endif // PARALLEL_LA == 3

#else // MATRIXFREE OFF
#if PARALLEL_LA == 0
  solution.reinit (dofs.locally_owned_dofs.n_elements());
  solution_tmp.reinit (dofs.locally_owned_dofs.n_elements());
#elif PARALLEL_LA == 3
  solution.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (dofs.locally_owned_dofs, mpi_communicator);
#else
  solution.reinit (dofs.locally_owned_dofs, dofs.locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (dofs.locally_owned_dofs, mpi_communicator);
#endif // PARALLEL_LA == 0
#endif // MATRIXFREE
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::setup_multigrid ()
{
  const unsigned int n_global_levels = mesh.triangulation.n_global_levels();
  mg_matrix.resize(min_level, n_global_levels-1);
#ifndef MATRIXFREE
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
  mg_transfer.build_matrices(dofs.dof_handler);
  mg_solution.resize(min_level, n_global_levels-1);
  mg_transfer.copy_to_mg(dofs.dof_handler,mg_solution,solution);
  system_matrix.reinit (&(dofs.dof_handler),&(fe.mapping), &(dofs.constraints), mpi_communicator, mesh.triangulation.n_global_levels()-1, solution);
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].set_timer(timer);
      mg_matrix[level].reinit(&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),mpi_communicator,level,mg_solution[level]);
    }
#else // MATRIXFREE ON
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].set_timer(timer);
      mg_matrix[level].reinit(&(dofs.dof_handler),&(fe.mapping),&(dofs.constraints),mpi_communicator,level);
    }
#endif // MATRIXFREE
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::solve ()
{
  timer.enter_subsection("solve::mg_initialization");
#ifdef MG
  // Setup coarse solver
  dealii::SolverControl coarse_solver_control (dofs.dof_handler.n_dofs(min_level)*10, 1e-15, false, false);
  dealii::PreconditionIdentity id;
#if PARALLEL_LA < 3
  mg_matrix[min_level].build_coarse_matrix();
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[min_level].get_coarse_matrix();
  dealii::SolverGMRES<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::MGCoarseGridIterativeSolver<LA::MPI::Vector,
         dealii::SolverGMRES<LA::MPI::Vector>,
         LA::MPI::SparseMatrix,
         dealii::PreconditionIdentity>
         mg_coarse(coarse_solver,
                   coarse_matrix,
                   id);
#else // PARALLEL_LA == 3
  // TODO allow for Matrix-based solver for dealii MPI vectors
  dealii::SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::MGCoarseGridIterativeSolver<LA::MPI::Vector,
         dealii::SolverCG<LA::MPI::Vector>,
         SystemMatrixType,
         dealii::PreconditionIdentity>
         mg_coarse(coarse_solver,
                   mg_matrix[min_level],
                   id);
#endif

  // Setup Multigrid-Smoother
  typedef PSCPreconditioner<dim, LA::MPI::Vector, double, same_diagonal> Smoother;
  //typedef MFPSCPreconditioner<dim, LA::MPI::Vector, double> Smoother;
  Smoother::timer = &timer;
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());
  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      smoother_data[level].dof_handler = &(dofs.dof_handler);
      smoother_data[level].level = level;
      smoother_data[level].mapping = &(fe.mapping);
      smoother_data[level].relaxation = 0.7;
      smoother_data[level].mg_constrained_dofs = mg_constrained_dofs;
#ifndef MATRIXFREE
      smoother_data[level].solution = &mg_solution[level];
#endif // MATRIXFREE
      smoother_data[level].mpi_communicator = mpi_communicator;
      //      uncomment to use the dictionary
      // if(!same_diagonal)
      //  {
      //    smoother_data[level].use_dictionary = true;
      //    smoother_data[level].tol = 0.05;
      //  }
      smoother_data[level].patch_type = Smoother::AdditionalData::cell_patches;
    }
  dealii::MGSmootherPrecondition<SystemMatrixType,Smoother,LA::MPI::Vector> mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(smoothing_steps);

  // Setup Multigrid-Transfer
#ifdef MATRIXFREE
  dealii::MGTransferMF<dim,SystemMatrixType> mg_transfer {mg_matrix};
  mg_transfer.build(dofs.dof_handler);
#else // MATRIXFREE OFF
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer {};
#ifdef CG
  mg_transfer.initialize_constraints(dofs.constraints, mg_constrained_dofs);
#endif // CG
  mg_transfer.build_matrices(dofs.dof_handler);
#endif // MATRIXFREE

  // Setup (Multigrid-)Preconditioner
  dealii::mg::Matrix<LA::MPI::Vector>         mglevel_matrix;
  mglevel_matrix.initialize(mg_matrix);
  dealii::Multigrid<LA::MPI::Vector> mg(mglevel_matrix,
                                        mg_coarse,
                                        mg_transfer,
                                        mg_smoother,
                                        mg_smoother,
					min_level);
  // mg.set_debug(10);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
#ifdef MATRIXFREE
  dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferMF<dim,SystemMatrixType> >
  preconditioner(dofs.dof_handler, mg, mg_transfer);
#else
  dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferPrebuilt<LA::MPI::Vector> >
  preconditioner(dofs.dof_handler, mg, mg_transfer);
#endif // MATRIXFREE

#else // MG OFF
  dealii::PreconditionIdentity preconditioner;
#endif // MG

  // Setup Solver
  dealii::ReductionControl             solver_control (dofs.dof_handler.n_dofs(), 1.e-20, 1.e-10,true);
  dealii::SolverGMRES<LA::MPI::Vector> solver (solver_control);
  timer.leave_subsection();

  // Solve the system
  timer.enter_subsection("solve::solve");
  dofs.constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution_tmp,rhs.right_hand_side,preconditioner);
#ifdef CG
  dofs.constraints.distribute(solution_tmp);
#endif
  solution = solution_tmp;
  timer.leave_subsection();
}


template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim, same_diagonal, degree>::compute_error () const
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
                                            mpi_communicator));

  pcout << "L2 error: " << L2_error << std::endl;
}

template <int dim, bool same_diagonal, unsigned int degree>
void Simulator<dim, same_diagonal, degree>::output_results (const unsigned int cycle) const
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

  const unsigned int n_proc = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
  if (n_proc>1)
    {
      const int n_digits = dealii::Utilities::needed_digits(n_proc);
      std::ofstream output
      ((filename + "."
        + dealii::Utilities::int_to_string(mesh.triangulation.locally_owned_subdomain(),n_digits)
        + ".vtu").c_str());
      data_out.write_vtu (output);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
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


template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::run ()
{
  timer.reset();
  timer.enter_subsection("refine_global");
  pcout << "Refine global" << std::endl;
  mesh.triangulation.refine_global (n_levels-1);
  timer.leave_subsection();
  pcout << "Finite element: " << fe.fe.get_name() << std::endl;
  pcout << "Number of active cells: "
        << mesh.triangulation.n_global_active_cells()
        << std::endl;
  timer.enter_subsection("setup_system");
  pcout << "Setup system" << std::endl;
  setup_system ();
  pcout << "Assemble system" << std::endl;
  rhs.assemble(solution);
  timer.leave_subsection();
  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=min_level; l<mesh.triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dofs.dof_handler.n_dofs(l);
  dealii::deallog << std::endl;
#ifdef MG
  timer.enter_subsection("setup_multigrid");
  pcout << "Setup multigrid" << std::endl;
  setup_multigrid ();
  timer.leave_subsection();
#endif
  output_results(n_levels);
  timer.enter_subsection("solve");
  pcout << "Solve" << std::endl;
  solve ();
  timer.leave_subsection();
  timer.enter_subsection("output");
  pcout << "Output" << std::endl;
  compute_error();
  output_results(n_levels);
  timer.leave_subsection();
  timer.print_summary();
  pcout << std::endl;
  // workaround regarding issue #2533
  // GrowingVectorMemory does not destroy the vectors
  // after this instance goes out of scope.
  // Unfortunately, the mpi_communicators given to the
  // remaining vectors might be invalid the next time
  // a vector is requested. Therefore, clean up everything
  // before going out of scope.
  dealii::GrowingVectorMemory<LA::MPI::Vector>::release_unused_memory();
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::run_non_linear ()
{
  timer.reset();
  timer.enter_subsection("refine_global");
  pcout << "Refine global" << std::endl;
  mesh.triangulation.refine_global (n_levels-1);
  timer.leave_subsection();
  pcout << "Finite element: " << fe.fe.get_name() << std::endl;
  pcout << "Number of active cells: "
        << mesh.triangulation.n_global_active_cells()
        << std::endl;
  timer.enter_subsection("setup_system");
  pcout << "Setup system" << std::endl;
  setup_system ();
  timer.leave_subsection();
  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0; l<mesh.triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dofs.dof_handler.n_dofs(l);
  dealii::deallog << std::endl;
  auto sol = solution_tmp ;
  for (auto &elem : sol) elem = 1. ;
  dealii::AnyData solution_data;
  solution_data.add(&sol, "solution");
  dealii::AnyData data;
  newton.control.set_reduction(1.E-10);
  timer.enter_subsection("solve");
  pcout << "Solve" << std::endl;
  newton(solution_data, data);
  solution = *(solution_data.try_read_ptr<LA::MPI::Vector>("solution"));
  timer.leave_subsection();
  timer.enter_subsection("output");
  output_results(n_levels);
  timer.leave_subsection();
  timer.print_summary();
  pcout << std::endl;
  // workaround regarding issue #2533
  // GrowingVectorMemory does not destroy the vectors
  // after this instance goes out of scope.
  // Unfortunately, the mpi_communicators given to the
  // remaining vectors might be invalid the next time
  // a vector is requested. Therefore, clean up everything
  // before going out of scope.
  dealii::GrowingVectorMemory<LA::MPI::Vector>::release_unused_memory();
}

#ifndef HEADER_IMPLEMENTATION
#include "Simulator.inst"
#endif 
