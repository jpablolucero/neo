#include <Simulator.h>

template <int dim,bool same_diagonal,unsigned int degree>
Simulator<dim,same_diagonal,degree>::Simulator (dealii::TimerOutput &timer_,
                                                MPI_Comm &mpi_communicator_,
                                                dealii::ConditionalOStream &pcout_)
  :
  n_levels(2),
  smoothing_steps(1),
  mpi_communicator(mpi_communicator_),
  triangulation(mpi_communicator,dealii::Triangulation<dim>::
                limit_level_difference_at_vertices,
                dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  mapping (),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1),
#else
  fe(dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree),1),1),
#endif
  reference_function(fe.n_components()),
  dof_handler (triangulation),
  pcout (pcout_),
  timer(timer_),
  residual(*this),
  inverse(*this),
  newton(residual,inverse)
{
  // initialize timer
  system_matrix.set_timer(timer);
#if PARALLEL_LA == 0
  pcout<< "Using deal.II parallel linear algebra" << std::endl;
#elif PARALLEL_LA == 1
  pcout<< "Using PETSc parallel linear algebra" << std::endl;
#else
  pcout<< "Using Trilinos parallel linear algebra" << std::endl;
#endif
#ifdef CG
  pcout<< "Using FE_Q elements" << std::endl;
#else
  pcout<< "Using FE_DGQ elements" << std::endl;
#endif

  dealii::GridGenerator::hyper_cube (triangulation,0.,1., true);

#ifdef PERIODIC
  //add periodicity
  typedef typename dealii::Triangulation<dim>::cell_iterator CellIteratorTria;
  std::vector<dealii::GridTools::PeriodicFacePair<CellIteratorTria> > periodic_faces;
  const unsigned int b_id1 = 2;
  const unsigned int b_id2 = 3;
  const unsigned int direction = 1;

  dealii::GridTools::collect_periodic_faces (triangulation, b_id1, b_id2,
                                             direction, periodic_faces, dealii::Tensor<1,dim>());
  triangulation.add_periodicity(periodic_faces);
#endif
}

template <int dim,bool same_diagonal,unsigned int degree>
Simulator<dim,same_diagonal,degree>::~Simulator ()
{}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();

  locally_owned_dofs = dof_handler.locally_owned_dofs();

  /*std::cout << "locally owned dofs on process "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            << std::endl;
  for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
    {
      std::cout << "level: " << l << " n_elements(): "
                << dof_handler.locally_owned_mg_dofs(l).n_elements()
                << " index set: ";
      dof_handler.locally_owned_mg_dofs(l).print(std::cout);
    }
  std::cout << "n_elements(): "
            << dof_handler.locally_owned_dofs().n_elements()
            <<std::endl;
  dof_handler.locally_owned_dofs().print(dealii::deallog);*/

  dealii::DoFTools::extract_locally_relevant_dofs
  (dof_handler, locally_relevant_dofs);
  /*std::cout << "locally relevant dofs on process "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
  locally_relevant_dofs.print(std::cout);*/

  //constraints
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
#ifdef CG
#ifdef PERIODIC
  //Periodic boundary conditions
  std::vector<dealii::GridTools::PeriodicFacePair
  <typename dealii::DoFHandler<dim>::cell_iterator> >
  periodic_faces;

  const unsigned int b_id1 = 2*dim-2;
  const unsigned int b_id2 = 2*dim-1;
  const unsigned int direction = dim-1;

  dealii::GridTools::collect_periodic_faces (dof_handler,
                                             b_id1, b_id2, direction,
                                             periodic_faces);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<dim> >
  (periodic_faces, constraints);
  for (unsigned int i=0; i<2*dim-2; ++i)
    dealii::VectorTools::interpolate_boundary_values(dof_handler, i,
                                                     reference_function,
                                                     constraints);
#else
  for (unsigned int i=0; i<2*dim; ++i)
    dealii::VectorTools::interpolate_boundary_values(dof_handler, i,
                                                     reference_function,
                                                     constraints);
#endif

  dealii::DoFTools::make_hanging_node_constraints
  (dof_handler, constraints);

#endif
  constraints.close();

#if PARALLEL_LA == 0
  solution.reinit (locally_owned_dofs.n_elements());
  solution_tmp.reinit (locally_owned_dofs.n_elements());
  right_hand_side.reinit (locally_owned_dofs.n_elements());
#else
  solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (locally_owned_dofs, mpi_communicator);
  right_hand_side.reinit (locally_owned_dofs, mpi_communicator);
#endif
}

template <int dim, bool same_diagonal, unsigned int degree>
void Simulator<dim, same_diagonal, degree>::assemble_system ()
{
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
                                       n_gauss_points,
                                       n_gauss_points);
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
                                     dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);

  dealii::AnyData src_data ;
  src_data.add<const LA::MPI::Vector *>(&solution,"Newton iterate");

  info_box.initialize(fe,mapping,src_data,LA::MPI::Vector {},&(dof_handler.block_info()));

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
//  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
  dealii::AnyData data;
  data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);
#ifdef CG
  rhs_assembler.initialize(constraints);
#endif

  RHSIntegrator<dim> rhs_integrator(fe.n_components());

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
                                                 dof_info, info_box,
                                                 rhs_integrator, rhs_assembler);
  right_hand_side.compress(dealii::VectorOperation::add);
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::setup_multigrid ()
{
  const unsigned int n_global_levels = triangulation.n_global_levels();
  mg_matrix.resize(0, n_global_levels-1);
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  mg_solution.resize(0, n_global_levels-1);
  mg_transfer.copy_to_mg(dof_handler,mg_solution,solution);
  system_matrix.reinit (&dof_handler,&mapping, &constraints, mpi_communicator, triangulation.n_global_levels()-1, solution);
  for (unsigned int level=0; level<n_global_levels; ++level)
    {
      mg_matrix[level].set_timer(timer);
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints, mpi_communicator, level, mg_solution[level]);
    }
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::solve ()
{
  timer.enter_subsection("solve::mg_initialization");
#ifdef MG
  mg_matrix[0].build_coarse_matrix();
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[0].get_coarse_matrix();

  dealii::SolverControl coarse_solver_control (dof_handler.n_dofs(0)*10, 1e-10, false, false);
  dealii::SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::PreconditionIdentity id;
  dealii::MGCoarseGridLACIteration<dealii::SolverCG<LA::MPI::Vector>,LA::MPI::Vector> mg_coarse(coarse_solver,
      coarse_matrix,
      id);

  // Smoother setup
  typedef PSCPreconditioner<dim, LA::MPI::Vector, double, same_diagonal> Smoother;
  //typedef MFPSCPreconditioner<dim, LA::MPI::Vector, double> Smoother;
  Smoother::timer = &timer;

  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());

  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      // setup smoother data
      smoother_data[level].dof_handler = &dof_handler;
      smoother_data[level].level = level;
      smoother_data[level].mapping = &mapping;
      smoother_data[level].relaxation = .7;
      smoother_data[level].mg_constrained_dofs = mg_constrained_dofs;
      smoother_data[level].solution = &mg_solution[level];
      smoother_data[level].mpi_communicator = mpi_communicator;
      //      uncomment to use the dictionary
      // if(!same_diagonal)
      //  {
      //    smoother_data[level].use_dictionary = true;
      //    smoother_data[level].tol = 0.05;
      //  }
      smoother_data[level].patch_type = Smoother::AdditionalData::cell_patches;
    }

  // SmootherSetup
  dealii::MGSmootherPrecondition<SystemMatrixType, Smoother, LA::MPI::Vector> mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(smoothing_steps);
  dealii::mg::Matrix<LA::MPI::Vector>         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
#ifdef CG
  mg_transfer.initialize_constraints(constraints, mg_constrained_dofs);
#endif
  mg_transfer.build_matrices(dof_handler);
  dealii::Multigrid<LA::MPI::Vector> mg(dof_handler, mgmatrix,
                                        mg_coarse, mg_transfer,
                                        mg_smoother, mg_smoother);
//  mg.set_debug(10);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, LA::MPI::Vector,
         dealii::MGTransferPrebuilt<LA::MPI::Vector> >
         preconditioner(dof_handler, mg, mg_transfer);
#else
  dealii::PreconditionIdentity preconditioner;
#endif

  dealii::ReductionControl          solver_control (dof_handler.n_dofs(), 1.e-20, 1.e-10,true);
  dealii::SolverCG<LA::MPI::Vector> solver (solver_control);

  timer.leave_subsection();
  timer.enter_subsection("solve::solve");
  constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution_tmp,right_hand_side,preconditioner);
#ifdef CG
  constraints.distribute(solution_tmp);
#endif
  solution = solution_tmp;
  timer.leave_subsection();
}


template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim, same_diagonal, degree>::compute_error () const
{
  dealii::QGauss<dim> quadrature (degree+2);
  dealii::Vector<double> local_errors;

  dealii::VectorTools::integrate_difference (mapping, dof_handler,
                                             solution,
                                             reference_function,
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
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");
  dealii::Vector<float> subdomain (triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector (subdomain, "subdomain");

  data_out.build_patches (fe.degree);

  const unsigned int n_proc = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
  if (n_proc>1)
    {
      const int n_digits = dealii::Utilities::needed_digits(n_proc);
      std::ofstream output
      ((filename + "."
        + dealii::Utilities::int_to_string(triangulation.locally_owned_subdomain(),n_digits)
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
  triangulation.refine_global (n_levels-1);
  timer.leave_subsection();
  pcout << "Finite element: " << fe.get_name() << std::endl;
  pcout << "Number of active cells: "
        << triangulation.n_global_active_cells()
        << std::endl;
  timer.enter_subsection("setup_system");
  pcout << "Setup system" << std::endl;
  setup_system ();
  pcout << "Assemble system" << std::endl;
  assemble_system();
  timer.leave_subsection();
  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dof_handler.n_dofs(l);
  dealii::deallog << std::endl;
#ifdef MG
  timer.enter_subsection("setup_multigrid");
  pcout << "Setup multigrid" << std::endl;
  setup_multigrid ();
  timer.leave_subsection();
#endif
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
  triangulation.refine_global (n_levels-1);
  timer.leave_subsection();
  pcout << "Finite element: " << fe.get_name() << std::endl;
  pcout << "Number of active cells: "
        << triangulation.n_global_active_cells()
        << std::endl;
  timer.enter_subsection("setup_system");
  pcout << "Setup system" << std::endl;
  setup_system ();
  timer.leave_subsection();
  dealii::deallog << "DoFHandler levels: ";
  for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dof_handler.n_dofs(l);
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

#include "Simulator.inst"
