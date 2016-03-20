#include <MyLaplace.h>
#include <GlobalTimer.h>
#include <DDHandler.h>
#include <PSCPreconditioner.h>

template <int dim,bool same_diagonal>
MyLaplace<dim,same_diagonal>::MyLaplace (const unsigned int degree)
  :
  mpi_communicator(MPI_COMM_WORLD),
  triangulation(mpi_communicator,dealii::Triangulation<dim>::
                limit_level_difference_at_vertices,
                dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  mapping (),
  fe (degree),
  dof_handler (triangulation),
  pcout (std::cout,(dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0))
{
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

  dealii::GridGenerator::hyper_cube (triangulation,-1.,1., true);

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

template <int dim,bool same_diagonal>
MyLaplace<dim,same_diagonal>::~MyLaplace ()
{}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();

  locally_owned_dofs = dof_handler.locally_owned_dofs();

  dealii::deallog << "locally owned dofs on process "
                  << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
                  << std::endl;
  for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
    {
      dealii::deallog << "level: " << l << " n_elements(): "
                      << dof_handler.locally_owned_mg_dofs(l).n_elements()
                      << " index set: ";
      dof_handler.locally_owned_mg_dofs(l).print(dealii::deallog);
    }
  dealii::deallog << "n_elements(): "
                  << dof_handler.locally_owned_dofs().n_elements()
                  <<std::endl;
  dof_handler.locally_owned_dofs().print(dealii::deallog);

  dealii::DoFTools::extract_locally_relevant_dofs
  (dof_handler, locally_relevant_dofs);
  std::cout << "locally relevant dofs on process "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
  locally_relevant_dofs.print(std::cout);

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

  solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (locally_owned_dofs, mpi_communicator);
  right_hand_side.reinit (locally_owned_dofs, mpi_communicator);

  system_matrix.reinit (&dof_handler,&mapping, &constraints, mpi_communicator, triangulation.n_global_levels()-1) ;
}


template <int dim, bool same_diagonal>
void MyLaplace<dim, same_diagonal>::assemble_system ()
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

  info_box.initialize(fe, mapping);

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);

//  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
  dealii::AnyData data;
  data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);
#ifdef CG
  rhs_assembler.initialize(constraints);
#endif

  RHSIntegrator<dim> rhs_integrator;

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
                                                 dof_info, info_box,
                                                 rhs_integrator, rhs_assembler);
  right_hand_side.compress(dealii::VectorOperation::add);
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::setup_multigrid ()
{
  const unsigned int n_global_levels = triangulation.n_global_levels();
  mg_matrix.resize(0, n_global_levels-1);
  for (unsigned int level=0; level<n_global_levels; ++level)
    {
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints, mpi_communicator, level);
      mg_matrix[level].build_matrix();
    }
//  coarse_matrix.reinit(dof_handler.n_dofs(0),dof_handler.n_dofs(0));
//  coarse_matrix.copy_from(mg_matrix[0]) ;
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve ()
{
  if (use_psc)
    {
      solve_psc();
    }
  else
    {
      solve_blockjacobi();
    }
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve_psc ()
{
  global_timer.enter_subsection("solve::mg_initialization");
#ifdef MG
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[0].get_coarse_matrix();

  dealii::SolverControl coarse_solver_control (dof_handler.n_dofs(0), 1e-10, false, false);
  dealii::SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::PreconditionIdentity id;
  dealii::MGCoarseGridLACIteration<dealii::SolverCG<LA::MPI::Vector>,LA::MPI::Vector> mg_coarse(coarse_solver,
      coarse_matrix,
      id);

  // Smoother setup
  typedef PSCPreconditioner<dim, LA::MPI::Vector, double> Smoother;

  dealii::MGLevelObject<dealii::FullMatrix<double> > local_level_inverse;
  local_level_inverse.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<DGDDHandler<dim> > level_ddh;
  level_ddh.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());

  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      // init ddhandler
      level_ddh[level].initialize(dof_handler, level);

      // init local inverse with first local level matrix
      const unsigned int n = dof_handler.get_fe().n_dofs_per_cell();
      typename dealii::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active(level);
      while (!cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
        ++cell;
      std::vector<dealii::types::global_dof_index> first_level_dof_indices (n);
      cell->get_mg_dof_indices (first_level_dof_indices);
      dealii::FullMatrix<double> local_matrix(n, n);
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          {
            const dealii::types::global_dof_index i1 = first_level_dof_indices [i];
            const dealii::types::global_dof_index i2 = first_level_dof_indices [j];

            local_matrix(i, j) = mg_matrix[level](i1, i2);
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
  LA::MPI::Vector> mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(6);
  dealii::mg::Matrix<LA::MPI::Vector>         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  dealii::Multigrid<LA::MPI::Vector> mg(dof_handler, mgmatrix,
                                        mg_coarse, mg_transfer,
                                        mg_smoother, mg_smoother);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, LA::MPI::Vector,
         dealii::MGTransferPrebuilt<LA::MPI::Vector> >
         preconditioner(dof_handler, mg, mg_transfer);
#else
  dealii::PreconditionIdentity preconditioner;
#endif

  dealii::ReductionControl          solver_control (dof_handler.n_dofs(), 1.e-20, 1.e-10);
  dealii::SolverCG<LA::MPI::Vector> solver (solver_control);

  global_timer.leave_subsection();
  global_timer.enter_subsection("solve::solve");
  constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution_tmp,right_hand_side,preconditioner);
#ifdef CG
  constraints.distribute(solution_tmp);
#endif
  solution = solution_tmp;
  global_timer.leave_subsection();
}

template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::solve_blockjacobi ()
{
  global_timer.enter_subsection("solve::mg_initialization");
#ifdef MG
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[0].get_coarse_matrix();

  dealii::SolverControl coarse_solver_control (dof_handler.n_dofs(0), 1e-10, false, false);
  dealii::SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::PreconditionIdentity id;
  dealii::MGCoarseGridLACIteration<dealii::SolverCG<LA::MPI::Vector>,LA::MPI::Vector> mg_coarse(coarse_solver,
      coarse_matrix,
      id);


//  dealii::MGCoarseGridSVD<float, LA::MPI::Vector >    mg_coarse;
//  dealii::MGCoarseGridHouseholder<float, LA::MPI::Vector> mg_coarse;
//  typename dealii::PreconditionBlockJacobi<SystemMatrixType >::AdditionalData
//      smoother_data(dof_handler.block_info().local().block_size(0),1.0,true,true);

//  mg_coarse.initialize(coarse_matrix);
//  dealii::MGCoarseGridLACIteration<double,
//              LA::MPI::Vector >    mg_coarse;
//  mg_coarse.initialize(coarse_matrix, 1.e-15);
//  typename LA::PreconditionBlockJacobi::AdditionalData
//    smoother_data(dof_handler.block_info().local().block_size(0),"linear",1.,0,1);

  dealii::MGSmootherPrecondition<SystemMatrixType,
         dealii::PreconditionIdentity,
//                 LA::PreconditionBlockJacobi,
//           dealii::PreconditionBlockJacobi<SystemMatrixType >,
         LA::MPI::Vector> mg_smoother;
  //mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(1);
  dealii::mg::Matrix<LA::MPI::Vector >         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  dealii::Multigrid<LA::MPI::Vector > mg(dof_handler, mgmatrix,
                                         mg_coarse, mg_transfer,
                                         mg_smoother, mg_smoother);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, LA::MPI::Vector,
         dealii::MGTransferPrebuilt<LA::MPI::Vector > >
         preconditioner(dof_handler, mg, mg_transfer);
#else
  dealii::PreconditionIdentity preconditioner;
#endif

  dealii::ReductionControl          solver_control (dof_handler.n_dofs(), 1.e-20, 1.e-10);
  dealii::SolverCG<LA::MPI::Vector> solver (solver_control);
  global_timer.leave_subsection();
  global_timer.enter_subsection("solve::solve");
  constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution_tmp,right_hand_side,preconditioner);
#ifdef CG
  constraints.distribute(solution_tmp);
#endif
  solution = solution_tmp;
  global_timer.leave_subsection();
}

template <int dim,bool same_diagonal>
void MyLaplace<dim, same_diagonal>::compute_error () const
{
  dealii::QGauss<dim> quadrature (fe.degree+2);
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

template <int dim, bool same_diagonal>
void MyLaplace<dim, same_diagonal>::output_results (const unsigned int cycle) const
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
  if (n_proc >1)
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


template <int dim,bool same_diagonal>
void MyLaplace<dim,same_diagonal>::run ()
{
  triangulation.refine_global (1);
  for (unsigned int cycle=0; cycle<6-dim; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      global_timer.reset();

      global_timer.enter_subsection("refine_global");
      pcout << "Refine global" << std::endl;
      triangulation.refine_global (1);
      global_timer.leave_subsection();

      dealii::deallog << "Number of active cells: " <<
                      triangulation.n_global_active_cells() << std::endl;
      global_timer.enter_subsection("setup_system");
      pcout << "Setup system" << std::endl;
      setup_system ();
      pcout << "Assemble system" << std::endl;
      assemble_system();
      global_timer.leave_subsection();
      dealii::deallog << "DoFHandler levels: ";
      for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
        dealii::deallog << ' ' << dof_handler.n_dofs(l);
      dealii::deallog << std::endl;
#ifdef MG
      global_timer.enter_subsection("setup_multigrid");
      pcout << "Setup multigrid" << std::endl;
      setup_multigrid ();
      global_timer.leave_subsection();
#endif
      global_timer.enter_subsection("solve");
      pcout << "Solve" << std::endl;
      solve ();
      global_timer.leave_subsection();
      global_timer.enter_subsection("output");
      pcout << "Ouput" << std::endl;
      compute_error();
      output_results(cycle);
      global_timer.leave_subsection();
      global_timer.print_summary();
      dealii::deallog << std::endl;
    }
}

template class MyLaplace<2,true>;
template class MyLaplace<3,true>;
template class MyLaplace<2,false>;
template class MyLaplace<3,false>;
//template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, true>,double >;
//template class dealii::PreconditionBlockJacobi<LaplaceOperator<2, 1, false>,double >;
//template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, true>,double >;
//template class dealii::PreconditionBlockJacobi<LaplaceOperator<3, 1, false>,double >;
