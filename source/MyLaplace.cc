#include <MyLaplace.h>
#include <DDHandler.h>
#include <PSCPreconditioner.h>
#include <deal.II/dofs/dof_renumbering.h>

template <int dim,bool same_diagonal,unsigned int degree>
MyLaplace<dim,same_diagonal,degree>::MyLaplace ()
  :
  mpi_communicator(MPI_COMM_WORLD),
  triangulation(mpi_communicator,dealii::Triangulation<dim>::
                limit_level_difference_at_vertices,
                dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  mapping (),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1),
#else
  fe(dealii::FE_DGQ<dim>(degree),1),
#endif
  dof_handler (triangulation),
  pcout (std::cout,(dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)),
  timer(mpi_communicator, pcout, dealii::TimerOutput::never,dealii::TimerOutput::wall_times)
{
  LaplaceOperator<dim, degree, same_diagonal>::timer = &timer;
  //initialize timer
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

template <int dim,bool same_diagonal,unsigned int degree>
MyLaplace<dim,same_diagonal,degree>::~MyLaplace ()
{}

template <int dim,bool same_diagonal,unsigned int degree>
void MyLaplace<dim,same_diagonal,degree>::setup_system ()
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

  system_matrix.reinit (&dof_handler,&mapping, &constraints, mpi_communicator, triangulation.n_global_levels()-1);
}


template <int dim, bool same_diagonal, unsigned int degree>
void MyLaplace<dim, same_diagonal, degree>::assemble_system ()
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

  info_box.initialize(fe, mapping, &(dof_handler.block_info()));

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());

  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
  dealii::AnyData data;
  data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);
  rhs_assembler.initialize(constraints);

  RHSIntegrator<dim> rhs_integrator;

  dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
                                                 dof_info, info_box,
						 rhs_integrator, rhs_assembler);
  right_hand_side.compress(dealii::VectorOperation::add);
  // dof_handler.block_info().print(dealii::deallog);
  // for( unsigned int c=0; c<dof_handler.n_dofs(); ++c)
  //   dealii::deallog << "AFTER::rhs[" << c << "]=" << right_hand_side[c] << std::endl;  
}

template <int dim,bool same_diagonal,unsigned int degree>
void MyLaplace<dim,same_diagonal,degree>::setup_multigrid ()
{
  const unsigned int n_global_levels = triangulation.n_global_levels();
  mg_matrix.resize(0, n_global_levels-1);
  for (unsigned int level=0; level<n_global_levels; ++level)
    {
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints, mpi_communicator, level);
      mg_matrix[level].build_matrix();
    }
}

template <int dim,bool same_diagonal,unsigned int degree>
void MyLaplace<dim,same_diagonal,degree>::solve ()
{
  timer.enter_subsection("solve::mg_initialization");
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
  Smoother::timer = &timer;

  dealii::MGLevelObject<std::vector<dealii::FullMatrix<double> > > local_level_inverse;
  local_level_inverse.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<DGDDHandler<dim> > level_ddh;
  level_ddh.resize(mg_matrix.min_level(), mg_matrix.max_level());
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());

  const unsigned int n = dof_handler.get_fe().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> first_level_dof_indices (n);
  dealii::FullMatrix<double> local_matrix(n, n);

  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      // init ddhandler
      level_ddh[level].initialize(dof_handler, level);
      smoother_data[level].local_inverses.resize(level_ddh[level].size());

      // setup smoother data
      smoother_data[level].ddh = &(level_ddh[level]);
      smoother_data[level].weight = 1.0;

      if (same_diagonal)
        {
          // init local inverse with first local level matrix
          local_level_inverse[level].resize(1, dealii::FullMatrix<double>(n));
          typename dealii::DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(level);
          while (cell!=dof_handler.end_mg(level)
                 && cell->level_subdomain_id()!=triangulation.locally_owned_subdomain())
            ++cell;

          if (cell!=dof_handler.end_mg(level))
            {
              cell->get_active_or_mg_dof_indices (first_level_dof_indices);
              local_matrix=0.;
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  {
                    const dealii::types::global_dof_index i1 = first_level_dof_indices [i];
                    const dealii::types::global_dof_index i2 = first_level_dof_indices [j];
                    local_matrix(i, j) = mg_matrix[level](i1, i2);
                  }

              local_level_inverse[level][0].invert(local_matrix);

              for (unsigned int i=0; i<level_ddh[level].size(); ++i)
                smoother_data[level].local_inverses[i]=&(local_level_inverse[level][0]);
            }
        }
      else
        {
          //just store information for locally owned cells
          local_level_inverse[level].resize(level_ddh[level].size(), dealii::FullMatrix<double>(n));
          unsigned int subdomain_idx = 0;
          for (auto cell = dof_handler.begin_mg(level);
               cell != dof_handler.end_mg(level);
               ++cell)
            if (cell->level_subdomain_id()==triangulation.locally_owned_subdomain())
              {
                cell->get_active_or_mg_dof_indices (first_level_dof_indices);
                local_matrix = 0.;
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int j = 0; j < n; ++j)
                    {
                      const dealii::types::global_dof_index i1 = first_level_dof_indices [i];
                      const dealii::types::global_dof_index i2 = first_level_dof_indices [j];
                      local_matrix(i, j) = mg_matrix[level](i1, i2);
                    }

                local_level_inverse[level][subdomain_idx].invert(local_matrix);

                smoother_data[level].local_inverses[subdomain_idx]
                  =&(local_level_inverse[level][subdomain_idx]);
                ++subdomain_idx;
              }
          AssertThrow(level_ddh[level].size()==subdomain_idx,
                      dealii::ExcDimensionMismatch(level_ddh[level].size(), subdomain_idx));
        }
    }

  // SmootherSetup
  dealii::MGSmootherPrecondition<SystemMatrixType, Smoother, LA::MPI::Vector> mg_smoother;
  mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(SMOOTHENINGSTEPS);
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
void MyLaplace<dim, same_diagonal, degree>::compute_error () const
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
void MyLaplace<dim, same_diagonal, degree>::output_results (const unsigned int cycle) const
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


template <int dim,bool same_diagonal,unsigned int degree>
void MyLaplace<dim,same_diagonal,degree>::run ()
{
  for (unsigned int cycle=0; cycle<10-2*dim; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;
      timer.reset();
      timer.enter_subsection("refine_global");
      pcout << "Refine global" << std::endl;
      triangulation.refine_global (1);
      timer.leave_subsection();
      dealii::deallog << "Finite element: " << fe.get_name() << std::endl;
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
      output_results(cycle);
      timer.leave_subsection();
      timer.print_summary();
      pcout << std::endl;
    }
}

template class MyLaplace<2,true,1>;
template class MyLaplace<2,true,2>;
template class MyLaplace<2,true,3>;
template class MyLaplace<2,true,4>;
template class MyLaplace<3,true,1>;
template class MyLaplace<3,true,2>;
template class MyLaplace<3,true,3>;
template class MyLaplace<3,true,4>;
template class MyLaplace<2,false,1>;
template class MyLaplace<2,false,2>;
template class MyLaplace<2,false,3>;
template class MyLaplace<2,false,4>;
template class MyLaplace<3,false,1>;
template class MyLaplace<3,false,2>;
template class MyLaplace<3,false,3>;
template class MyLaplace<3,false,4>;
