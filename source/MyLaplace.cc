#include <MyLaplace.h>

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mpi_communicator(MPI_COMM_WORLD),
  triangulation(mpi_communicator,dealii::Triangulation<dim>::
                limit_level_difference_at_vertices,
                dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  mapping (),
  fe (1),
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

  //add periodicity
  typedef typename dealii::Triangulation<dim>::cell_iterator CellIteratorTria;
  std::vector<dealii::GridTools::PeriodicFacePair<CellIteratorTria> > periodic_faces;
  const unsigned int b_id1 = 2;
  const unsigned int b_id2 = 3;
  const unsigned int direction = 1;

  dealii::GridTools::collect_periodic_faces (triangulation, b_id1, b_id2,
                                             direction, periodic_faces, dealii::Tensor<1,dim>());
  triangulation.add_periodicity(periodic_faces);
}

template <int dim>
MyLaplace<dim>::~MyLaplace ()
{}

template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs(fe);
  dof_handler.initialize_local_block_info();

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  std::cout << "locally owned dofs on process "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
  locally_owned_dofs.print(std::cout);

  dealii::DoFTools::extract_locally_relevant_dofs
  (dof_handler, locally_relevant_dofs);
  std::cout << "locally relevant dofs on process "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator) << " ";
  locally_relevant_dofs.print(std::cout);

  //Periodic boundary conditions
  std::vector<dealii::GridTools::PeriodicFacePair
  <typename dealii::DoFHandler<dim>::cell_iterator> >
  periodic_faces;

  const unsigned int b_id1 = 2;
  const unsigned int b_id2 = 3;
  const unsigned int direction = 1;

  dealii::GridTools::collect_periodic_faces (dof_handler,
                                             b_id1, b_id2, direction,
                                             periodic_faces);

  //constraints
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
#ifdef CG
  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<dim> >
    (periodic_faces, constraints);

  dealii::DoFTools::make_hanging_node_constraints
  (dof_handler, constraints);

  for (unsigned int i=0; i<2*dim; ++i)
    dealii::VectorTools::interpolate_boundary_values(dof_handler, i,
                                                     reference_function,
                                                     constraints);
#endif
  constraints.close();

  solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  right_hand_side.reinit (locally_owned_dofs, mpi_communicator);

  system_matrix.reinit (&dof_handler,&mapping, &constraints, mpi_communicator, triangulation.n_levels()-1) ;

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
                                     dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping);

  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
//  ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
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

template <int dim>
void MyLaplace<dim>::setup_multigrid ()
{
  const unsigned int n_levels = triangulation.n_levels();
  mg_matrix.resize(0, n_levels-1);  
  for (unsigned int level=0;level<n_levels;++level)
    {
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints, mpi_communicator, level);
      mg_matrix[level].build_matrix(true);
    }
//  coarse_matrix.reinit(dof_handler.n_dofs(0),dof_handler.n_dofs(0));
//  coarse_matrix.copy_from(mg_matrix[0]) ;
}

template <int dim>
void MyLaplace<dim>::solve ()
{
  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[0].get_coarse_matrix();
  dealii::MGTransferPrebuilt<LA::MPI::Vector > mg_transfer;
  mg_transfer.build_matrices(dof_handler);

  dealii::SolverControl coarse_solver_control (1000, 1e-10, false, false);
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
                         LA::PreconditionIdentity,
//                 LA::PreconditionBlockJacobi,
//           dealii::PreconditionBlockJacobi<SystemMatrixType >,
                 LA::MPI::Vector> mg_smoother;
  //mg_smoother.initialize(mg_matrix, smoother_data);
  mg_smoother.set_steps(1);
  dealii::mg::Matrix<LA::MPI::Vector >         mgmatrix;
  mgmatrix.initialize(mg_matrix);
  dealii::Multigrid<LA::MPI::Vector > mg(dof_handler, mgmatrix,
						mg_coarse, mg_transfer,
						mg_smoother, mg_smoother);
  mg.set_minlevel(mg_matrix.min_level());
  mg.set_maxlevel(mg_matrix.max_level());
  dealii::PreconditionMG<dim, LA::MPI::Vector,
             dealii::MGTransferPrebuilt<LA::MPI::Vector > >
  preconditioner(dof_handler, mg, mg_transfer);  
  dealii::ReductionControl          solver_control (1000, 1.E-20, 1.E-10);
  dealii::SolverCG<LA::MPI::Vector> solver (solver_control);
  solution_tmp=0.;
  constraints.set_zero(solution_tmp);
  solver.solve(system_matrix,solution,right_hand_side,preconditioner);
  constraints.distribute(solution_tmp);
  solution = solution_tmp;
}

template <int dim>
void MyLaplace<dim>::compute_error () const
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

template <int dim>
void MyLaplace<dim>::output_results (const unsigned int cycle) const
{
    std::string filename = "solution-"+dealii::Utilities::int_to_string(cycle,2);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "u_free");
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


template <int dim>
void MyLaplace<dim>::run ()
{
  for (unsigned int cycle=0; cycle<9-dim; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
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
      output_results (cycle);
      dealii::deallog << std::endl;
    }
}

//template void dealii::VectorTools::integrate_difference<2, LA::MPI::Vector, LA::MPI::Vector , 2>
//(const Mapping< 2, 2> &mapping, const DoFHandler< 2, 2> &dof,
// const LA::MPI::Vector &fe_function, const Function<2, double > &exact_solution,
// LA::MPI::Vector &difference, const Quadrature<2> &q, const NormType &norm,
// const Function< 2, double > *weight=0, const double exponent=2.);
//template void dealii::VectorTools::integrate_difference<3, LA::MPI::Vector, LA::MPI::Vector , 3>
//(const Mapping< 3, 3> &mapping, const DoFHandler< 3, 3> &dof,
// const LA::MPI::Vector &fe_function, const Function<3, double > &exact_solution,
// LA::MPI::Vector &difference, const Quadrature<3> &q, const NormType &norm,
// const Function< 3, double > *weight=0, const double exponent=2.);
template class MyLaplace<2>;
template class MyLaplace<3>;
//template class LA::PreconditionBlockJacobi<LaplaceOperator<2,1>,double >;
//template class LA::PreconditionBlockJacobi<LaplaceOperator<3,1>,double >;
