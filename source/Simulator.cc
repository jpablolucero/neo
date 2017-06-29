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
  triangulation(mpi_communicator,dealii::Triangulation<dim>::
                limit_level_difference_at_vertices,
                dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  mapping (degree),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1),
#else
  fe(dealii::FE_DGQ<dim>(degree),1),
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


  // const double half_width = 1. ;
  // dealii::GridGenerator::hyper_cube (triangulation, -half_width, half_width, true);
  // std::cout << "Physical domain:   [" << -half_width << ", " << half_width << "]^" << dim << std::endl;

  // create ball domain
  const dealii::Point<dim> center ;
  const double             radius {1.0} ;
  dealii::GridGenerator::hyper_ball (triangulation, center, radius) ;

  static const dealii::SphericalManifold<dim> boundary_description (center);
  triangulation.set_all_manifold_ids_on_boundary (0);
  triangulation.set_manifold (0, boundary_description);


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
  /*  std::cout << "locally relevant dofs on process "
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

#ifdef MATRIXFREE
  system_matrix.reinit (&dof_handler,&mapping,&constraints,mpi_communicator,dealii::numbers::invalid_unsigned_int);
#else
  system_matrix.reinit (&dof_handler,&mapping,&constraints,mpi_communicator,triangulation.n_global_levels()-1);
#endif

#ifdef MATRIXFREE
#if PARALLEL_LA == 3
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(solution_tmp);
  right_hand_side.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
#elif PARALLEL_LA == 0
  solution.reinit (locally_owned_dofs.n_elements());
  solution_tmp.reinit (locally_owned_dofs.n_elements());
  right_hand_side.reinit (locally_owned_dofs.n_elements());
#else // PARALLEL_LA == 1,2
  AssertThrow(false, dealii::ExcNotImplemented());
#endif // PARALLEL_LA == 3

#else // MATRIXFREE OFF
#if PARALLEL_LA == 0
  solution.reinit (locally_owned_dofs.n_elements());
  solution_tmp.reinit (locally_owned_dofs.n_elements());
  right_hand_side.reinit (locally_owned_dofs.n_elements());
#elif PARALLEL_LA == 3
  solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (locally_owned_dofs, mpi_communicator);
  right_hand_side.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
#else
  solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  solution_tmp.reinit (locally_owned_dofs, mpi_communicator);
  right_hand_side.reinit (locally_owned_dofs, mpi_communicator);
#endif // PARALLEL_LA == 0
#endif // MATRIXFREE
}

template <int dim, bool same_diagonal, unsigned int degree>
void Simulator<dim, same_diagonal, degree>::assemble_system ()
{
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  const unsigned int n_gauss_points = degree+1;
#ifdef CG
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
#else
  info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
#endif // CG
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
                                     dealii::update_values | dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);
  
  dealii::AnyData src_data;
#ifdef MATRIXFREE
  info_box.initialize(fe, mapping);
  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler);
#else
  src_data.add<const LA::MPI::Vector *>(&solution,"Newton iterate");
  info_box.initialize(fe,mapping,src_data,LA::MPI::Vector {},&(dof_handler.block_info()));
  dealii::MeshWorker::DoFInfo<dim> dof_info(dof_handler.block_info());
#endif // MATRIXFREE

// //   dealii::AnyData data;
// //   data.add<LA::MPI::Vector *>(&right_hand_side, "RHS");

// //   ResidualSimpleConstraints<LA::MPI::Vector > rhs_assembler;
// // //  dealii::MeshWorker::Assembler::ResidualSimple<LA::MPI::Vector > rhs_assembler;
// //   rhs_assembler.initialize(data);
// // #ifdef CG
// //   rhs_assembler.initialize(constraints);
// // #endif
// //   RHSIntegrator<dim> rhs_integrator(fe.n_components());

// //   dealii::MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(),
// //                                                  dof_handler.end(),
// //                                                  dof_info,
// //                                                  info_box,
// //                                                  rhs_integrator,
// //                                                  rhs_assembler);

  const unsigned int fe_degree = degree ;
  const unsigned int n_components = 1 ;
  typedef double number ;
    //  dealii::QGauss<dim>   quadrature_formula (n_q_points_1d);
  dealii::QGauss<dim>   quadrature_formula (fe_degree+1);
  dealii::QGauss<dim-1> quadrature_face (fe_degree+1);
  dealii::FEValues<dim> fe_values (mapping
				   , fe
                                   , quadrature_formula
                                   , dealii::update_values
                                   | dealii::update_gradients
                                   | dealii::update_JxW_values
                                   | dealii::update_quadrature_points);
  dealii::FEFaceValues<dim> fe_face_values (mapping
					    , fe
                                            , quadrature_face
                                            , dealii::update_values
                                            | dealii::update_gradients
                                            | dealii::update_quadrature_points
                                            | dealii::update_normal_vectors);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points = quadrature_formula.size();
  const unsigned int   n_q_points_face = quadrature_face.size();
    
  dealii::Vector<number>                 cell_rhs(dofs_per_cell) ;
  std::vector<dealii::types::global_dof_index>   local_dof_indices (dofs_per_cell);

  RightHandSide<dim>    rhs_function {n_components};
  std::vector<double>   rhs_values(n_q_points);
  
  Solution<dim>         g_Dirichlet {n_components};
  std::vector<double>   bdry_values(n_q_points_face)  ;

  Coefficient<dim>      coefficient_function ;
  std::vector<double>   coeff_values(n_q_points_face) ;
  
  typename dealii::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
	cell_rhs = 0. ;
        fe_values.reinit (cell);

        // CELL integral
        rhs_function.value_list (fe_values.get_quadrature_points(), rhs_values);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
	  for (unsigned int q=0; q<n_q_points; ++q)
	    cell_rhs(i)
	      += fe_values.JxW(q)
	      * fe_values.shape_value(i,q)
	      * rhs_values[q] ;

        // BOUNDARY integral (Nitsche)
        for (unsigned int face_no=0; face_no<dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
          if (cell->at_boundary(face_no))
            {
              fe_face_values.reinit(cell, face_no);

	      const unsigned int normal_dir = dealii::GeometryInfo<dim>::unit_normal_direction[face_no];
	      const double sigmaF
		= static_cast<double>( (fe_degree == 0) ? 1. : fe_degree * (fe_degree+1.) )
		/ cell->extent_in_direction(normal_dir) ;
	      
              const auto &q_points = fe_face_values.get_quadrature_points ();
              g_Dirichlet.value_list (q_points, bdry_values);
	      coefficient_function.value_list (q_points, coeff_values);

	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		for (unsigned int q=0; q<quadrature_face.size(); ++q)
		  cell_rhs(i)
		    += coeff_values[q] * fe_face_values.JxW(q)
		    * ( 2. * sigmaF * fe_face_values.shape_value(i,q)
			- fe_face_values.shape_grad(i,q) * fe_face_values.normal_vector(q) )
		    * bdry_values[q] ;
            } // is boundary face ?

        cell->get_dof_indices (local_dof_indices);
	for (unsigned int i=0; i<dofs_per_cell; ++i)
	  right_hand_side(local_dof_indices[i]) += cell_rhs(i);
      } // is locally owned ?
  
  right_hand_side.compress(dealii::VectorOperation::add);
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::setup_multigrid ()
{
  const unsigned int n_global_levels = triangulation.n_global_levels();
  mg_matrix.resize(min_level, n_global_levels-1);
#ifndef MATRIXFREE
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer;
  mg_transfer.build_matrices(dof_handler);
  mg_solution.resize(min_level, n_global_levels-1);
  mg_transfer.copy_to_mg(dof_handler,mg_solution,solution);
  system_matrix.reinit (&dof_handler,&mapping, &constraints, mpi_communicator, triangulation.n_global_levels()-1, solution);
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].set_timer(timer);
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints,mpi_communicator,level,mg_solution[level]);
    }
#else // MATRIXFREE ON
  for (unsigned int level=min_level; level<n_global_levels; ++level)
    {
      mg_matrix[level].set_timer(timer);
      mg_matrix[level].reinit(&dof_handler,&mapping,&constraints,mpi_communicator,level);
    }
#endif // MATRIXFREE
}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::solve ()
{
  timer.enter_subsection("solve::mg_initialization");
#ifdef MG
  // Setup coarse solver
  dealii::SolverControl coarse_solver_control (dof_handler.n_dofs(min_level)*10, 1e-16, false, false);
  dealii::PreconditionIdentity id;
#if PARALLEL_LA < 3
  // mg_matrix[min_level].build_coarse_matrix();
  //  const LA::MPI::SparseMatrix &coarse_matrix = mg_matrix[min_level].get_coarse_matrix();
  dealii::SolverCG<LA::MPI::Vector> coarse_solver(coarse_solver_control);
  dealii::MGCoarseGridIterativeSolver<LA::MPI::Vector,
				      dealii::SolverCG<LA::MPI::Vector>,
				      SystemMatrixType, /*LA::MPI::SparseMatrix*/
				      dealii::PreconditionIdentity>
		  mg_coarse(coarse_solver,
			    mg_matrix[min_level], /*coarse_matrix*/
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

  const double dictionary_tolerance = 0.01 ;
  dealii::MGLevelObject<typename Smoother::AdditionalData> smoother_data;
  smoother_data.resize(mg_matrix.min_level(), mg_matrix.max_level());
  for (unsigned int level = mg_matrix.min_level();
       level <= mg_matrix.max_level();
       ++level)
    {
      smoother_data[level].dof_handler = &dof_handler;
      smoother_data[level].level = level;
      smoother_data[level].mapping = &mapping;
      smoother_data[level].relaxation = 0.7;
      smoother_data[level].mg_constrained_dofs = mg_constrained_dofs;
#ifndef MATRIXFREE
      smoother_data[level].solution = &mg_solution[level];
#endif // MATRIXFREE
      smoother_data[level].mpi_communicator = mpi_communicator;

      // uncomment to use the dictionary
      if(!same_diagonal)
       {
         smoother_data[level].use_dictionary = false;
         smoother_data[level].tol = dictionary_tolerance;
       }

      smoother_data[level].patch_type = Smoother::AdditionalData::cell_patches;
    }
  std::cout << "Dictionary status:    (Tolerance)    (" << dictionary_tolerance << ")" << std::endl;
  dealii::MGSmootherPrecondition<SystemMatrixType,Smoother,LA::MPI::Vector> mg_smoother;

  {
    // // construct scale factors per cell
    // Coefficient<dim> coeff;
    // typedef typename dealii::DoFHandler<dim>::level_cell_iterator level_cell_iterator;
    // std::vector<std::map<level_cell_iterator,double> > cell_to_factor(n_levels);
    // unsigned int l = 0;
    // for (unsigned int level = smoother_data.min_level();
    //      level <= smoother_data.max_level();
    //      ++level, ++l)
    //   {
    //     for ( auto cell=dof_handler.begin_mg(level);
    //           cell!=dof_handler.end_mg(level);
    //           ++cell )
    //       cell_to_factor[l].insert ( std::pair<level_cell_iterator,double>(cell,coeff.value(cell->center())) );
    //     smoother_data[level].cell_to_factor = &(cell_to_factor[l]);
    //   }

    mg_smoother.initialize(mg_matrix, smoother_data);
    mg_smoother.set_steps(smoothing_steps);

    // // avoid dangling pointers
    // for (unsigned int level = smoother_data.min_level();
    //      level <= smoother_data.max_level();
    //      ++level, ++l)
    //   smoother_data[level].cell_to_factor = nullptr;
  }

  // Setup Multigrid-Transfer
#ifdef MATRIXFREE
  dealii::MGTransferMF<dim,SystemMatrixType> mg_transfer {mg_matrix};
  mg_transfer.build(dof_handler);
#else // MATRIXFREE OFF
  dealii::MGTransferPrebuilt<LA::MPI::Vector> mg_transfer {};
#ifdef CG
  mg_transfer.initialize_constraints(constraints, mg_constrained_dofs);
#endif // CG
  mg_transfer.build_matrices(dof_handler);
#endif // MATRIXFREE

  // Setup (Multigrid-)Preconditioner
  dealii::mg::Matrix<LA::MPI::Vector>         mglevel_matrix;
  mglevel_matrix.initialize(mg_matrix);
  dealii::Multigrid<LA::MPI::Vector> mg(mglevel_matrix,
                                        mg_coarse,
                                        mg_transfer,
                                        mg_smoother,
                                        mg_smoother,
                                        mg_matrix.min_level(),
                                        mg_matrix.max_level());
  // mg.set_debug(10);
#ifdef MATRIXFREE
  dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferMF<dim,SystemMatrixType> >
  preconditioner(dof_handler, mg, mg_transfer);
#else
  dealii::PreconditionMG<dim, LA::MPI::Vector, dealii::MGTransferPrebuilt<LA::MPI::Vector> >
  preconditioner(dof_handler, mg, mg_transfer);
#endif // MATRIXFREE

#else // MG OFF
  dealii::PreconditionIdentity preconditioner;
#endif // MG

  // Setup Solver
  dealii::ReductionControl             solver_control (dof_handler.n_dofs(), 1.e-20, 1.e-10,true);
  dealii::SolverCG<LA::MPI::Vector> solver (solver_control);
  timer.leave_subsection();

  // Solve the system
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
void Simulator<dim, same_diagonal, degree>::compute_error (unsigned int cycle)
{
  dealii::QGauss<dim>    quadrature {degree+4} ;
  dealii::Vector<float>  difference_per_cell_local (triangulation.n_active_cells ()) ;

  dealii::VectorTools::integrate_difference (mapping
				     , dof_handler
				     , solution
				     , reference_function
				     , difference_per_cell_local
				     , quadrature
					     , dealii::VectorTools::L2_norm);
  // dealii::VectorTools::integrate_difference (mapping, dof_handler,
  //                                            solution,
  //                                            reference_function,
  //                                            local_errors, quadrature,
  //                                            dealii::VectorTools::L2_norm);

  const double L2_error_local = difference_per_cell_local.l2_norm();
  const double   L2_error
    = std::sqrt(dealii::Utilities::MPI::sum(L2_error_local * L2_error_local,
                                            mpi_communicator));
  // = VectorTools::compute_global_error (triangulation,
  //   					 difference_per_cell_local,
  //   					 VectorTools::L2_norm);

  difference_per_cell_local = 0.;
  dealii::VectorTools::integrate_difference (mapping
				     , dof_handler
				     , solution
				     , reference_function
				     , difference_per_cell_local
				     , quadrature
					     , dealii::VectorTools::H1_seminorm);

  const double H1_error_local = difference_per_cell_local.l2_norm();
  const double H1_error
    = std::sqrt(dealii::Utilities::MPI::sum(H1_error_local * H1_error_local,
    					    mpi_communicator));
    // = VectorTools::compute_global_error (triangulation,
    // 					 difference_per_cell_local,
    // 					 VectorTools::H1_seminorm);

  //  pcout << "L2 error: " << L2_error << std::endl;
  const unsigned int n_dofs = dof_handler.n_dofs () ;
  const unsigned int n_active_cells = triangulation.n_active_cells () ;

        convergence_table.add_value ("#levels", cycle);
  	convergence_table.add_value ("cells", n_active_cells);
  	convergence_table.add_value ("dofs", n_dofs);
  	convergence_table.add_value ("L2", L2_error);
  	convergence_table.add_value ("H1", H1_error);
  // // dealii::QGauss<dim> quadrature (degree+2);
  // // dealii::Vector<double> local_errors;

  // // dealii::VectorTools::integrate_difference (mapping, dof_handler,
  // //                                            solution,
  // //                                            reference_function,
  // //                                            local_errors, quadrature,
  // //                                            dealii::VectorTools::L2_norm);
  // // const double L2_error_local = local_errors.l2_norm();
  // // const double L2_error
  // //   = std::sqrt(dealii::Utilities::MPI::sum(L2_error_local * L2_error_local,
  // //                                           mpi_communicator));

  // // pcout << "L2 error: " << L2_error << std::endl;
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
  for (unsigned int cycle = min_level+1; cycle < n_levels; ++cycle)
    {
      timer.reset();
      timer.enter_subsection("refine_global");
      pcout << "Refine global" << std::endl;
      //      triangulation.refine_global (n_levels-1);
      triangulation.refine_global ();
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
  for (unsigned int l=min_level; l<triangulation.n_global_levels(); ++l)
    dealii::deallog << ' ' << dof_handler.n_dofs(l);
  dealii::deallog << std::endl;
#ifdef MG

  timer.enter_subsection("setup_multigrid");
  pcout << "Setup multigrid" << std::endl;
  setup_multigrid ();
  timer.leave_subsection();
#endif

  // output_results(n_levels);
  timer.enter_subsection("solve");
  pcout << "Solve" << std::endl;
  solve ();
  timer.leave_subsection();
  // timer.enter_subsection("output");
  // pcout << "Output" << std::endl;
  compute_error(cycle);
  output_results(cycle);
  // timer.leave_subsection();
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

    convergence_table.set_precision ("L2", 3);
    convergence_table.set_precision ("H1", 3);
    convergence_table.set_scientific ("L2", true);
    convergence_table.set_scientific ("H1", true);
    
    convergence_table.evaluate_convergence_rates("L2"
    						 , dealii::ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("L2"
    						 , "dofs"
    						 , dealii::ConvergenceTable::reduction_rate_log2
    						 , dim);
    convergence_table.evaluate_convergence_rates("H1", dealii::ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("H1"
    						 , "dofs"
    						 , dealii::ConvergenceTable::reduction_rate_log2
    						 , dim);
    
    std::cout << std::endl ;
    convergence_table.write_text (std::cout) ;
    std::cout << std::endl ;

}

template <int dim,bool same_diagonal,unsigned int degree>
void Simulator<dim,same_diagonal,degree>::run_non_linear ()
{
  // timer.reset();
  //     timer.enter_subsection("refine_global");
  //     pcout << "Refine global" << std::endl;
  //     triangulation.refine_global (n_levels-1);
  // timer.leave_subsection();
  // pcout << "Finite element: " << fe.get_name() << std::endl;
  // pcout << "Number of active cells: "
  //       << triangulation.n_global_active_cells()
  //       << std::endl;
  // timer.enter_subsection("setup_system");
  // pcout << "Setup system" << std::endl;
  // setup_system ();
  // timer.leave_subsection();
  // dealii::deallog << "DoFHandler levels: ";
  // for (unsigned int l=0; l<triangulation.n_global_levels(); ++l)
  //   dealii::deallog << ' ' << dof_handler.n_dofs(l);
  // dealii::deallog << std::endl;
  // auto sol = solution_tmp ;
  // for (auto &elem : sol) elem = 1. ;
  // dealii::AnyData solution_data;
  // solution_data.add(&sol, "solution");
  // dealii::AnyData data;
  // newton.control.set_reduction(1.E-10);
  // timer.enter_subsection("solve");
  // pcout << "Solve" << std::endl;
  // newton(solution_data, data);
  // solution = *(solution_data.try_read_ptr<LA::MPI::Vector>("solution"));
  // timer.leave_subsection();
  // // timer.enter_subsection("output");
  // // output_results(n_levels);
  // // timer.leave_subsection();
  // timer.print_summary();
  // pcout << std::endl;
  // // workaround regarding issue #2533
  // // GrowingVectorMemory does not destroy the vectors
  // // after this instance goes out of scope.
  // // Unfortunately, the mpi_communicators given to the
  // // remaining vectors might be invalid the next time
  // // a vector is requested. Therefore, clean up everything
  // // before going out of scope.
  // dealii::GrowingVectorMemory<LA::MPI::Vector>::release_unused_memory();
}

#include "Simulator.inst"
