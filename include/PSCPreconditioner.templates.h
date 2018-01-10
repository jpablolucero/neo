#include <PSCPreconditioner.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

namespace
{
  namespace WorkStream
  {
    template <int dim, typename VectorType, typename number, bool same_diagonal>
    class Copy
    {
    public:
      dealii::Vector<number>  local_solution;
      unsigned int subdomain_idx;
      VectorType *dst;

      std::shared_ptr<DDHandlerBase<dim> > ddh;
    };

    template <int dim, typename VectorType, typename number, bool same_diagonal>
    class Scratch
    {
    public:
      const std::vector<std::shared_ptr<dealii::LAPACKFullMatrix<double> > > *local_inverses;

      dealii::Vector<number>  local_src;
      const VectorType *src;
    };

    template <int dim, typename VectorType, typename number, bool same_diagonal>
    void assemble(const Copy<dim, VectorType, number, same_diagonal> &copy)
    {
      // write back to global vector
      copy.ddh->prolongate_add(*(copy.dst),
                               copy.local_solution,
                               copy.subdomain_idx);
    }

    template <int dim, typename VectorType, typename number, bool same_diagonal>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, VectorType, number, same_diagonal> &scratch, Copy<dim, VectorType, number, same_diagonal> &copy)
    {
      const unsigned int subdomain_idx = *iterator;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);

      copy.subdomain_idx = subdomain_idx;
      ddh.reinit(scratch.local_src, subdomain_idx);
      ddh.reinit(copy.local_solution, subdomain_idx);
      // get local contributions and copy to dealii vector
      ddh.restrict_add(scratch.local_src, *(scratch.src), subdomain_idx);
      (*(scratch.local_inverses))[subdomain_idx]->vmult(copy.local_solution,
                                                        scratch.local_src);
    }
  }
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::PSCPreconditioner()
{}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::~PSCPreconditioner()
{
  system_matrix = nullptr ;
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number,same_diagonal>::initialize(const SystemMatrixType & system_matrix_,
    const AdditionalData &data)
{
  Assert(data.dof_handler != 0, dealii::ExcInternalError());
  Assert(data.level != dealii::numbers::invalid_unsigned_int, dealii::ExcInternalError());
  Assert(data.mapping != 0, dealii::ExcInternalError());

  system_matrix = &system_matrix_ ;
  
  this->data = data;
  level = data.level;
  const dealii::DoFHandler<dim> &dof_handler = *(data.dof_handler);
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();

  // We need to be able to get the values on locally relevant dofs
  {
#ifdef DEBUG
    const dealii::parallel::distributed::Triangulation<dim> *distributed_tria
      = dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>* > (&(dof_handler.get_triangulation()));
#endif
    Assert(distributed_tria, dealii::ExcInternalError());

    dealii::IndexSet locally_owned_level_dofs = dof_handler.locally_owned_mg_dofs(level);
    dealii::IndexSet locally_relevant_level_dofs;
    dealii::DoFTools::extract_locally_relevant_level_dofs
    (dof_handler, level, locally_relevant_level_dofs);
    ghosted_src.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
    ghosted_dst.reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
    ghosted_solution.resize(level, level);
    ghosted_solution[level].reinit(locally_owned_level_dofs,
                                   locally_relevant_level_dofs,
                                   *mpi_communicator);
    ghosted_solution[level] = *(data.solution);
  }

  if (data.patch_type == AdditionalData::PatchType::cell_patches)
    ddh.reset(new DGDDHandlerCell<dim>());
  else
    ddh.reset(new DGDDHandlerVertex<dim>());
  ddh->initialize(dof_handler, level);

  const unsigned int n_gauss_points = fe.degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
                                       n_gauss_points,
                                       n_gauss_points);
  info_box.initialize_update_flags();
  const dealii::UpdateFlags update_flags_cell
    = dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_values | dealii::update_gradients;
  const dealii::UpdateFlags update_flags_face
    = dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_values | dealii::update_gradients | dealii::update_normal_vectors;
  info_box.add_update_flags_boundary(update_flags_face);
  info_box.add_update_flags_face(update_flags_face);
  info_box.add_update_flags_cell(update_flags_cell);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);
  info_box.cell_selector.add("Newton iterate", true, true, false);
  info_box.boundary_selector.add("Newton iterate", true, true, false);
  info_box.face_selector.add("Newton iterate", true, true, false);

  dealii::AnyData src_data ;
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_solution,"src");
  src_data.add<const dealii::MGLevelObject<VectorType >*>(&ghosted_solution,"Newton iterate");
  info_box.initialize(fe, *(data.mapping), src_data, VectorType {},&(dof_handler.block_info()));
  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> (dof_handler.block_info()));

  patch_inverses.resize(ddh->global_dofs_on_subdomain.size());
  //setup local matrices/inverses
  {
    timer->enter_subsection("PSC::build_patch_inverses");

    // SAME_DIAGONAL
    if (same_diagonal && !data.use_dictionary && patch_inverses.size()!=0)
      {
        // TODO broadcast local patch inverse instead of solving a local problem
        // find the first interior cell if there is any and use it
        typename std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >::iterator it
          = ddh->subdomain_to_global_map.begin();
        unsigned int subdomain = 0;

        for (int i=0; it!=ddh->subdomain_to_global_map.end(); ++it, ++i)
          {
            bool all_interior = true;
            typename std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator>::iterator it_cell
              = it->begin();
            for (; it_cell!=it->end(); ++it_cell)
              if ((*it_cell)->at_boundary())
                {
                  all_interior = false;
                  break;
                }
            if (all_interior)
              {
                subdomain = i;
                break;
              }
          }

        const unsigned int n = fe.n_dofs_per_cell();
        patch_inverses[0].reset(new LAPACKMatrix(n));
        build_matrix(ddh->subdomain_to_global_map[subdomain],
                     ddh->global_dofs_on_subdomain[subdomain],
                     ddh->all_to_unique[subdomain],
                     *patch_inverses[0]);
        // patch_inverses[0]->print_formatted(std::cout);

        patch_inverses[0]->compute_inverse_svd();
        for ( unsigned int j=1; j<patch_inverses.size(); ++j )
          patch_inverses[j] = patch_inverses[0];
      }

    // FULL BLOCK JACOBI
    else if (!same_diagonal && !data.use_dictionary && patch_inverses.size()!=0)
      {
        dealii::Threads::TaskGroup<> tasks;
        for (unsigned int i=0; i<ddh->subdomain_to_global_map.size(); ++i)
          {
            tasks += dealii::Threads::new_task([i,this]()
            {
              patch_inverses[i].reset(new LAPACKMatrix);
              build_matrix(ddh->subdomain_to_global_map[i],
                           ddh->global_dofs_on_subdomain[i],
                           ddh->all_to_unique[i],
                           *patch_inverses[i]);
              // std::cout << std::endl;
              // patch_inverses[i]->print_formatted(std::cout);
              patch_inverses[i]->compute_inverse_svd();
            });
          }
        tasks.join_all ();
      }

    // DICTIONARY
    else if (!same_diagonal && data.use_dictionary && patch_inverses.size()!=0)
      {
        Assert(data.tol > 0., dealii::ExcInternalError());
        std::vector<unsigned int> id_range;
        for ( unsigned int id=0; id<ddh->global_dofs_on_subdomain.size(); ++id)
          id_range.push_back(id);
        unsigned int patch_id, dict_size = 0;
        // loop over subdomain range
        while (id_range.size()!=0)
          {
            // build local inverse of first subdomain in the remaining id_range of subdomains
            patch_id = id_range.front();
            patch_inverses[patch_id].reset(new LAPACKMatrix);
            build_matrix(ddh->subdomain_to_global_map[patch_id],
                         ddh->global_dofs_on_subdomain[patch_id],
                         ddh->all_to_unique[patch_id],
                         *patch_inverses[patch_id]);
            patch_inverses[patch_id]->invert();
            id_range.erase(id_range.begin());
            ++dict_size;
            // check 'inverse-similarity' with the remainder of subdomains
            auto j = id_range.begin();
            while (j!=id_range.end())
              {
                LAPACKMatrix A_j;
                build_matrix(ddh->subdomain_to_global_map[*j],
                             ddh->global_dofs_on_subdomain[*j],
                             ddh->all_to_unique[*j],
                             A_j);
                dealii::FullMatrix<double> S_j {A_j.m()};
                patch_inverses[patch_id]->mmult(S_j,A_j);
                S_j.diagadd(-1.);
                // test if currently observed inverse is a good approximation of inv(A_j)
                Assert(S_j.m() == S_j.n(), dealii::ExcInternalError());
                const double tol_m = data.tol * S_j.m();
                if (S_j.frobenius_norm() < tol_m)
                  {
                    patch_inverses[*j] = patch_inverses[patch_id];
                    j = id_range.erase(j);
                  }
                else
                  ++j;
              }
          }
        //Output
        dealii::deallog << "DEAL::Dictionary(level=" << level
                        << ", mpi_proc=" << 0
                        << ", Tol=" << data.tol << ") contains "
                        << dict_size << " inverse(s)." << std::endl;
      }
  }
  ordered_iterators.clear();
  ordered_gens.clear();
  auto & dirs = data.dirs ;
  for (unsigned int d = 0 ; d < dirs.size() ; ++d)
    {
      ordered_iterators.resize(d+1);
      ordered_gens.resize(d+1);
      add_cell_ordering(dirs[d]);
    }
  timer->leave_subsection();
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::clear()
{}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult (VectorType &dst,
    const VectorType &src) const
{
  ghosted_dst = 0;
  vmult_add(ghosted_dst, src);
  ghosted_dst.compress(dealii::VectorOperation::add);
  dst = ghosted_dst;
  dst *= data.relaxation;
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult_add (VectorType &dst,
    const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);

  if (data.smoother_type == AdditionalData::SmootherType::additive)
    {
      //TODO make sure that the source vector is ghosted
      WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;

      WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.src = &ghosted_src;
      scratch_sample.local_inverses = &patch_inverses;

      ghosted_src = src;
      dealii::WorkStream::run(ddh->colorized_iterators(),
			      WorkStream::work<dim, VectorType, number, same_diagonal>,
			      WorkStream::assemble<dim, VectorType, number, same_diagonal>,
			      scratch_sample, copy_sample);
    }
  else if (data.smoother_type == AdditionalData::SmootherType::additive_with_coarse)
    {
      ghosted_src = src;
      const dealii::DoFHandler<dim> &dof_handler = *(data.dof_handler);
      VectorType dummy1(dof_handler.n_dofs(level-1)),
	dummy2(dof_handler.n_dofs(level-1)),
	dummy3(dof_handler.n_dofs(level));
      VectorType dummy0(ghosted_src);
      dealii::ReductionControl control(10000, 1e-20, 1e-14, false, false) ;
      dealii::SolverGMRES<VectorType> coarse_solver(control);
      dealii::MGTransferPrebuilt<VectorType > mg_transfer ;
      dealii::MGLevelObject<VectorType > mg_solution ;
      mg_transfer.build_matrices(*(data.dof_handler));
      mg_transfer.restrict_and_add(data.level,dummy1,dummy0);
      coarse_solver.solve(*(data.coarse_matrix),dummy2,dummy1,dealii::PreconditionIdentity{});
      mg_transfer.prolongate(data.level,dummy3,dummy2);
      dst = dummy3 ;
      //TODO make sure that the source vector is ghosted
      WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;

      WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.src = &ghosted_src;
      scratch_sample.local_inverses = &patch_inverses;

      ghosted_src = src;
      dealii::WorkStream::run(ddh->colorized_iterators(),
			      WorkStream::work<dim, VectorType, number, same_diagonal>,
			      WorkStream::assemble<dim, VectorType, number, same_diagonal>,
			      scratch_sample, copy_sample);
    }
  else if (data.smoother_type == AdditionalData::SmootherType::hybrid)
    {
      Assert(ddh->colorized_iterators().size() == 1, dealii::ExcInternalError());
      int local_n_subdomains = ddh->subdomain_to_global_map.size() ;
      int max_subdomains = 0;
      const int ierr = MPI_Allreduce (&local_n_subdomains, &max_subdomains, 1, MPI_INT, MPI_MAX, *mpi_communicator);
      AssertThrowMPI(ierr);

      WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;

      WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.src = &ghosted_src;
      scratch_sample.local_inverses = &patch_inverses;

      ghosted_src = src ;
      for (int p = 0 ; p < max_subdomains ; ++p)
	  {
	    system_matrix->vmult(ghosted_src,dst);
	    ghosted_src.sadd(-1.,src);
	    if ((p < local_n_subdomains) and (ddh->get_dofh().locally_owned_mg_dofs(level).n_elements() != 0))
	      {
		WorkStream::work(ddh->colorized_iterators()[0][p],scratch_sample,copy_sample) ;
		WorkStream::assemble(copy_sample) ;
	      }
	  }
    }      
  else if (data.smoother_type == AdditionalData::SmootherType::multiplicative)
    {
      WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;

      WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.src = &ghosted_src;
      scratch_sample.local_inverses = &patch_inverses;

      ghosted_src = src ;
      for (unsigned int d = 0 ; d < ordered_iterators.size() ; ++d)
	for (unsigned int g = 0 ; g < global_last_gen ; ++g)
	  {
	    std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> selected_iterators ;
	    for (unsigned int s = 0 ; s < ordered_iterators[d].size() ; ++s)
	      {
		int loc_g = g-ordered_gens[d][s] ;
		for (int sm_g = loc_g-1 ; sm_g <= loc_g+1 ; ++sm_g)
		  if ( (0 <= sm_g) and (sm_g < static_cast<int>(ordered_iterators[d][s].size())) )
		    for (auto & cell : ordered_iterators[d][s][sm_g])
		      selected_iterators.push_back(ddh->subdomain_to_global_map[*cell][0]);
	      }
	    const_cast<SystemMatrixType*>(system_matrix)->set_cell_range(selected_iterators);
	    system_matrix->vmult(ghosted_src,dst);
	    const_cast<SystemMatrixType*>(system_matrix)->unset_cell_range();
	    ghosted_src.sadd(-1.,src);
	    for (unsigned int s = 0 ; s < ordered_iterators[d].size() ; ++s)
	      {
		const unsigned int loc_g = g-ordered_gens[d][s] ;
		if ( (0 <= loc_g) and (loc_g < ordered_iterators[d][s].size()) )
		  for (unsigned int c = 0 ; c < ordered_iterators[d][s][loc_g].size() ; ++c)
		    {
		      WorkStream::work(ordered_iterators[d][s][loc_g][c],scratch_sample,copy_sample) ;
		      WorkStream::assemble(copy_sample) ;
		    }
	      }
	  }
    }
  else
    AssertThrow(false, dealii::ExcNotImplemented());
  
  timer->leave_subsection();
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult_add (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::build_matrix
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
 const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
 const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
 dealii::LAPACKFullMatrix<double> &matrix)
{
  dealii::MGLevelObject<dealii::FullMatrix<double> > mg_matrix ;
  mg_matrix.resize(level,level);

  mg_matrix[level] = std::move(dealii::FullMatrix<double>(global_dofs_on_subdomain.size()));

  Assembler::MGMatrixSimpleMapped<dealii::FullMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  // assembler.initialize(data.mg_constrained_dofs);
#endif
  assembler.initialize(all_to_unique);

  //now assemble everything
  dealii::MeshWorker::LoopControl lctrl;
  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::one;
  lctrl.ghost_cells = true;
  //TODO possibly colorize iterators, assume thread-safety for the moment
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > colored_iterators(1, cell_range);


  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler,lctrl, colored_iterators[0]);

  matrix.copy_from(mg_matrix[level]);
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void PSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::add_cell_ordering(dealii::Tensor<1,dim> dir)
{
  Assert(ddh->colorized_iterators().size() == 1, dealii::ExcInternalError());
  class CellVectorContainer
  {
  public:
    CellVectorContainer(std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > &
			subdomain_to_global_map_):
      subdomain_to_global_map(subdomain_to_global_map_){};
    auto & operator [](unsigned int c) {return subdomain_to_global_map[c][0];}
    auto size() {return subdomain_to_global_map.size();}
    auto begin() {return subdomain_to_global_map.begin();}
    auto end() {return subdomain_to_global_map.end();}
  private:
    std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > & subdomain_to_global_map;
  } local_cells(ddh->subdomain_to_global_map) ;
  const unsigned int n_local_cells = local_cells.size() ;
  auto & local_iterators = ddh->colorized_iterators()[0];
  auto & triangulation = ddh->get_dofh().get_triangulation();
  auto & fe = ddh->get_dofh().get_fe();
      
  dealii::QGauss<dim-1> face_quadrature_formula(3);
  dealii::FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,dealii::update_normal_vectors);

  std::vector<unsigned int> b_data(n_local_cells,0);
  std::vector<unsigned int> p_data(n_local_cells,0);
  std::vector<unsigned int> i_data(n_local_cells,0);

  for (unsigned int c = 0 ; c < n_local_cells ; ++c)
    for (unsigned int f = 0 ; f < 2*dim ; ++f)
      {
	if (local_cells[c]->at_boundary(f))
	  b_data[c] += 1;
	else if (local_cells[c]->neighbor(f)->level_subdomain_id()!=triangulation.locally_owned_subdomain())
	  p_data[c] += 1;
      }
      	      
  bool global_gen = false ;
  std::set<unsigned int> level_ghost_owners ;
  std::set<unsigned int> upstream_procs ;
      
  std::function<void(int,dealii::Tensor<1,dim>,unsigned int)> sweep = [&](int c,dealii::Tensor<1,dim> dir_,unsigned int gen)
    {
      std::vector<unsigned int> downstream_faces ;
      unsigned int d_data = 0;
      for (unsigned int f = 0 ; f < 2*dim ; ++f)
	if (not (local_cells[c]->at_boundary(f)))
	  {
	    if (local_cells[c]->neighbor(f)->level_subdomain_id()==triangulation.locally_owned_subdomain())
	      {
		fe_face_values.reinit(local_cells[c],f) ;
		auto nor = fe_face_values.normal_vector(0);
		if ( dir_*nor > 0 )
		  {
		    d_data += 1;
		    downstream_faces.push_back(f);
		  }
	      }
	    else
	      {
		level_ghost_owners.insert(local_cells[c]->neighbor(f)->level_subdomain_id());
		fe_face_values.reinit(local_cells[c],f) ;
		auto nor = fe_face_values.normal_vector(0);
		if (gen == 0)
		  {
		    if ( dir_*nor < 0 ) upstream_procs.insert(local_cells[c]->neighbor(f)->level_subdomain_id());
		    else
		      {
			p_data[c] -= 1 ;
			d_data += 1 ;
		      }
		  }
	      }		    
	  }
      if (b_data[c]+p_data[c]+i_data[c]+d_data < 2*dim) return ;
      if (b_data[c]+p_data[c]+i_data[c]+d_data == 2*dim)
	{
	  if (b_data[c]+d_data == 2*dim)
	    {
	      global_gen = true ;
	      ordered_gens.back().back() = 0;
	    }
	  ++gen ;
	  if (gen > ordered_iterators.back().back().size()) ordered_iterators.back().back().resize(gen);
	  ordered_iterators.back().back()[gen-1].push_back(local_iterators[c]);
	  for (auto f : downstream_faces)
	    {
	      std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> v{local_cells[c]->neighbor(f)};
	      int q = std::distance(local_cells.begin(),std::find(local_cells.begin(),local_cells.end(),v)) ;
	      i_data[q] += 1;
	      sweep(q,dir_,gen);
	    }
	}
      return;
    };
	      
  auto find_sweep_starting_cell = [&](dealii::Tensor<1,dim> dir_)
    {
      std::vector<unsigned int> cells ;
      for (unsigned int c = 0 ; c < n_local_cells ; ++c)
	{
	  unsigned int d_data = 0;
	  for (unsigned int f = 0 ; f < 2*dim ; ++f)
	    if (not (local_cells[c]->at_boundary(f)) and
		(local_cells[c]->neighbor(f)->level_subdomain_id()==triangulation.locally_owned_subdomain()))
	      {
		fe_face_values.reinit(local_cells[c],f) ;
		auto nor = fe_face_values.normal_vector(0);
		if ( dir_*nor > 0 ) d_data += 1;
	      }
	  if (b_data[c]+p_data[c]+d_data == 2*dim) cells.push_back(c);
	}
      return cells ;
    };

  for (auto c : find_sweep_starting_cell(dir))
    {
      ordered_iterators.back().resize(ordered_iterators.back().size()+1);
      ordered_gens.back().resize(ordered_gens.back().size()+1,-1);
      sweep(c,dir,0);
    }

  const unsigned int n_proc = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int this_proc = dealii::Utilities::MPI::this_mpi_process(*mpi_communicator);

  downstream_outbox.clear();
  
  for (unsigned int proc = 0 ; proc < n_proc ; ++proc)
    {
      if (this_proc == proc)
	{
	  for (auto iproc : level_ghost_owners)
	    {
	      bool downstream_request = 0 ;
	      MPI_Recv(&downstream_request,1,MPI_LOGICAL,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
	      if (downstream_request)
		{
		  unsigned int n_downstream_requests = 0 ;
		  MPI_Recv(&n_downstream_requests,1,MPI_INT,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		  for (unsigned int r = 0 ; r < n_downstream_requests ; ++r)
		    {
		      std::vector<double> requested_cell(dim,0.) ;
		      MPI_Recv(&requested_cell[0],dim,MPI_DOUBLE,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		      unsigned int downstream_sweep = 0;
		      MPI_Recv(&downstream_sweep,1,MPI_INT,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		      for (unsigned int s = 0 ; s < ordered_iterators.back().size() ; ++s)
			for (unsigned int g = 0 ; g < ordered_iterators.back()[s].size() ; ++g)
			  for (unsigned int c = 0 ; c < ordered_iterators.back()[s][g].size() ; ++c)
			    {
			      auto & this_cell = local_cells[*(ordered_iterators.back()[s][g][c])] ;
			      double distance = 0 ;
			      for (unsigned int d = 0 ; d < dim ; ++d)
				distance += std::pow(this_cell->center()[d]-requested_cell[d],2) ;
			      if (distance < 1.E-8 )
				downstream_outbox.push_back(std::vector<unsigned int> {s,g,c,iproc,downstream_sweep});
			    }
		    }
		}
	    }
	}
      else if (level_ghost_owners.find(proc) != level_ghost_owners.end()) 
	{
	  const bool upstream_request = (upstream_procs.find(proc)!=upstream_procs.end()) ? true : false ;
	  MPI_Send(&upstream_request,1,MPI_LOGICAL,proc,0,*mpi_communicator);
	  if (upstream_request)
	    {
	      std::vector<std::vector<double> > upstream_cell_requests ;
	      std::vector<unsigned int> local_sweep_list ;
	      for (unsigned int s = 0 ; s < ordered_iterators.back().size() ; ++s)
		{		      
		  auto & this_cell = local_cells[*(ordered_iterators.back()[s][0][0])] ;
		  for (unsigned int f = 0 ; f < 2*dim ; ++f)
		    if ((not (this_cell->at_boundary(f))) and (this_cell->neighbor(f)->level_subdomain_id()==proc))
		      {
			fe_face_values.reinit(this_cell,f) ;
			auto nor = fe_face_values.normal_vector(0);
			if ( dir*nor < 0 )
			  {
			    std::vector<double> cell(dim,0.) ;
			    for (unsigned int d = 0 ; d < dim ; ++d) cell[d] = this_cell->neighbor(f)->center()[d];
			    upstream_cell_requests.push_back(cell);
			    local_sweep_list.push_back(s) ;
			  }
		      }
		}
	      const unsigned int n_upstream_requests = upstream_cell_requests.size() ;
	      MPI_Send(&n_upstream_requests,1,MPI_INT,proc,0,*mpi_communicator);
	      for (unsigned int r = 0 ; r < n_upstream_requests ; ++r)
		{
		  MPI_Send(&upstream_cell_requests[r][0],dim,MPI_DOUBLE,proc,0,*mpi_communicator);
		  MPI_Send(&local_sweep_list[r],1,MPI_INT,proc,0,*mpi_communicator);
		}			  
	    }
	}
    }

  bool complete_gen_data = false ;
  while (not complete_gen_data)
    {
      for (unsigned int proc = 0 ; proc < n_proc ; ++proc)
	{
	  if (this_proc == proc)
	    {
	      for (auto iproc : level_ghost_owners)
		{
		  unsigned int n_generations = 0 ;
		  MPI_Recv(&n_generations,1,MPI_INT,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		  for (unsigned int i = 0 ; i < n_generations ; ++i)
		    {
		      unsigned int generation = 0 ;
		      unsigned int sweep = 0 ;
		      MPI_Recv(&generation,1,MPI_INT,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		      MPI_Recv(&sweep,1,MPI_INT,iproc,0,*mpi_communicator,MPI_STATUS_IGNORE);
		      if (ordered_gens.back()[sweep] == -1) ordered_gens.back()[sweep] = generation + 1 ;
		    }
		}
	    }
	  else if (level_ghost_owners.find(proc) != level_ghost_owners.end()) 
	    {
	      std::vector<unsigned int> generations ;
	      std::vector<unsigned int> dest_sweeps ;
	      for (unsigned int s = 0 ; s < ordered_gens.back().size() ; ++s)
		if (ordered_gens.back()[s] != -1)
		  {
		    for (unsigned int i = 0 ; i < downstream_outbox.size() ; ++i)
		      if ( (downstream_outbox[i][3] == proc) and (downstream_outbox[i][0] == s) )
			{
			  generations.push_back(downstream_outbox[i][1] + ordered_gens.back()[s]);
			  dest_sweeps.push_back(downstream_outbox[i][4]);
			}
		  }
	      const unsigned int n_generations = generations.size() ;
	      MPI_Send(&n_generations,1,MPI_INT,proc,0,*mpi_communicator) ;
	      for (unsigned int i = 0 ; i < n_generations ; ++i)
		{
		  MPI_Send(&generations[i],1,MPI_INT,proc,0,*mpi_communicator) ;
		  MPI_Send(&dest_sweeps[i],1,MPI_INT,proc,0,*mpi_communicator) ;
		}
	    }
	}
      bool gen_data_is_complete = true ;
      for (auto & elem : ordered_gens.back())
	if (elem == -1) gen_data_is_complete = false ;
      MPI_Allreduce (&gen_data_is_complete, &complete_gen_data, 1, MPI_LOGICAL, MPI_LAND, *mpi_communicator);
    }

  unsigned int local_last_gen = 0 ;
  for (unsigned int s = 0 ; s < ordered_iterators.back().size() ; ++s)
    if (local_last_gen < static_cast<unsigned int>(ordered_gens.back()[s])+ordered_iterators.back()[s].size() )
      local_last_gen = static_cast<unsigned int>(ordered_gens.back()[s])+ordered_iterators.back()[s].size() ;
      
  global_last_gen = 0 ;
  MPI_Allreduce (&local_last_gen, &global_last_gen, 1, MPI_INT, MPI_MAX, *mpi_communicator);
}
