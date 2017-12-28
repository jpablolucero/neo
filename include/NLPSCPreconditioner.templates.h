#include <NLPSCPreconditioner.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

namespace
{
  namespace NLWorkStream
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
      VectorType *src;
      dealii::MGLevelObject<VectorType> *solution;
      const dealii::Mapping<dim>  *mapping;
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
      ddh.reinit(copy.local_solution, subdomain_idx);

      dealii::MeshWorker::DoFInfo<dim> dof_info (ddh.get_dofh().block_info());
      dealii::MeshWorker::IntegrationInfoBox<dim> info_box ;
      const unsigned int n_gauss_points = ddh.get_dofh().get_fe().degree+1;
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
      
      src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(scratch.solution,"src");
      src_data.add<const dealii::MGLevelObject<LA::MPI::Vector >*>(scratch.solution,"Newton iterate");
      info_box.initialize(ddh.get_dofh().get_fe(), *(scratch.mapping), src_data, LA::MPI::Vector {}, &(ddh.get_dofh().block_info()));
      
      std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >
	colored_iterators(1,copy.ddh->subdomain_to_global_map[subdomain_idx]);

      dealii::MGLevelObject<dealii::FullMatrix<double> > mg_matrix ;
      mg_matrix.resize(ddh.get_level(),ddh.get_level());
      mg_matrix[ddh.get_level()] = std::move(dealii::FullMatrix<double>(copy.ddh->global_dofs_on_subdomain[subdomain_idx].size()));
      Assembler::MGMatrixSimpleMapped<dealii::FullMatrix<double> > massembler;
      massembler.initialize(mg_matrix);
      massembler.initialize(copy.ddh->all_to_unique[subdomain_idx]);
      MatrixIntegrator<dim> matrix_integrator;

      dealii::MGLevelObject<dealii::Vector<double> > mg_vector ;
      mg_vector.resize(ddh.get_level(),ddh.get_level());
      mg_vector[ddh.get_level()] = std::move(dealii::Vector<double>(copy.ddh->global_dofs_on_subdomain[subdomain_idx].size()));
      dealii::AnyData adata;
      adata.add<dealii::Vector<double> *>(&(mg_vector[ddh.get_level()]), "RHS");
      Assembler::ResidualSimpleMapped<dealii::Vector<double> > rassembler;
      rassembler.initialize(adata);
      rassembler.initialize(copy.ddh->all_to_unique[subdomain_idx]);
      RHSIntegrator<dim> rhs_integrator(ddh.get_dofh().get_fe().n_components());
      
      auto inverse_derivative = [&](dealii::Vector<number> & Du, dealii::Vector<number> & res)
	{
	  mg_matrix[ddh.get_level()] = 0.;

	  dealii::MeshWorker::LoopControl lctrl;
	  lctrl.own_faces = dealii::MeshWorker::LoopControl::both;
	  lctrl.ghost_cells = true;
	  dealii::colored_loop<dim, dim> (colored_iterators, dof_info, info_box, matrix_integrator,
					  massembler,lctrl,colored_iterators[0]);

	  dealii::LAPACKFullMatrix<double> m ;
	  m.copy_from(mg_matrix[ddh.get_level()]) ;
	  m.compute_inverse_svd();
	  Du = 0.;
	  m.vmult(Du,res);
	};

      auto residual = [&](dealii::Vector<number> & res,dealii::Vector<number> & u)
	{
	  mg_vector[ddh.get_level()] = 0. ;
	  dealii::MeshWorker::LoopControl lctrl;
	  lctrl.own_faces = dealii::MeshWorker::LoopControl::both;
	  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
	  lctrl.ghost_cells = false;
	  dealii::colored_loop<dim, dim> (colored_iterators, dof_info, info_box, rhs_integrator,
	  				  rassembler,lctrl, colored_iterators[0]);
	  res = mg_vector[ddh.get_level()] ;
	};
      
      dealii::Vector<number>  u;
      dealii::Vector<number>  u0;
      dealii::Vector<number>  Du;
      dealii::Vector<number>  res;
      ddh.reinit(u, subdomain_idx);
      ddh.reinit(u0, subdomain_idx);
      ddh.restrict_add(u, (*(scratch.solution))[ddh.get_level()], subdomain_idx);
      ddh.restrict_add(u0, (*(scratch.solution))[ddh.get_level()], subdomain_idx);
      Du.reinit(u);
      res.reinit(u);
      residual(res,u);
      double resnorm = res.l2_norm();
      const unsigned int n_stepsize_iterations = 21;
      const double resnorm0 = resnorm ;
      while(resnorm/resnorm0 > 1.E-2)
      {
        Du.reinit(u);
      	inverse_derivative(Du,res);
      	u.add(-1.,Du);
      	copy.ddh->prolongate((*scratch.solution)[ddh.get_level()],u,copy.subdomain_idx);
	double old_residual = resnorm;
	dealii::IndexSet locally_relevant_level_dofs;
	dealii::DoFTools::extract_locally_relevant_level_dofs
	  (ddh.get_dofh(), ddh.get_level(), locally_relevant_level_dofs);
	residual(res,u);
      	resnorm = res.l2_norm();
      	// Step size control
        unsigned int step_size = 0;
        while (resnorm >= old_residual)
          {
            ++step_size;
            if (step_size > n_stepsize_iterations) break;
            u.add(1./(1<<step_size), Du);
      	    copy.ddh->prolongate((*scratch.solution)[ddh.get_level()],u,copy.subdomain_idx);
      	    residual(res,u);
            resnorm = res.l2_norm();
          }
      }
      copy.ddh->prolongate((*scratch.solution)[ddh.get_level()],u0,copy.subdomain_idx);
      copy.local_solution -= u;
    }
  }
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::NLPSCPreconditioner()
{}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::~NLPSCPreconditioner()
{
  system_matrix = nullptr ;
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number,same_diagonal>::initialize(const SystemMatrixType & system_matrix_,
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
#ifndef MATRIXFREE
    ghosted_solution.resize(level, level);
#if PARALLEL_LA == 0
    ghosted_solution[level].reinit(locally_owned_level_dofs.n_elements());
#elif PARALLEL_LA < 3
    ghosted_solution[level].reinit(locally_owned_level_dofs,
                                   locally_relevant_level_dofs,
                                   *mpi_communicator,true);
#else
    ghosted_solution[level].reinit(locally_owned_level_dofs,
                                   locally_relevant_level_dofs,
                                   *mpi_communicator);
#endif // PARALLEL_LA
    ghosted_solution[level] = *(data.solution);
#endif // MATRIXFREE
  }

  if (data.patch_type == AdditionalData::PatchType::cell_patches)
    ddh.reset(new DGDDHandlerCell<dim>());
  else
    ddh.reset(new DGDDHandlerVertex<dim>());
  ddh->initialize(dof_handler, level);

  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> (dof_handler.block_info()));

  ordered_iterators.clear();
  ordered_gens.clear();
  auto & dirs = data.dirs ;
  for (unsigned int d = 0 ; d < dirs.size() ; ++d)
    {
      ordered_iterators.resize(d+1);
      ordered_gens.resize(d+1);
      add_cell_ordering(dirs[d]);
    }
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::clear()
{}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult (VectorType &dst,
    const VectorType &src) const
{
  dst = 0;
  ghosted_solution[level] = *(data.solution) ;
  dst = *(data.solution) ;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  dst *= data.relaxation;
  // AssertIsFinite(dst.l2_norm());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult_add (VectorType &dst,
    const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);

  if (data.smoother_type == AdditionalData::SmootherType::additive)
    {
      // TODO make sure that the source vector is ghosted
      NLWorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;

      NLWorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.src = &ghosted_src;
      scratch_sample.solution = &ghosted_solution;
      scratch_sample.mapping = data.mapping ;

      ghosted_src = src;
      dealii::WorkStream::run(ddh->colorized_iterators(),
			      NLWorkStream::work<dim, VectorType, number, same_diagonal>,
			      NLWorkStream::assemble<dim, VectorType, number, same_diagonal>,
			      scratch_sample, copy_sample);
    }
  else AssertThrow(false, dealii::ExcNotImplemented());
  
  timer->leave_subsection();
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult_add (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::add_cell_ordering(dealii::Tensor<1,dim> dir)
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
