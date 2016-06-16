#include <PSCPreconditioner.h>

namespace implementation
{
  namespace WorkStream
  {
    template <int dim, typename VectorType, class number, bool same_diagonal>
    class Copy
    {
    public:
      dealii::Vector<number>  local_solution;
      unsigned int subdomain_idx;
      VectorType *dst;

      std::shared_ptr<DDHandlerBase<dim> > ddh;
    };

    template <int dim, typename VectorType, class number, bool same_diagonal>
    class Scratch
    {
    public:
      const std::vector<dealii::FullMatrix<double>* > *local_inverses;

      dealii::Vector<number>  local_src;
      const VectorType *src;
    };

    template <int dim, typename VectorType, class number, bool same_diagonal>
    void assemble(const Copy<dim, VectorType, number, same_diagonal> &copy)
    {
      // write back to global vector
      copy.ddh->prolongate_add(*(copy.dst),
                               copy.local_solution,
                               copy.subdomain_idx);
    }

    template <int dim, typename VectorType, class number, bool same_diagonal>
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

template <int dim, typename VectorType, class number, bool same_diagonal>
PSCPreconditioner<dim, VectorType, number, same_diagonal>::PSCPreconditioner()
{}

template <int dim, typename VectorType, class number, bool same_diagonal>
template <class GlobalOperatorType>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::initialize(const GlobalOperatorType & /*global_operator*/,
									   const AdditionalData &data)
{
  Assert(data.dof_handler != 0, dealii::ExcInternalError());
  Assert(data.level != -1, dealii::ExcInternalError());
  Assert(data.mapping != 0, dealii::ExcInternalError());

  this->data = data;
  level = data.level;
  const dealii::DoFHandler<dim> &dof_handler = *(data.dof_handler);
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();

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
  dealii::UpdateFlags update_flags = dealii::update_JxW_values |
    dealii::update_quadrature_points |
    dealii::update_values |
    dealii::update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);
  info_box.initialize(fe, *(data.mapping), &(dof_handler.block_info()));
  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> (dof_handler.block_info()));

  patch_inverses.resize(ddh->global_dofs_on_subdomain.size());
  //setup local matrices/inverses
  {
    timer->enter_subsection("LO::build_matrices");
    if (same_diagonal)
      {
        real_patch_inverses.resize(1);
        const unsigned int n = fe.n_dofs_per_cell();
        real_patch_inverses[0]=dealii::FullMatrix<double>(n);
        dealii::Triangulation<dim> local_triangulation;
        dealii::DoFHandler<dim> local_dof_handler(local_triangulation);
        if (level == 0)
          dealii::GridGenerator::hyper_cube (local_triangulation,0., 1.);
        else
          dealii::GridGenerator::hyper_cube (local_triangulation,0., 2./std::pow(2., level));
        if (level != 0) local_triangulation.refine_global(1);

        local_dof_handler.distribute_dofs (fe);
        local_dof_handler.initialize_local_block_info();
        dealii::MeshWorker::IntegrationInfoBox<dim> local_info_box;
        const unsigned int local_n_gauss_points = local_dof_handler.get_fe().degree+1;
        local_info_box.initialize_gauss_quadrature(local_n_gauss_points,
                                                   local_n_gauss_points,
                                                   local_n_gauss_points);
        local_info_box.initialize_update_flags();
        dealii::UpdateFlags local_update_flags = dealii::update_quadrature_points |
	  dealii::update_values |
	  dealii::update_gradients;
        local_info_box.add_update_flags(local_update_flags, true, true, true, true);
        local_info_box.initialize(fe, *(data.mapping), &(local_dof_handler.block_info()));
        dealii::MeshWorker::DoFInfo<dim> local_dof_info(local_dof_handler.block_info());
        dealii::FullMatrix<double> dummy_matrix(local_dof_handler.n_dofs(),local_dof_handler.n_dofs());
        dealii::MeshWorker::Assembler::MatrixSimple<dealii::FullMatrix<double> > local_assembler;
        local_assembler.initialize(dummy_matrix);
        MatrixIntegrator<dim,false> local_integrator ;
        dealii::MeshWorker::integration_loop<dim, dim>
	  (local_dof_handler.begin_active(),
	   local_dof_handler.end(),
	   local_dof_info, local_info_box,
	   local_integrator,local_assembler);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              real_patch_inverses[0](i, j) = dummy_matrix(i, j);
            }
        //assign to the smoother
        for (unsigned int i=0; i<ddh->subdomain_to_global_map.size(); ++i)
          patch_inverses[i] = &real_patch_inverses[0];
      }
    else
      {
        real_patch_inverses.resize(ddh->subdomain_to_global_map.size());
        for (unsigned int i=0; i<=ddh->subdomain_to_global_map.size()/dealii::MultithreadInfo::n_threads(); ++i)
	  {
	    dealii::Threads::ThreadGroup<> threads;
	    for (unsigned int j=i*dealii::MultithreadInfo::n_threads(); 
		 (j<(i+1)*dealii::MultithreadInfo::n_threads())and(j<ddh->subdomain_to_global_map.size()) ; ++j)
	      {
		threads += dealii::Threads::new_thread([j,this](){
		    build_matrix(ddh->subdomain_to_global_map[j],
				 ddh->global_dofs_on_subdomain[j],
				 ddh->all_to_unique[j],
				 real_patch_inverses[j]);
		  });
		patch_inverses[j] = &real_patch_inverses[j];
	      }
	    threads.join_all ();
	  }
      }
    timer->leave_subsection();
  }
  for (unsigned int i=0; i<=real_patch_inverses.size()/dealii::MultithreadInfo::n_threads(); ++i)
    {
      dealii::Threads::ThreadGroup<> threads;
      for (unsigned int j=i*dealii::MultithreadInfo::n_threads(); 
  	   (j<(i+1)*dealii::MultithreadInfo::n_threads())and(j<real_patch_inverses.size()) ; ++j)
  	{
  	  threads += dealii::Threads::new_thread([j,this](){ 
  	      real_patch_inverses[j].gauss_jordan();});
  	}
      threads.join_all ();
    } 
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::clear()
{}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::vmult (VectorType &dst,
								       const VectorType &src) const
{
  dst = 0;
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::Tvmult (VectorType &/*dst*/,
									const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::vmult_add (VectorType &dst,
									   const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);

  {
    implementation::WorkStream::Copy<dim, VectorType, number, same_diagonal> copy_sample;
    copy_sample.dst = &dst;
    copy_sample.ddh = ddh;

    implementation::WorkStream::Scratch<dim, VectorType, number, same_diagonal> scratch_sample;
    scratch_sample.src = &src;
    scratch_sample.local_inverses = &patch_inverses;

    dealii::WorkStream::run(ddh->colorized_iterators(),
                            implementation::WorkStream::work<dim, VectorType, number, same_diagonal>,
                            implementation::WorkStream::assemble<dim, VectorType, number, same_diagonal>,
                            scratch_sample, copy_sample);
  }

  dst *= data.weight;
  timer->leave_subsection();
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::Tvmult_add (VectorType &/*dst*/,
									    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename VectorType, class number, bool same_diagonal>
void PSCPreconditioner<dim, VectorType, number, same_diagonal>::build_matrix
(const std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &cell_range,
 const std::vector<dealii::types::global_dof_index> &global_dofs_on_subdomain,
 const std::map<dealii::types::global_dof_index, unsigned int> &all_to_unique,
 dealii::FullMatrix<double> &matrix)
{

  dealii::MGLevelObject<dealii::FullMatrix<double> > mg_matrix ;
  mg_matrix.resize(level,level);

  mg_matrix[level] = std::move(dealii::FullMatrix<double>(global_dofs_on_subdomain.size()));

  Assembler::MGMatrixSimpleMapped<dealii::FullMatrix<double> > assembler;
  assembler.initialize(mg_matrix);
#ifdef CG
  assembler.initialize(constraints);
#endif
  assembler.initialize(all_to_unique);

  //now assemble everything
  dealii::MeshWorker::LoopControl lctrl;
  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
  lctrl.ghost_cells = true;
  //TODO possibly colorize iterators, assume thread-safety for the moment
  std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> > colored_iterators(1, cell_range);


  dealii::colored_loop<dim, dim> (colored_iterators, *dof_info, info_box, matrix_integrator, assembler,lctrl, colored_iterators[0]);

  matrix.copy_from(mg_matrix[level]);
}

#include "PSCPreconditioner.inst"
