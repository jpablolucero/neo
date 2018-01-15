#include <NLPSCPreconditioner.h>

extern std::unique_ptr<dealii::TimerOutput>        timer ;
extern std::unique_ptr<MPI_Comm>                   mpi_communicator ;

namespace
{
  namespace NLWorkStream
  {
    template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
    class Copy
    {
    public:
      dealii::Vector<number>  local_solution;
      unsigned int subdomain_idx;
      VectorType *dst;
      std::shared_ptr<DDHandlerBase<dim> > ddh;
    };

    template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
    class Scratch
    {
    public:
      SystemMatrixType* system_matrix ;
      dealii::MGLevelObject<VectorType> *solution;
      const dealii::Mapping<dim>  *mapping;
    };

    template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
    void assemble(const Copy<dim, SystemMatrixType, VectorType, number, same_diagonal> &copy)
    {
      copy.ddh->prolongate_add(*(copy.dst),copy.local_solution,copy.subdomain_idx);
    }

    template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
    void work(const std::vector<unsigned int>::const_iterator &iterator,
              Scratch<dim, SystemMatrixType, VectorType, number, same_diagonal> &scratch,
	      Copy<dim, SystemMatrixType, VectorType, number, same_diagonal> &copy)
    {
      const unsigned int subdomain_idx = *iterator;
      const DDHandlerBase<dim> &ddh = *(copy.ddh);
      copy.subdomain_idx = subdomain_idx;
      ddh.reinit(copy.local_solution, subdomain_idx);
      dealii::MeshWorker::DoFInfo<dim> dof_info (ddh.get_dofh().block_info());
      dealii::MeshWorker::IntegrationInfoBox<dim> info_box ;
      const unsigned int n_gauss_points = ddh.get_dofh().get_fe().degree+1;
      info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);
      info_box.initialize_update_flags();
      const dealii::UpdateFlags update_flags_cell = dealii::update_JxW_values | dealii::update_quadrature_points |
	dealii::update_values | dealii::update_gradients;
      const dealii::UpdateFlags update_flags_face = dealii::update_JxW_values | dealii::update_quadrature_points |
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
      src_data.add<const dealii::MGLevelObject<VectorType >*>(scratch.solution,"src");
      src_data.add<const dealii::MGLevelObject<VectorType >*>(scratch.solution,"Newton iterate");
      info_box.initialize(ddh.get_dofh().get_fe(), *(scratch.mapping), src_data, VectorType {}, &(ddh.get_dofh().block_info()));
      std::vector<std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> >
	colored_iterators(1,copy.ddh->subdomain_to_global_map[subdomain_idx]);
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
	  dealii::ReductionControl solver_control (res.size()*10, 1.e-20, 1.E-5,false,false);
	  typename dealii::SolverGMRES<dealii::Vector<double> >::AdditionalData data(100,false);
	  dealii::SolverGMRES<dealii::Vector<double> >                          solver (solver_control,data);
	  scratch.system_matrix->set_cell_range(ddh.subdomain_to_global_map[subdomain_idx]);
	  scratch.system_matrix->set_subdomain(subdomain_idx);
	  Du = 0.;
	  solver.solve(*(scratch.system_matrix),Du,res,dealii::PreconditionIdentity());
	  scratch.system_matrix->unset_cell_range();
	};
      auto residual = [&](dealii::Vector<number> & res,dealii::Vector<number> & u)
	{
	  mg_vector[ddh.get_level()] = 0. ;
	  dealii::MeshWorker::LoopControl lctrl;
	  lctrl.faces_to_ghost = dealii::MeshWorker::LoopControl::both;
	  lctrl.own_faces = dealii::MeshWorker::LoopControl::both;
	  dealii::colored_loop<dim, dim> (colored_iterators,dof_info,info_box,rhs_integrator,rassembler,lctrl,colored_iterators[0]);
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
	dealii::DoFTools::extract_locally_relevant_level_dofs(ddh.get_dofh(), ddh.get_level(), locally_relevant_level_dofs);
	residual(res,u);
      	resnorm = res.l2_norm();
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
    ghosted_solution.resize(level, level);
    ghosted_solution[level].reinit(locally_owned_level_dofs,locally_relevant_level_dofs,*mpi_communicator);
  }
  if (data.patch_type == AdditionalData::PatchType::cell_patches)
    ddh.reset(new DGDDHandlerCell<dim>());
  else
    ddh.reset(new DGDDHandlerVertex<dim>());
  ddh->initialize(dof_handler, level);
  dof_info.reset(new dealii::MeshWorker::DoFInfo<dim> (dof_handler.block_info()));
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult (VectorType &dst,
    const VectorType &src) const
{
  ghosted_solution[level] = dst ;
  vmult_add(dst, src);
  dst *= data.relaxation;
  AssertIsFinite(dst.l2_norm());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::vmult_add (VectorType &dst,
    const VectorType &src) const
{
  std::string section = "Smoothing @ level ";
  section += std::to_string(level);
  timer->enter_subsection(section);
  SystemMatrixType internal_system_matrix ;
  internal_system_matrix.reinit (data.dof_handler,data.mapping,nullptr,level,ghosted_solution[level]);
  internal_system_matrix.clear();
  if (data.smoother_type == AdditionalData::SmootherType::additive)
    {
      NLWorkStream::Copy<dim, SystemMatrixType, VectorType, number, same_diagonal> copy_sample;
      copy_sample.dst = &dst;
      copy_sample.ddh = ddh;
      NLWorkStream::Scratch<dim, SystemMatrixType, VectorType, number, same_diagonal> scratch_sample;
      scratch_sample.solution = &ghosted_solution;
      scratch_sample.mapping = data.mapping ;
      scratch_sample.system_matrix = &internal_system_matrix;
      dealii::WorkStream::run(ddh->colorized_iterators(),
			      NLWorkStream::work<dim, SystemMatrixType, VectorType, number, same_diagonal>,
			      NLWorkStream::assemble<dim, SystemMatrixType, VectorType, number, same_diagonal>,
			      scratch_sample, copy_sample);
    }
  else AssertThrow(false, dealii::ExcNotImplemented());
  timer->leave_subsection();
}

template <int dim, typename SystemMatrixType, typename VectorType, typename number, bool same_diagonal>
void NLPSCPreconditioner<dim, SystemMatrixType, VectorType, number, same_diagonal>::Tvmult_add (VectorType &/*dst*/,
    const VectorType &/*src*/) const
{
  AssertThrow(false, dealii::ExcNotImplemented());
}

