#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/algorithms/newton.h>
#include <deal.II/algorithms/newton.templates.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/data_out.h>

#include <MFOperator.h>
#include <EquationData.h>
#include <ResidualSimpleConstraints.h>
#include <PSCPreconditioner.h>
#include <Mesh.h>
#include <FiniteElement.h>
#include <Dofs.h>
#include <RHS.h>
#include <GMGPreconditioner.h>
#include <NLPSCPreconditioner.h>

#include <string>
#include <fstream>

template <typename SystemMatrixType,typename VectorType,typename Preconditioner,int dim=2,unsigned int fe_degree = 1>
class Simulator final
{
public:
  Simulator ();
  ~Simulator ();
  Simulator (const Simulator &) = delete ;
  Simulator &operator = (const Simulator &) = delete;
  void run ();
  void run_non_linear ();
  unsigned int n_levels ;
  unsigned int min_level;
  unsigned int smoothing_steps ;
private:
  void setup_system ();
  void solve ();
  void compute_error () const;
  void output_results (const unsigned int cycle) const;

  Mesh<dim>             mesh;
  FiniteElement<dim>    fe;
  Dofs<dim>             dofs;
  RHS<dim>              rhs;
  Preconditioner        preconditioner;
  SystemMatrixType      system_matrix;
  
  VectorType       ghosted_solution;
  VectorType       solution;


  friend class Residual;
  template <typename SystemMatrixType_,typename VectorType_,typename Preconditioner_>
  class Residual : public dealii::Algorithms::OperatorBase
  {
  public:
    Residual(Simulator<SystemMatrixType_,VectorType_,Preconditioner_,dim,fe_degree> &sim_):sim(sim_) {} ;
    void operator() (dealii::AnyData &out, const dealii::AnyData &in) override
    {
      sim.ghosted_solution = *(in.try_read_ptr<VectorType_>("Newton iterate"));
      sim.rhs.assemble(sim.ghosted_solution);
      dealii::deallog << "Residual: " << sim.rhs.right_hand_side.l2_norm() << std::endl ;
      *out.entry<VectorType_ *>(0) = sim.rhs.right_hand_side ;
    }
    Simulator<SystemMatrixType_,VectorType_,Preconditioner_,dim,fe_degree> &sim ;
  };
  Residual<SystemMatrixType,VectorType,Preconditioner> residual ;

  friend class InverseDerivative ;
  template <typename SystemMatrixType_,typename VectorType_,typename Preconditioner_>
  class InverseDerivative : public dealii::Algorithms::OperatorBase
  {
  public:
    InverseDerivative(Simulator<SystemMatrixType_,VectorType_,Preconditioner_,dim,fe_degree> &sim_):sim(sim_) {} ;
    void operator() (dealii::AnyData &out, const dealii::AnyData &in) override
    {
      sim.ghosted_solution = *(in.try_read_ptr<VectorType_>("Newton iterate"));
      sim.ghosted_solution.update_ghost_values();
#ifdef ASPIN
	{
	  typename NLPSCPreconditioner<dim, SystemMatrixType_, VectorType_, double, false>::AdditionalData data ;
	  data.dof_handler = &(sim.dofs.dof_handler);
	  data.level = sim.n_levels-1;
	  data.n_levels = sim.n_levels ;
	  data.mapping = &(sim.fe.mapping);
	  data.relaxation = 1.;
	  data.patch_type = NLPSCPreconditioner<dim, SystemMatrixType_, VectorType_, double, false>::AdditionalData::cell_patches;
	  data.smoother_type = NLPSCPreconditioner<dim, SystemMatrixType_, VectorType_, double, false>::AdditionalData::additive;
	  NLPSCPreconditioner<dim, SystemMatrixType_, VectorType_, double, false> prec ;
	  prec.initialize(sim.system_matrix,data);
	  prec.vmult(sim.ghosted_solution,*(in.try_read_ptr<VectorType_>("Newton residual")));
	  *out.entry<VectorType_ *>(0) = sim.ghosted_solution ;
	  sim.ghosted_solution = *(in.try_read_ptr<VectorType_>("Newton iterate"));
	  sim.ghosted_solution.add(-1.,*out.entry<VectorType_ *>(0));
	  sim.rhs.assemble(sim.ghosted_solution);
	  dealii::deallog << "ASPIN Residual: " << sim.rhs.right_hand_side.l2_norm() << std::endl ;
	  *const_cast<VectorType_*>(in.try_read_ptr<VectorType_>("Newton iterate")) = sim.ghosted_solution;
	}
#endif
      sim.solution = 0.;
      sim.solve ();
      *out.entry<VectorType_ *>(0) = sim.ghosted_solution ;
    }
    Simulator<SystemMatrixType_,VectorType_,Preconditioner_,dim,fe_degree> &sim ;
  };
  InverseDerivative<SystemMatrixType,VectorType,Preconditioner> inverse ;

  dealii::Algorithms::Newton<VectorType> newton;

};

// #ifdef HEADER_IMPLEMENTATION
#include <Simulator.templates.h>
// #endif

#endif // SIMULATOR_H
