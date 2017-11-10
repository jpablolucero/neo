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
#include <MfreeOperator.h>
#include <EquationData.h>
#include <ResidualSimpleConstraints.h>
#include <PSCPreconditioner.h>
#include <MFPSCPreconditioner.h>
#include <Mesh.h>
#include <FiniteElement.h>
#include <Dofs.h>
#include <RHS.h>
#include <GMGPreconditioner.h>

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
  
  VectorType       solution;
  VectorType       solution_tmp;

  friend class Residual;
  template <typename SystemMatrixType_,typename VectorType_,typename Preconditioner_>
  class Residual : public dealii::Algorithms::OperatorBase
  {
  public:
    Residual(Simulator<SystemMatrixType_,VectorType_,Preconditioner_,dim,fe_degree> &sim_):sim(sim_) {} ;
    void operator() (dealii::AnyData &out, const dealii::AnyData &in) override
    {
      sim.solution = *(in.try_read_ptr<VectorType_>("Newton iterate"));
      sim.rhs.assemble(sim.solution);
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
    void operator() (dealii::AnyData &out, const dealii::AnyData &) override
    {
      sim.solve ();
      *out.entry<VectorType_ *>(0) = sim.solution ;
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
