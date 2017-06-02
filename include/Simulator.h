#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/algorithms/newton.h>
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
#include <Preconditioner.h>

#include <string>
#include <fstream>

template <int dim=2,bool same_diagonal = true, unsigned int fe_degree = 1>
class Simulator final
{
public:
  Simulator (dealii::TimerOutput &timer_,
             MPI_Comm &mpi_communicator_,
             dealii::ConditionalOStream &pcout_);
  ~Simulator ();
  Simulator (const Simulator &) = delete ;
  Simulator &operator = (const Simulator &) = delete;
  void run ();
  void run_non_linear ();
  unsigned int n_levels ;
  unsigned int min_level;
  unsigned int smoothing_steps ;
private:
#ifdef MATRIXFREE
  typedef MfreeOperator<dim,fe_degree,fe_degree+1,double> SystemMatrixType;
#else
  typedef MFOperator<dim,fe_degree,double> SystemMatrixType;
#endif // MATRIXFREE

  void setup_system ();
  void solve ();
  void compute_error () const;
  void output_results (const unsigned int cycle) const;

  MPI_Comm                   &mpi_communicator;

  Mesh<dim>                                        mesh;
  FiniteElement<dim>                               fe;
  Dofs<dim>                                        dofs;
  RHS<dim>                                         rhs;
#ifdef MG   
  Preconditioner<dim,same_diagonal,fe_degree>      preconditioner;
#else // MG OFF
  dealii::PreconditionIdentity                     preconditioner;
#endif // MG

  SystemMatrixType      system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_tmp;

  dealii::ConditionalOStream &pcout;
  dealii::TimerOutput &timer;

  friend class Residual;
  class Residual : public dealii::Algorithms::OperatorBase
  {
  public:
    Residual(Simulator<dim,same_diagonal,fe_degree> &sim_):sim(sim_) {} ;
    void operator() (dealii::AnyData &out, const dealii::AnyData &in) override
    {
      sim.setup_system();
      sim.solution = *(in.try_read_ptr<LA::MPI::Vector>("Newton iterate"));
      sim.rhs.assemble(sim.solution);
      *out.entry<LA::MPI::Vector *>(0) = sim.rhs.right_hand_side ;
    }
    Simulator<dim,same_diagonal,fe_degree> &sim ;
  } residual ;

  friend class InverseDerivative ;
  class InverseDerivative : public dealii::Algorithms::OperatorBase
  {
  public:
    InverseDerivative(Simulator<dim,same_diagonal,fe_degree> &sim_):sim(sim_) {} ;
    void operator() (dealii::AnyData &out, const dealii::AnyData &in) override
    {
      sim.setup_system();
      sim.solution = *(in.try_read_ptr<LA::MPI::Vector>("Newton iterate"));
      sim.rhs.right_hand_side = *(in.try_read_ptr<LA::MPI::Vector>("Newton residual"));
#ifdef MG           
      sim.preconditioner.setup(sim.solution);
#endif // MG                 
      sim.solve ();
      *out.entry<LA::MPI::Vector *>(0) = sim.solution ;
    }
    Simulator<dim,same_diagonal,fe_degree> &sim ;
  } inverse ;

  dealii::Algorithms::Newton<LA::MPI::Vector> newton;

};

#ifdef HEADER_IMPLEMENTATION
#include <Simulator.cc>
#endif

#endif // SIMULATOR_H
