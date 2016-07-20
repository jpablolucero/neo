#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition_block.templates.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <MFOperator.h>
#include <EquationData.h>
#include <ResidualSimpleConstraints.h>
#include <PSCPreconditioner.h>
#include <MFPSCPreconditioner.h>

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
  unsigned int n_levels ;
  unsigned int smoothing_steps ;
private:
  void setup_system ();
  void setup_multigrid ();
  void assemble_system ();
  void solve ();
  void compute_error () const;
  void output_results (const unsigned int cycle) const;

  // TODO Simulator needs interface(template) to define n_q_points_1d & number
  typedef MFOperator<dim,fe_degree,fe_degree+1,double> SystemMatrixType;

  dealii::IndexSet           locally_owned_dofs;
  dealii::IndexSet           locally_relevant_dofs;
  MPI_Comm                   &mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim>   triangulation;
  const dealii::MappingQ1<dim>                        mapping;
  dealii::ConstraintMatrix                            constraints;
  dealii::FESystem<dim>                               fe;
  ReferenceFunction<dim>                              reference_function;

  dealii::DoFHandler<dim>      dof_handler;

  SystemMatrixType             system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       solution_tmp;
  LA::MPI::Vector       right_hand_side;

  dealii::MGLevelObject<SystemMatrixType >            mg_matrix ;

  dealii::ConditionalOStream &pcout;

  dealii::TimerOutput &timer;
};

// Is it possible to use this or something similar for Trilinos
namespace dealii
{
  template <int dim, typename LOPERATOR>
  class MGTransferMF : public dealii::MGTransferMatrixFree<dim, typename LOPERATOR::value_type>
  {
  public:
    MGTransferMF(const MGLevelObject<LOPERATOR> &op)
      :
      mg_operator (op)
    {};

    // Analogous to 'copy_to_mg' from MGTransferPrebuilt ???
    template <class InVector, int spacedim>
    void
    copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof,
                MGLevelObject<dealii::parallel::distributed::Vector<typename LOPERATOR::value_type> > &dst,
                const InVector &src) const
    {
      for (unsigned int level=dst.min_level();
           level<=dst.max_level(); ++level)
        mg_operator[level].initialize_dof_vector(dst[level]);
      dealii::MGLevelGlobalTransfer
      <dealii::parallel::distributed::Vector<typename LOPERATOR::value_type> >::copy_to_mg(mg_dof, dst, src);
    }

  private:
    const MGLevelObject<LOPERATOR> &mg_operator;
  };
}

#ifdef HEADER_IMPLEMENTATION
#include <Simulator.cc>
#endif

#endif // SIMULATOR_H
