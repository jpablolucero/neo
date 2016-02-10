#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/l2.h>

#include <iostream>
#include <fstream>
#include <vector>


using namespace dealii;

template <int dim>
class RHSIntegrator : public MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void face(MeshWorker::DoFInfo<dim> &dinfo1,
	    MeshWorker::DoFInfo<dim> &dinfo2,
	    typename MeshWorker::IntegrationInfo<dim> &info1,
	    typename MeshWorker::IntegrationInfo<dim> &info2) const;
};
template <int dim>
void RHSIntegrator<dim>::cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
{
  const FEValuesBase<dim> &fe_v = info.fe_values();
  Vector<double> &local_vector = dinfo.vector(0).block(0);
  const std::vector<double> input_vector(local_vector.size(),1.);
  LocalIntegrators::L2::L2(local_vector,fe_v,input_vector,1.);
}
  
template <int dim>
void RHSIntegrator<dim>::boundary(MeshWorker::DoFInfo<dim> &, typename MeshWorker::IntegrationInfo<dim> &) const
{}
template <int dim>
void RHSIntegrator<dim>::face(MeshWorker::DoFInfo<dim> &,
			      MeshWorker::DoFInfo<dim> &,
			      typename MeshWorker::IntegrationInfo<dim> &,
			      typename MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
class MyLaplace
{
public:
  MyLaplace ();
  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve (Vector<double> &solution);
  void output_results () const;

  Triangulation<dim>   triangulation;
  const MappingQ1<dim> mapping;

  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       right_hand_side;

  typedef MeshWorker::DoFInfo<dim> DoFInfo;
  typedef MeshWorker::IntegrationInfo<dim> CellInfo;

  static void integrate_cell_term (DoFInfo &dinfo,
				   CellInfo &info);
  static void integrate_boundary_term (DoFInfo &dinfo,
				       CellInfo &info);
  static void integrate_face_term (DoFInfo &dinfo1,
				   DoFInfo &dinfo2,
				   CellInfo &info1,
				   CellInfo &info2);
    
  RHSIntegrator<dim>    rhs_integrator ;
    
};

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mapping (),
  fe (1),
  dof_handler (triangulation)
{}


template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  right_hand_side.reinit (dof_handler.n_dofs());
}

template <int dim>
void MyLaplace<dim>::assemble_system ()
{
  MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points |
    update_values            |
    update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping);

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  MeshWorker::Assembler::ResidualSimple<Vector<double> > rhs_assembler;
  AnyData data;
  data.add<Vector<double>*>(&right_hand_side, "RHS");
  rhs_assembler.initialize(data);

  RHSIntegrator<dim> rhs_integrator;

  MeshWorker::integration_loop<dim, dim>(dof_handler.begin_active(), dof_handler.end(),
					 dof_info, info_box,
					 rhs_integrator, rhs_assembler);

  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> >
    assembler;
  assembler.initialize(system_matrix, right_hand_side);

  MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
    (dof_handler.begin_active(), dof_handler.end(),
     dof_info, info_box,
     &MyLaplace<dim>::integrate_cell_term,
     &MyLaplace<dim>::integrate_boundary_term,
     &MyLaplace<dim>::integrate_face_term,
     assembler);
}

template <int dim>
void MyLaplace<dim>::integrate_cell_term (DoFInfo &dinfo,
						 CellInfo &info)
{
  const FEValuesBase<dim> &fe_v = info.fe_values();
  FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
  LocalIntegrators::Laplace::cell_matrix(local_matrix,fe_v) ;
}

template <int dim>
void MyLaplace<dim>::integrate_face_term (DoFInfo &dinfo1,
						 DoFInfo &dinfo2,
						 CellInfo &info1,
						 CellInfo &info2)
{
  const FEValuesBase<dim> &fe_v = info1.fe_values();
  const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();

  FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
  FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
  FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
  FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  LocalIntegrators::Laplace::ip_matrix(u1_v1_matrix,u2_v1_matrix,u1_v2_matrix,u2_v2_matrix,fe_v,fe_v_neighbor,
				       LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));

}

template <int dim>
void MyLaplace<dim>::integrate_boundary_term (DoFInfo &dinfo,
						     CellInfo &info)
{
  const FEValuesBase<dim> &fe_v = info.fe_values();
  FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  LocalIntegrators::Laplace::nitsche_matrix(local_matrix,fe_v,
					    LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}



template <int dim>
void MyLaplace<dim>::solve (Vector<double> &solution)
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  solver.solve (system_matrix, solution, right_hand_side,
		PreconditionIdentity());
}

template <int dim>
void MyLaplace<dim>::output_results () const
{
  std::string filename = "solution";

  filename += ".vtk";
  deallog << "Writing solution to <" << filename << ">" << std::endl;
  std::ofstream vtk_output (filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");

  data_out.build_patches ();

  data_out.write_vtk(vtk_output);
}


template <int dim>
void MyLaplace<dim>::run ()
{
  GridGenerator::hyper_cube (triangulation,-1.,1.);
    
  triangulation.refine_global (3);
  deallog << "Number of active cells:       "
	  << triangulation.n_active_cells()
	  << std::endl;

  setup_system ();
    
  deallog << "Number of degrees of freedom: "
	  << dof_handler.n_dofs()
	  << std::endl;

  assemble_system ();
  solve (solution);
  output_results ();
}


int main ()
{
  dealii::deallog.depth_console (2);
  MyLaplace<2> dgmethod;
  dgmethod.run ();
  return 0;
}
