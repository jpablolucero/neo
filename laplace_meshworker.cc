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
class MatrixIntegrator : public MeshWorker::LocalIntegrator<dim>
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
void MatrixIntegrator<dim>::cell(MeshWorker::DoFInfo<dim> &dinfo, 
				 typename MeshWorker::IntegrationInfo<dim> &info) const
{
  const FEValuesBase<dim> &fe = info.fe_values() ;
  Vector<double> &dst = dinfo.vector(0).block(0) ;
  const std::vector<Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  LocalIntegrators::Laplace::cell_residual(dst,fe,Dsrc) ;
}
  
template <int dim>
void MatrixIntegrator<dim>::face(MeshWorker::DoFInfo<dim> &dinfo1,
				 MeshWorker::DoFInfo<dim> &dinfo2,
				 typename MeshWorker::IntegrationInfo<dim> &info1,
				 typename MeshWorker::IntegrationInfo<dim> &info2) const
{
  const FEValuesBase<dim> &fe1 = info1.fe_values();
  const FEValuesBase<dim> &fe2 = info2.fe_values();

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  const std::vector<double> &src1 = info1.values[0][0];
  const std::vector<Tensor<1,dim> > &Dsrc1 = info1.gradients[0][0];
  Vector<double> &dst1 = dinfo1.vector(0).block(0) ;

  const std::vector<double> &src2 = info2.values[0][0];
  const std::vector<Tensor<1,dim> > &Dsrc2 = info2.gradients[0][0];
  Vector<double> &dst2 = dinfo2.vector(0).block(0) ;

  LocalIntegrators::Laplace::ip_residual(dst1,dst2,
					 fe1,fe2,
					 src1,Dsrc1,
					 src2,Dsrc2,
					 LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));

}

template <int dim>
void MatrixIntegrator<dim>::boundary(MeshWorker::DoFInfo<dim> &dinfo, 
				     typename MeshWorker::IntegrationInfo<dim> &info) const
{
  const FEValuesBase<dim> &fe = info.fe_values();
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const std::vector<double> &src = info.values[0][0];
  const std::vector<double> data(src.size());
  const std::vector<Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  Vector<double> &dst = dinfo.vector(0).block(0) ;

  LocalIntegrators::Laplace::nitsche_residual(dst,
					      fe,
					      src,
					      Dsrc,
					      data,
					      LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));

}

template <int dim, int fe_degree, typename number>
class LaplaceOperator : public Subscriptor
{
public:
  LaplaceOperator (const Triangulation<dim>& triangulation_,
		   const MappingQ1<dim>&  mapping_,
		   const FE_DGQ<dim>&  fe_,
		   const DoFHandler<dim>&  dof_handler_); 

  void vmult (Vector<number> &dst,
              const Vector<number> &src) const;
  void Tvmult (Vector<number> &dst,
               const Vector<number> &src) const;
  void vmult_add (Vector<number> &dst,
                  const Vector<number> &src) const;
  void Tvmult_add (Vector<number> &dst,
                   const Vector<number> &src) const;

 private:
  const Triangulation<dim>& triangulation;
  const MappingQ1<dim>&  mapping;
  const FE_DGQ<dim>&  fe;
  const DoFHandler<dim>&  dof_handler; 
};

template<int dim, int fe_degree, typename number>
LaplaceOperator<dim, fe_degree, number>::LaplaceOperator(const Triangulation<dim>& triangulation_,
							 const MappingQ1<dim>&  mapping_,
							 const FE_DGQ<dim>&  fe_,
							 const DoFHandler<dim>&  dof_handler_) : 
  triangulation(triangulation_),
  mapping(mapping_),
  fe(fe_),
  dof_handler(dof_handler_)
{}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::vmult (Vector<number> &dst,
	    const Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::Tvmult (Vector<number> &dst,
	     const Vector<number> &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::vmult_add (Vector<number> &dst,
						       const Vector<number> &src) const
{
  MeshWorker::IntegrationInfoBox<dim> info_box;

  const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
  info_box.initialize_gauss_quadrature(n_gauss_points,
				       n_gauss_points,
				       n_gauss_points);

  AnyData src_data ;
  src_data.add<const Vector<double>*>(&src,"src");

  info_box.cell_selector.add("src", true, true, false);
  info_box.boundary_selector.add("src", true, true, false);
  info_box.face_selector.add("src", true, true, false);


  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points |
    update_values            |
    update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping, src_data, src);

  MeshWorker::DoFInfo<dim> dof_info(dof_handler);

  MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;
  AnyData dst_data;
  dst_data.add<Vector<double>*>(&dst, "dst");
  assembler.initialize(dst_data);

  MatrixIntegrator<dim> matrix_integrator ;

  MeshWorker::integration_loop<dim, dim>
    (dof_handler.begin_active(), dof_handler.end(),
     dof_info, info_box,
     matrix_integrator,
     assembler);

}

template <int dim, int fe_degree, typename number>
void LaplaceOperator<dim,fe_degree,number>::Tvmult_add (Vector<number> &dst,
		 const Vector<number> &src) const
{
  vmult_add(dst, src);
}

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

  typedef LaplaceOperator<dim,1,double> SystemMatrixType;

  Triangulation<dim>   triangulation;
  const MappingQ1<dim> mapping;

  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof_handler;

  SystemMatrixType     system_matrix;

  Vector<double>       solution;
  Vector<double>       right_hand_side;

  RHSIntegrator<dim>    rhs_integrator ;
    
};

template <int dim>
MyLaplace<dim>::MyLaplace ()
  :
  mapping (),
  fe (1),
  dof_handler (triangulation),
  system_matrix ( triangulation,
		  mapping,
		  fe,
		  dof_handler) 
{}


template <int dim>
void MyLaplace<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
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
    update_values;            
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

}

template <int dim>
void MyLaplace<dim>::solve (Vector<double> &solution)
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  solver_control.log_history(true);
  solver.solve (system_matrix, solution, right_hand_side,
  		PreconditionIdentity());
}

template <int dim>
void MyLaplace<dim>::output_results () const
{
  std::string filename = "solution";

  filename += ".gnuplot";
  deallog << "Writing solution to <" << filename << ">" << std::endl;
  std::ofstream gnuplot_output (filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "u");

  data_out.build_patches ();

  data_out.write_gnuplot(gnuplot_output);
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
