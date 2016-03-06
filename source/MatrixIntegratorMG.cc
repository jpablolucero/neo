#include <MatrixIntegratorMG.h>

template <int dim>
MatrixIntegratorMG<dim>::MatrixIntegratorMG()
{}

template <int dim>
void MatrixIntegratorMG<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values() ;
  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  
  dealii::LocalIntegrators::Laplace::cell_matrix(M,fe) ;
}
  
template <int dim>
void MatrixIntegratorMG<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
				   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fe1 = info1.fe_values();
  const dealii::FEValuesBase<dim> &fe2 = info2.fe_values();

  dealii::FullMatrix<double> &M11 = dinfo1.matrix(0,false).matrix;
  dealii::FullMatrix<double> &M21 = dinfo1.matrix(0,true).matrix;
  dealii::FullMatrix<double> &M12 = dinfo2.matrix(0,true).matrix;
  dealii::FullMatrix<double> &M22 = dinfo2.matrix(0,false).matrix;

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  dealii::LocalIntegrators::Laplace::ip_matrix(M11,M21,M12,M22,fe1,fe2,
					       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
}


template <int dim>
void MatrixIntegratorMG<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
   
  dealii::LocalIntegrators::Laplace::nitsche_matrix(M,fe,
						    dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

template class MatrixIntegratorMG<2>;
template class MatrixIntegratorMG<3>;
