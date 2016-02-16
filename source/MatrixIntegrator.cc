#include <MatrixIntegrator.h>

template <int dim>
void MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				 typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values() ;
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;
  const std::vector<dealii::Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  dealii::LocalIntegrators::Laplace::cell_residual(dst,fe,Dsrc) ;
}
  
template <int dim>
void MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
				 dealii::MeshWorker::DoFInfo<dim> &dinfo2,
				 typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
				 typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fe1 = info1.fe_values();
  const dealii::FEValuesBase<dim> &fe2 = info2.fe_values();

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  const std::vector<double> &src1 = info1.values[0][0];
  const std::vector<dealii::Tensor<1,dim> > &Dsrc1 = info1.gradients[0][0];
  dealii::Vector<double> &dst1 = dinfo1.vector(0).block(0) ;

  const std::vector<double> &src2 = info2.values[0][0];
  const std::vector<dealii::Tensor<1,dim> > &Dsrc2 = info2.gradients[0][0];
  dealii::Vector<double> &dst2 = dinfo2.vector(0).block(0) ;

  dealii::LocalIntegrators::Laplace::ip_residual(dst1,dst2,
						 fe1,fe2,
						 src1,Dsrc1,
						 src2,Dsrc2,
						 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));

}

template <int dim>
void MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				     typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const std::vector<double> &src = info.values[0][0];
  const std::vector<double> data(src.size());
  const std::vector<dealii::Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;

  dealii::LocalIntegrators::Laplace::nitsche_residual(dst,
						      fe,
						      src,
						      Dsrc,
						      data,
						      dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));

}

template class MatrixIntegrator<2>;
