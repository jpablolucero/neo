#include <Integrators.h>

template <int dim,bool same_diagonal>
MatrixIntegrator<dim,same_diagonal>::MatrixIntegrator()
{}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values() ;
  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  LocalIntegrators::Diffusion::cell_matrix<dim, Coefficient<dim> >(M,fe) ;
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                               dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fe1 = info1.fe_values();
  const dealii::FEValuesBase<dim> &fe2 = info2.fe_values();

  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(0,false).matrix;
  dealii::FullMatrix<double> M21(dinfo1.matrix(0,true).matrix.n());// = dinfo1.matrix(0,true).matrix;
  dealii::FullMatrix<double> M12(dinfo2.matrix(0,true).matrix.n());// = dinfo2.matrix(0,true).matrix;
  dealii::FullMatrix<double> &RM22 = dinfo2.matrix(0,false).matrix;
  dealii::FullMatrix<double> M22(dinfo2.matrix(0,false).matrix.n());// = dinfo2.matrix(0,false).matrix;

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  if (same_diagonal)
    {
      LocalIntegrators::Diffusion::ip_matrix<dim,Coefficient<dim> >(RM11,M21,M12,M22,fe1,fe2,
						   dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
  else
    {
      LocalIntegrators::Diffusion::ip_matrix<dim,Coefficient<dim> >(RM11,M21,M12,RM22,fe1,fe2,
						   dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();

  LocalIntegrators::Diffusion::nitsche_matrix<dim,Coefficient<dim> >
    (M,
     fe,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator()
{}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values() ;
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;
  const std::vector<dealii::Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  LocalIntegrators::Diffusion::cell_residual<dim,Coefficient<dim> >(dst, fe, Dsrc) ;
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
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

  LocalIntegrators::Diffusion::ip_residual<dim,Coefficient<dim> >
    (dst1,dst2,
     fe1,fe2,
     src1,Dsrc1,
     src2,Dsrc2,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const std::vector<double> &src = info.values[0][0];
  std::vector<double> data{};
  data.resize(src.size());
  const std::vector<dealii::Tensor<1,dim> > &Dsrc = info.gradients[0][0];
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;

  LocalIntegrators::Diffusion::nitsche_residual<dim,Coefficient<dim> >
    (dst, 
     fe,
     src,
     Dsrc,
     data,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

template class MatrixIntegrator<2,false>;
template class MatrixIntegrator<3,false>;
template class MatrixIntegrator<2,true>;
template class MatrixIntegrator<3,true>;
template class ResidualIntegrator<2>;
template class ResidualIntegrator<3>;
