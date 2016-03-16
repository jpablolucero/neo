#include <BlockIntegrators.h>

// MATRIX INTEGRATOR
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
  dealii::FullMatrix<double> &RM22 = dinfo2.matrix(0,false).matrix;
  dealii::FullMatrix<double> M21 = dinfo1.matrix(0,true).matrix;
  dealii::FullMatrix<double> M12 = dinfo2.matrix(0,true).matrix;
  dealii::FullMatrix<double> M22 = dinfo2.matrix(0,false).matrix;

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  if (same_diagonal)
    {
      LocalIntegrators::Diffusion::ip_matrix<dim,Coefficient<dim> >
	(RM11,M21,M12,M22,fe1,fe2,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
  else
    {
      LocalIntegrators::Diffusion::ip_matrix<dim,Coefficient<dim> >
	(RM11,M21,M12,RM22,fe1,fe2,
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

// RESIDUAL INTEGRATOR
template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator()
{}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values() ;
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];  
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

  dealii::Vector<double> &dst1 = dinfo1.vector(0).block(0) ;
  dealii::Vector<double> &dst2 = dinfo2.vector(0).block(0) ;

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1_compvec = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2_compvec = info2.gradients[0];
  const std::vector<std::vector<double> > &src1_compvec = info1.values[0];
  const std::vector<std::vector<double> > &src2_compvec = info2.values[0];

  LocalIntegrators::Diffusion::ip_residual<dim,Coefficient<dim> >
    (dst1,dst2,
     fe1,fe2,
     src1_compvec,Dsrc1_compvec,
     src2_compvec,Dsrc2_compvec,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  dealii::Vector<double> &dst = dinfo.vector(0).block(0) ;

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc_compvec = info.gradients[0];
  const std::vector<std::vector<double> > &src_compvec = info.values[0];

  const unsigned int n_comps = fe.get_fe().n_components() ;
  std::vector<double> data_comp{};
  data_comp.resize(src_compvec[0].size(),0.0);
  const std::vector<std::vector<double> > data_compvec{n_comps,data_comp};

  LocalIntegrators::Diffusion::nitsche_residual<dim,Coefficient<dim> >
    (dst, 
     fe,
     src_compvec,
     Dsrc_compvec,
     data_compvec,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

// RHS INTEGRATOR
template <int dim>
RHSIntegrator<dim>::RHSIntegrator()
{}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fe = info.fe_values();
  dealii::Vector<double> &local_vector = dinfo.vector(0).block(0);

  const unsigned int n_blocks = fe.get_fe().n_blocks();
  Assert(n_blocks == fe.get_fe().n_components(), dealii::ExcDimensionMismatch(n_blocks, fe.get_fe().n_components()));

  std::vector<std::vector<double> > rhs_values{n_blocks};
  for(unsigned int b=0; b<n_blocks; ++b)
    rhs_values[b].resize(fe.get_fe().block_indices().block_size(b),1.0);
  dealii::LocalIntegrators::L2::L2(local_vector,fe,rhs_values) ;
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &, typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template class MatrixIntegrator<2,false>;
template class MatrixIntegrator<3,false>;
template class MatrixIntegrator<2,true>;
template class MatrixIntegrator<3,true>;
template class ResidualIntegrator<2>;
template class ResidualIntegrator<3>;
template class RHSIntegrator<2>;
template class RHSIntegrator<3>;
