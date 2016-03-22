#include <CompIntegrators.h>

// MATRIX INTEGRATOR
template <int dim,bool same_diagonal>
CMatrixIntegrator<dim,same_diagonal>::CMatrixIntegrator()
{}

template <int dim,bool same_diagonal>
void CMatrixIntegrator<dim,same_diagonal>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0) ;

  const unsigned int n_comps = fev.get_fe().n_components();
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs[d],d);
    }
  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs) ;
}

template <int dim,bool same_diagonal>
void CMatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
					       dealii::MeshWorker::DoFInfo<dim> &dinfo2,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();
  dealii::FullMatrix<double> Mtrash1(dinfo1.matrix(0).matrix.m());
  dealii::FullMatrix<double> Mtrash2(dinfo2.matrix(0).matrix.m());

  const unsigned int n_comps = fev1.get_fe().n_components();
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev1.n_quadrature_points);
      diffcoeff.value_list(fev1.get_quadrature_points(),coeffs[d],d);
    }

  if (same_diagonal)
    {
      dealii::FullMatrix<double> &RM11 = dinfo1.matrix(0,false).matrix;
      LocalIntegrators::Diffusion::ip_matrix<dim>
	(RM11,Mtrash1,Mtrash2,Mtrash2,fev1,fev2,coeffs,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
  else
    {
      AssertDimension(dinfo1.matrix(0).matrix.m(), fev1.dofs_per_cell);
      AssertDimension(dinfo1.matrix(0).matrix.n(), fev1.dofs_per_cell);
      AssertDimension(dinfo2.matrix(0).matrix.m(), fev2.dofs_per_cell);
      AssertDimension(dinfo2.matrix(0).matrix.n(), fev2.dofs_per_cell);
      
      dealii::FullMatrix<double> &RM11 = dinfo1.matrix(0,false).matrix;
      dealii::FullMatrix<double> &RM22 = dinfo2.matrix(0,false).matrix;
      LocalIntegrators::Diffusion::ip_matrix<dim>
	(RM11,Mtrash1,Mtrash2,RM22,fev1,fev2,coeffs,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}

template <int dim,bool same_diagonal>
void CMatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
						   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();

  AssertDimension(dinfo.matrix(0).matrix.m(), fev.dofs_per_cell);
  AssertDimension(dinfo.matrix(0).matrix.n(), fev.dofs_per_cell);

  const unsigned int n_comps = fev.get_fe().n_components();
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs[d],d);
    }

  dealii::FullMatrix<double> &M = dinfo.matrix(0).matrix;
  LocalIntegrators::Diffusion::nitsche_matrix<dim>
    (M,fev,coeffs,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

// RESIDUAL INTEGRATOR
template <int dim>
CResidualIntegrator<dim>::CResidualIntegrator()
{}

template <int dim>
void CResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::Vector<double> &localdst = dinfo.vector(0).block(0);
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];

  const unsigned int n_comps = fev.get_fe().n_components();
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs[d],d);
    }
  
  LocalIntegrators::Diffusion::cell_residual<dim>(localdst, fev, Dsrc, coeffs) ;
}
  
template <int dim>
void CResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
				   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  dealii::BlockVector<double> &localdst1 = dinfo1.vector(0);
  dealii::BlockVector<double> &localdst2 = dinfo2.vector(0);
  const unsigned int deg1 = fev1.get_fe().tensor_degree();
  const unsigned int deg2 = fev2.get_fe().tensor_degree();

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1 = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2 = info2.gradients[0];
  const std::vector<std::vector<double> > &src1 = info1.values[0];
  const std::vector<std::vector<double> > &src2 = info2.values[0];

  const unsigned int n_comps = fev1.get_fe().n_components();
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev1.n_quadrature_points);
      diffcoeff.value_list(fev1.get_quadrature_points(),coeffs[d],d);
    }

  LocalIntegrators::Diffusion::ip_residual<dim>
    (localdst1.block(0),localdst2.block(0),
     fev1,fev2,
     src1,Dsrc1,
     src2,Dsrc2,
     coeffs,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
}
template <int dim>
void CResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = fev.get_fe().tensor_degree();
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  const std::vector<std::vector<double> > &src = info.values[0];

  const unsigned int n_comps = fev.get_fe().n_components();
  //  std::vector<double> dirichlet_g(n_comps, 0.0);
  std::vector<std::vector<double> > bdata_values(n_comps);
  std::vector<std::vector<double> > coeffs(n_comps);
  for( unsigned int d=0; d<n_comps; ++d)
    {
      coeffs[d].resize(fev.n_quadrature_points);
      bdata_values[d].resize(fev.n_quadrature_points,0.0);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs[d],d);
    }

  LocalIntegrators::Diffusion::nitsche_residual<dim>
    (localdst.block(0), 
     fev,
     src, Dsrc,
     bdata_values, coeffs,
     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
}

// RHS INTEGRATOR
template <int dim>
CRHSIntegrator<dim>::CRHSIntegrator()
{}

template <int dim>
void CRHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::BlockVector<double> &result = dinfo.vector(0);
  const unsigned int n_quads = fev.n_quadrature_points;

  const unsigned int n_comps = fev.get_fe().n_components();
  std::vector<std::vector<double> > rhsvalues(n_comps);
  std::vector<double> rhs_f(n_comps, 1.0);
  rhs_f[1] = 1.;
  for( unsigned int d=0; d<n_comps; ++d)
      rhsvalues[d].resize(n_quads,rhs_f[d]);

  dealii::LocalIntegrators::L2::L2(result.block(0),fev,rhsvalues);
}

template <int dim>
void CRHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &, typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
void CRHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template class CMatrixIntegrator<2,false>;
template class CMatrixIntegrator<3,false>;
template class CMatrixIntegrator<2,true>;
template class CMatrixIntegrator<3,true>;
template class CResidualIntegrator<2>;
template class CResidualIntegrator<3>;
template class CRHSIntegrator<2>;
template class CRHSIntegrator<3>;
