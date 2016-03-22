#include <BlockIntegrators.h>

// MATRIX INTEGRATOR
template <int dim,bool same_diagonal>
BMatrixIntegrator<dim,same_diagonal>::BMatrixIntegrator()
{}

template <int dim,bool same_diagonal>
void BMatrixIntegrator<dim,same_diagonal>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0) ;
  const unsigned int n_blocks = dinfo.block_info->global().size();

  AssertDimension( dinfo.n_matrices(), n_blocks*n_blocks );
  // dealii::deallog << "local_blocksize: " << dinfo.block_info->local().size()
  // 		  << "global_blocksize: " << dinfo.block_info->global().size()
  // 		  << std::endl;;

  for( unsigned int b=0; b<n_blocks; ++b )
    {
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.m(), fev.dofs_per_cell);
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.n(), fev.dofs_per_cell);

      std::vector<double> coeffs(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs) ;
    }
}

template <int dim,bool same_diagonal>
void BMatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
					       dealii::MeshWorker::DoFInfo<dim> &dinfo2,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  const unsigned int n_blocks = dinfo1.block_info->global().size();
  // dealii::deallog << "local_blocksize1: " << dinfo1.block_info->local().size()
  // 		  << "local_blocksize2: " << dinfo2.block_info->local().size()
  // 		  << "global_blocksize2: " << dinfo2.block_info->global().size()
  // 		  << "global_blocksize2: " << dinfo2.block_info->global().size()
  // 		  << std::endl;;

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();
  dealii::FullMatrix<double> Mtrash1(dinfo1.matrix(0).matrix.m());
  dealii::FullMatrix<double> Mtrash2(dinfo2.matrix(0).matrix.m());

  AssertDimension( dinfo1.n_matrices(), n_blocks*n_blocks );
  AssertDimension( dinfo2.n_matrices(), dinfo2.block_info->global().size()*dinfo2.block_info->global().size());

  if (same_diagonal)
    {
      for( unsigned int b=0; b<n_blocks; ++b )
	{
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.m(), fev1.dofs_per_cell);
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.n(), fev1.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.m(), fev2.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.n(), fev2.dofs_per_cell);
	  
	  std::vector<double> coeffs(fev1.n_quadrature_points);
    	  diffcoeff.value_list(fev1.get_quadrature_points(), coeffs, b);
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  LocalIntegrators::Diffusion::ip_matrix<dim>
	    (RM11,Mtrash1,Mtrash2,Mtrash2,fev1,fev2,coeffs,
	     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
	}
    }
  else
    {
      for( unsigned int b=0; b<n_blocks; ++b )
	{
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.m(), fev1.dofs_per_cell);
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.n(), fev1.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.m(), fev2.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.n(), fev2.dofs_per_cell);
	  
	  std::vector<double> coeffs(fev1.n_quadrature_points);
    	  diffcoeff.value_list(fev1.get_quadrature_points(), coeffs, b);
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  dealii::FullMatrix<double> &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
	  LocalIntegrators::Diffusion::ip_matrix<dim>
	    (RM11,Mtrash1,Mtrash2,RM22,fev1,fev2,coeffs,
	     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
	}
    }
}

template <int dim,bool same_diagonal>
void BMatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
						   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const unsigned int n_blocks = dinfo.block_info->global().size();

  AssertDimension( dinfo.n_matrices(), n_blocks*n_blocks );

  for( unsigned int b=0; b<n_blocks; ++b )
    {
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.m(), fev.dofs_per_cell);
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.n(), fev.dofs_per_cell);

      std::vector<double> coeffs(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
	(M,fev,coeffs,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

// RESIDUAL INTEGRATOR
template <int dim>
BResidualIntegrator<dim>::BResidualIntegrator()
{}

template <int dim>
void BResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];

  for( unsigned int b=0; b<n_blocks; ++b)
    {
      std::vector<double> coeffs(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);
      LocalIntegrators::Diffusion::cell_residual<dim>(localdst.block(b), fev, Dsrc[b], coeffs) ;
    }
}
  
template <int dim>
void BResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
				   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  dealii::BlockVector<double> &localdst1 = dinfo1.vector(0);
  dealii::BlockVector<double> &localdst2 = dinfo2.vector(0);
  const unsigned int n_blocks = localdst1.n_blocks();

  const unsigned int deg1 = fev1.get_fe().tensor_degree();
  const unsigned int deg2 = fev2.get_fe().tensor_degree();

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1 = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2 = info2.gradients[0];
  const std::vector<std::vector<double> > &src1 = info1.values[0];
  const std::vector<std::vector<double> > &src2 = info2.values[0];

  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(localdst1.block(b).size(), fev1.dofs_per_cell);
      
      std::vector<double> coeffs(fev1.n_quadrature_points);
      diffcoeff.value_list(fev1.get_quadrature_points(),coeffs,b);
      LocalIntegrators::Diffusion::ip_residual<dim>
	(localdst1.block(b),localdst2.block(b),
	 fev1,fev2,
	 src1[b],Dsrc1[b],
	 src2[b],Dsrc2[b],
	 coeffs,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}
template <int dim>
void BResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = fev.get_fe().tensor_degree();
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  const std::vector<std::vector<double> > &src = info.values[0];

  std::vector<double> dirichlet_g(n_blocks, 0.0);
  std::vector<std::vector<double> > bdata_values(n_blocks);
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);

      bdata_values[b].resize(src[b].size(),dirichlet_g[b]);
      std::vector<double> coeffs(fev.n_quadrature_points);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      LocalIntegrators::Diffusion::nitsche_residual<dim>
	(localdst.block(b), 
	 fev,
	 src[b], Dsrc[b],
	 bdata_values[b], coeffs,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

// RHS INTEGRATOR
template <int dim>
BRHSIntegrator<dim>::BRHSIntegrator()
{}

template <int dim>
void BRHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::BlockVector<double> &result = dinfo.vector(0);
  const unsigned int n_blocks = result.n_blocks();
  const unsigned int n_quads = fev.n_quadrature_points;

  std::vector<std::vector<double> > rhsvalues(n_blocks);
  std::vector<double> rhs_f(n_blocks, 1.0);
  rhs_f[1] = 0.44;
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(result.block(b).size(), fev.dofs_per_cell);

      rhsvalues[b].resize(n_quads,rhs_f[b]);
      dealii::LocalIntegrators::L2::L2(result.block(b),fev,rhsvalues[b]);
    }
}

template <int dim>
void BRHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &, typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template <int dim>
void BRHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
                              dealii::MeshWorker::DoFInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

template class BMatrixIntegrator<2,false>;
template class BMatrixIntegrator<3,false>;
template class BMatrixIntegrator<2,true>;
template class BMatrixIntegrator<3,true>;
template class BResidualIntegrator<2>;
template class BResidualIntegrator<3>;
template class BRHSIntegrator<2>;
template class BRHSIntegrator<3>;
