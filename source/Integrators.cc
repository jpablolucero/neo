#include <Integrators.h>

// MATRIX INTEGRATOR
template <int dim,bool same_diagonal>
MatrixIntegrator<dim,same_diagonal>::MatrixIntegrator()
{}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0) ;
  const unsigned int n_blocks = dinfo.block_info->global().size();
  const unsigned int n_quads = fev.n_quadrature_points;

  for( unsigned int b=0; b<n_blocks; ++b )
    {
      std::vector<double> coeffs_D(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs_D,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs_D) ;
      
      // Reaction: block internal
      std::vector<double> input(n_quads);
      std::vector<double> &coeffs_r = coeffs_D;
      totalcoeff.value_list(fev.get_quadrature_points(),input,b);
      reaccoeff.value_list(fev.get_quadrature_points(),coeffs_r,b,b);
      for(unsigned int q=0; q<n_quads; ++q)
	input[q] -= coeffs_r[q];
      dealii::LocalIntegrators::L2::weighted_mass_matrix(M,fev,input);
      
      // Reaction: external with other blocks
      for(unsigned int bj=0; bj<b; ++bj)
      	{
      	  std::vector<double> &input_bj = input;
      	  reaccoeff.value_list(fev.get_quadrature_points(),input_bj,b,bj);
	  dealii::FullMatrix<double> &Mj = dinfo.matrix(b*n_blocks + bj).matrix;
      	  dealii::LocalIntegrators::L2::weighted_mass_matrix(Mj,fev,input_bj);
      	}
      for(unsigned int bj=b+1; bj<n_blocks; ++bj)
      	{
      	  std::vector<double> &input_bj = coeffs_r;
      	  reaccoeff.value_list(fev.get_quadrature_points(),input_bj,b,bj);
	  dealii::FullMatrix<double> &Mj = dinfo.matrix(b*n_blocks + bj).matrix;
      	  dealii::LocalIntegrators::L2::weighted_mass_matrix(Mj,fev,input_bj);
      	}
    }
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
					       dealii::MeshWorker::DoFInfo<dim> &dinfo2,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  const unsigned int n_blocks = dinfo1.block_info->global().size();
  const unsigned int n_quads = fev1.n_quadrature_points;
  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  dealii::FullMatrix<double> M21 = dinfo1.matrix(0,true).matrix;
  dealii::FullMatrix<double> M12 = dinfo2.matrix(0,true).matrix;
  dealii::FullMatrix<double> M22 = dinfo2.matrix(0,false).matrix;
  if (same_diagonal)
    {
      for( unsigned int b=0; b<n_blocks; ++b )
	{
	  std::vector<double> coeffs_D(n_quads);
    	  diffcoeff.value_list(fev1.get_quadrature_points(), coeffs_D, b);
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  LocalIntegrators::Diffusion::ip_matrix<dim>
	    (RM11,M12,M21,M22,fev1,fev2,coeffs_D,
	     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
	}
    }
  else
    {
      for( unsigned int b=0; b<n_blocks; ++b )
	{
	  std::vector<double> coeffs_D(n_quads);
    	  diffcoeff.value_list(fev1.get_quadrature_points(), coeffs_D, b);
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  dealii::FullMatrix<double> &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
	  LocalIntegrators::Diffusion::ip_matrix<dim>
	    (RM11,M12,M21,RM22,fev1,fev2,coeffs_D,
	     dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
	}
    }
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
						   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
  const unsigned int n_blocks = dinfo.block_info->global().size();
  const unsigned int n_quads = fev.n_quadrature_points;

  for( unsigned int b=0; b<n_blocks; ++b )
    {
      std::vector<double> coeffs_D(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs_D,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
	(M,fev,coeffs_D,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

// RESIDUAL INTEGRATOR
template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator()
{}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  const unsigned int n_quads = fev.n_quadrature_points;
  
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  const std::vector<std::vector<double> > &src = info.values[0];

  for( unsigned int b=0; b<n_blocks; ++b)
    {
      // Diffusion
      std::vector<double> coeffs_D(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs_D,b);
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);
      LocalIntegrators::Diffusion::cell_residual<dim>(localdst.block(b), fev, Dsrc[b], coeffs_D) ;

      // Reaction: block internal
      std::vector<double> input(n_quads);
      std::vector<double> &coeffs_r = coeffs_D;
      totalcoeff.value_list(fev.get_quadrature_points(),input,b);
      reaccoeff.value_list(fev.get_quadrature_points(),coeffs_r,b,b);
      for(unsigned int q=0; q<n_quads; ++q)
      	input[q] = (input[q]-coeffs_r[q]) * src[b][q];
      dealii::LocalIntegrators::L2::L2(localdst.block(b),fev,input);
      
      // Reaction: external with other blocks
      for(unsigned int bj=0; bj<b; ++bj)
      	{
      	  std::vector<double> &input_bj = input;
      	  reaccoeff.value_list(fev.get_quadrature_points(),input_bj,b,bj);
      	  for(unsigned int q=0; q<n_quads; ++q)
      	    input_bj[q] *= src[bj][q];
      	  dealii::LocalIntegrators::L2::L2(localdst.block(b),fev,input_bj);
      	}
      for(unsigned int bj=b+1; bj<n_blocks; ++bj)
      	{
      	  std::vector<double> &input_bj = coeffs_r;
      	  reaccoeff.value_list(fev.get_quadrature_points(),input_bj,b,bj);
      	  for(unsigned int q=0; q<n_quads; ++q)
      	    input_bj[q] *= src[bj][q];
      	  dealii::LocalIntegrators::L2::L2(localdst.block(b),fev,input_bj);
      	}
    }
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
				   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
				   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(0);
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(0);
  dealii::BlockVector<double> &localdst1 = dinfo1.vector(0);
  dealii::BlockVector<double> &localdst2 = dinfo2.vector(0);
  const unsigned int n_blocks = localdst1.n_blocks();
  const unsigned int n_quads = fev1.n_quadrature_points;
  const unsigned int deg1 = fev1.get_fe().tensor_degree();
  const unsigned int deg2 = fev2.get_fe().tensor_degree();

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1 = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2 = info2.gradients[0];
  const std::vector<std::vector<double> > &src1 = info1.values[0];
  const std::vector<std::vector<double> > &src2 = info2.values[0];
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      std::vector<double> coeffs_D(n_quads);
      diffcoeff.value_list(fev1.get_quadrature_points(),coeffs_D,b);
      LocalIntegrators::Diffusion::ip_residual<dim>
	(localdst1.block(b),localdst2.block(b),
	 fev1,fev2,
	 src1[b],Dsrc1[b],
	 src2[b],Dsrc2[b],
	 coeffs_D,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = fev.get_fe().tensor_degree();
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  const unsigned int n_quads = fev.n_quadrature_points;

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  const std::vector<std::vector<double> > &src = info.values[0];
  std::vector<std::vector<double> > bdata_values(n_blocks);
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      bdata_values[b].resize(src[b].size(),0.);
      std::vector<double> coeffs_D(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs_D,b);
      LocalIntegrators::Diffusion::nitsche_residual<dim>
	(localdst.block(b), 
	 fev,
	 src[b], Dsrc[b],
	 bdata_values[b], coeffs_D,
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

// RHS INTEGRATOR
template <int dim>
RHSIntegrator<dim>::RHSIntegrator()
{}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  dealii::BlockVector<double> &result = dinfo.vector(0);
  const unsigned int n_blocks = result.n_blocks();
  const unsigned int n_quads = fev.n_quadrature_points;

  std::vector<std::vector<double> > rhsvalues(n_blocks);
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      rhsvalues[b].resize(n_quads,1.0);
      dealii::LocalIntegrators::L2::L2(result.block(b),fev,rhsvalues[b]);
    }
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
