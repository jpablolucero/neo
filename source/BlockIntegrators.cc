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
  // dealii::deallog << "dinfo.n_blocks=" << dinfo.block_info->global().size() << std::endl;
  // dealii::deallog << "dinfo.n_matrices()" <<  dinfo.n_matrices() << std::endl;
  // dealii::deallog << "dofs_per_cell wrt each block =" << fev.dofs_per_cell << std::endl;

  //  for( unsigned int n=0; n<dinfo.n_matrices(); ++n )
    // dealii::deallog << "dinfo.matrix(" << n << ").globalrow = " << dinfo.matrix(n).row
    // 		    << " ,dinfo.matrix(" << n << ").globalcol = " << dinfo.matrix(n).column 
    // 		    << " ,dinfo.matrix(" << n << ").n_rows = " << dinfo.matrix(n).matrix.m()
    // 		    << " ,dinfo.matrix(" << n << ").n_cols = " << dinfo.matrix(n).matrix.n() << std::endl;

  for( unsigned int b=0; b<n_blocks; ++b )
    {
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.m(), fev.dofs_per_cell);
      AssertDimension(dinfo.matrix(b*n_blocks + b).matrix.n(), fev.dofs_per_cell);

      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      dealii::LocalIntegrators::Laplace::cell_matrix<dim>(M,fev) ;
    }
}

template <int dim,bool same_diagonal>
void BMatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
					       dealii::MeshWorker::DoFInfo<dim> &dinfo2,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
					       typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const dealii::FEValuesBase<dim> &fev1 = info1.fe_values();
  const dealii::FEValuesBase<dim> &fev2 = info2.fe_values();

  const unsigned int n_blocks = dinfo1.block_info->global().size();

  //AssertDimension( dinfo.n_matrices(), n_blocks*n_blocks )
  // dealii::deallog << "dinfo.n_blocks=" << dinfo.block_info->global().size() << std::endl;
  // dealii::deallog << "dinfo.n_matrices()" <<  dinfo.n_matrices() << std::endl;
  // dealii::deallog << "dofs_per_cell wrt each block =" << fev.dofs_per_cell << std::endl;
  
  // for( unsigned int n=0; n<dinfo1.n_matrices(); ++n )
  //   dealii::deallog << "dinfo1.matrix(" << n << ",false).globalrow = " << dinfo1.matrix(n,false).row
  //   		    << " ,dinfo1.matrix(" << n << ",false).globalcol = " << dinfo1.matrix(n,false).column 
  //   		    << " ,dinfo1.matrix(" << n << ",false).n_rows = " << dinfo1.matrix(n,false).matrix.m()
  //   		    << " ,dinfo1.matrix(" << n << ",false).n_cols = " << dinfo1.matrix(n,false).matrix.n() << std::endl;
  
  dealii::FullMatrix<double> Mtrash1(dinfo1.matrix(0).matrix.m());
  dealii::FullMatrix<double> Mtrash2(dinfo2.matrix(0).matrix.m());

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  if (same_diagonal)
    {
      for( unsigned int b=0; b<n_blocks; ++b )
	{
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.m(), fev1.dofs_per_cell);
	  AssertDimension(dinfo1.matrix(b*n_blocks + b).matrix.n(), fev1.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.m(), fev2.dofs_per_cell);
	  AssertDimension(dinfo2.matrix(b*n_blocks + b).matrix.n(), fev2.dofs_per_cell);
	  
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  dealii::LocalIntegrators::Laplace::ip_matrix<dim>
	    (RM11,Mtrash1,Mtrash2,Mtrash2,fev1,fev2,
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
	  
	  dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
	  dealii::FullMatrix<double> &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
	  dealii::LocalIntegrators::Laplace::ip_matrix<dim>
	    (RM11,Mtrash1,Mtrash2,RM22,fev1,fev2,
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

      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      dealii::LocalIntegrators::Laplace::nitsche_matrix<dim>
	(M,
	 fev,
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
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);
      dealii::LocalIntegrators::Laplace::cell_residual<dim>(localdst.block(b), fev, Dsrc[b]) ;
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

  const unsigned int deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values(0).get_fe().tensor_degree();

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1 = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2 = info2.gradients[0];
  const std::vector<std::vector<double> > &src1 = info1.values[0];
  const std::vector<std::vector<double> > &src2 = info2.values[0];

  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(localdst1.block(b).size(), fev1.dofs_per_cell);
      dealii::LocalIntegrators::Laplace::ip_residual<dim>
	(localdst1.block(b),localdst2.block(b),
	 fev1,fev2,
	 src1[b],Dsrc1[b],
	 src2[b],Dsrc2[b],
	 dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}
template <int dim>
void BResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
				       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const dealii::FEValuesBase<dim> &fev = info.fe_values(0);
  const unsigned int deg = info.fe_values(0).get_fe().tensor_degree();
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
      dealii::LocalIntegrators::Laplace::nitsche_residual<dim>
	(localdst.block(b), 
	 fev,
	 src[b], Dsrc[b],
	 bdata_values[b],
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

  // dealii::deallog << "result.n_blocks" << n_blocks << std::endl;
  // dealii::deallog << "result(0).size" << result.block(0).size() << std::endl;
  // dealii::deallog << "result(1).size" << result.block(1).size() << std::endl;
  // dealii::deallog << "dinfo.n_blocks=" << dinfo.block_info->global().size() << std::endl;

  std::vector<std::vector<double> > rhsvalues(n_blocks);
  rhsvalues[0].resize(result.block(0).size(),1.0);
  // std::vector<double> rhs_f(n_blocks, 1.0);
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(result.block(b).size(), fev.dofs_per_cell);
      dealii::LocalIntegrators::L2::L2(result.block(b),fev,rhsvalues[0]);
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
