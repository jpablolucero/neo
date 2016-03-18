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

  AssertDimension( dinfo.n_matrices(), n_blocks*n_blocks )
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
      LocalIntegrators::Diffusion::cell_matrix<dim, Coefficient<dim> >(M,fev) ;
    }
}

template <int dim,bool same_diagonal>
void BMatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
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
void BMatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
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
      LocalIntegrators::Diffusion::cell_residual<dim,Coefficient<dim> >(localdst.block(b), fev, Dsrc[b]) ;
    }
}
  
template <int dim>
void BResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
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
void BResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, 
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
  // // dealii::deallog << "result(1).size" << result.block(1).size() << std::endl;
  // dealii::deallog << "dinfo.n_blocks=" << dinfo.block_info->global().size() << std::endl;

  std::vector<std::vector<double> > rhsvalues(n_blocks);
  std::vector<double> rhs_f(n_blocks, 1.0);
  //  rhsvalues[0].resize(result.block(0).size(),1.0);
  //  rhsvalues[1].resize(result.block(1).size(),1.0);
  for( unsigned int b=0; b<n_blocks; ++b)
    {
      AssertDimension(result.block(b).size(), fev.dofs_per_cell);
      rhsvalues[b].resize(result.block(b).size(),rhs_f[b]);
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
