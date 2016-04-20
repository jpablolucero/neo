#include <Integrators.h>

// MATRIX INTEGRATOR
template <int dim,bool same_diagonal>
MatrixIntegrator<dim,same_diagonal>::MatrixIntegrator()
{}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim, same_diagonal>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                                typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const unsigned int n_blocks = dinfo.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  std::vector<double> coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int n_quads = fev.n_quadrature_points;
      coeffs.resize(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs);
    }
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                               dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{

  const unsigned int n_blocks = dinfo1.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  std::vector<double> coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const unsigned int deg1 = info1.fe_values(dinfo1.block_info->base_element(b)).get_fe().tensor_degree();
      const unsigned int deg2 = info2.fe_values(dinfo2.block_info->base_element(b)).get_fe().tensor_degree();
      const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(dinfo1.block_info->base_element(b));
      const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(dinfo2.block_info->base_element(b));

      dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
      const unsigned int n_quads = fev1.n_quadrature_points;
      coeffs.resize(n_quads);
      diffcoeff.value_list(fev1.get_quadrature_points(), coeffs, b);
      //These are unused
      dealii::FullMatrix<double> M21(dinfo1.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> M12(dinfo2.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> M22(dinfo2.matrix(b*n_blocks+b,false).matrix.n());
      if (same_diagonal)
        {
          LocalIntegrators::Diffusion::ip_matrix<dim>
          (RM11,M12,M21,M22,fev1,fev2,coeffs,
           dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
        }
      else
        {
          dealii::FullMatrix<double> &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
          LocalIntegrators::Diffusion::ip_matrix<dim>
          (RM11,M12,M21,RM22,fev1,fev2,coeffs,
           dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
        }
    }
}

template <int dim,bool same_diagonal>
void MatrixIntegrator<dim,same_diagonal>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const unsigned int n_blocks = dinfo.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  std::vector<double> coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int deg = info.fe_values(dinfo.block_info->base_element(b)).get_fe().tensor_degree();
      const unsigned int n_quads = fev.n_quadrature_points;
      coeffs.resize(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
      (M,fev,coeffs,
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
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  std::vector<double> coeffs;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int n_quads = fev.n_quadrature_points;
      coeffs.resize(n_quads);
      diffcoeff.value_list(fev.get_quadrature_points(),coeffs,b);
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);
      LocalIntegrators::Diffusion::cell_residual<dim>(localdst.block(b), fev, Dsrc[b], coeffs);
    }
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  dealii::BlockVector<double> &localdst1 = dinfo1.vector(0);
  dealii::BlockVector<double> &localdst2 = dinfo2.vector(0);
  const unsigned int n_blocks = localdst1.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc1 = info1.gradients[0];
  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc2 = info2.gradients[0];
  const std::vector<std::vector<double> > &src1 = info1.values[0];
  const std::vector<std::vector<double> > &src2 = info2.values[0];

  std::vector<double> coeffs;
  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const dealii::FEValuesBase<dim> &fev1 = info1.fe_values(dinfo1.block_info->base_element(b));
      const dealii::FEValuesBase<dim> &fev2 = info2.fe_values(dinfo2.block_info->base_element(b));
      const unsigned int deg1 = fev1.get_fe().tensor_degree();
      const unsigned int deg2 = fev2.get_fe().tensor_degree();

      const unsigned int n_quads = fev1.n_quadrature_points;
      coeffs.resize(n_quads);
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
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  dealii::BlockVector<double> &localdst = dinfo.vector(0);
  const unsigned int n_blocks = localdst.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  const std::vector<std::vector<dealii::Tensor<1,dim> > > &Dsrc = info.gradients[0];
  const std::vector<std::vector<double> > &src = info.values[0];
  std::vector<std::vector<double> > bdata_values(n_blocks);
  std::vector<double> coeffs;
  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int deg = fev.get_fe().tensor_degree();
      const unsigned int n_quads = fev.n_quadrature_points;
      bdata_values[b].resize(src[b].size(),0.);
      coeffs.resize(n_quads);
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
RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components)
  : exact_solution(n_components)
{}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  dealii::BlockVector<double> &result = dinfo.vector(0);
  const unsigned int n_blocks = result.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  std::vector<double> exact_laplacian;
  std::vector<dealii::Tensor<1,dim> > exact_gradients;
  std::vector<double> coeffs_values;
  std::vector<dealii::Tensor<1,dim> > coeffs_gradients;
  std::vector<double> f;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int n_quads = fev.n_quadrature_points;
      const std::vector<dealii::Point<dim> > &q_points = fev.get_quadrature_points();

      exact_laplacian.resize(n_quads);
      exact_gradients.resize(n_quads);
      coeffs_values.resize(n_quads);
      coeffs_gradients.resize(n_quads);
      f.resize(n_quads);

      exact_solution.laplacian_list(q_points, exact_laplacian, b);
      exact_solution.gradient_list(q_points, exact_gradients, b);
      diffcoeff.value_list(q_points,coeffs_values,b);
      diffcoeff.gradient_list(q_points,coeffs_gradients,b);

      for (unsigned int q=0; q<n_quads; ++q)
        f[q] = coeffs_gradients[q]*exact_gradients[q]+coeffs_values[q]*exact_laplacian[q];

      dealii::LocalIntegrators::L2::L2(result.block(b),fev,f,-1.);
#ifdef CG
      //we need to do the same thing as for matrix integrator
      dealii::FullMatrix<double> &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs_values);
#endif
    }
}
template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
#ifndef CG
  dealii::BlockVector<double> &result = dinfo.vector(0);
  const unsigned int n_blocks = result.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  std::vector<double> coeffs;
  std::vector<double> boundary_values;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      dealii::Vector<double> &local_vector = dinfo.vector(0).block(b);
      const unsigned int deg = fev.get_fe().tensor_degree();
      const double penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();
      boundary_values.resize(fev.n_quadrature_points);
      coeffs.resize(fev.n_quadrature_points);
      const std::vector<dealii::Point<dim> > &q_points = fev.get_quadrature_points();
      diffcoeff.value_list(q_points,coeffs,b);
      exact_solution.value_list(q_points, boundary_values, b);

      for (unsigned k=0; k<fev.n_quadrature_points; ++k)
        for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
          local_vector(i) += coeffs[k]
                             * (fev.shape_value(i,k) * penalty * boundary_values[k]
                                - (fev.normal_vector(k) * fev.shape_grad(i,k)) * boundary_values[k])
                             * fev.JxW(k);
    }
#endif


}

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
