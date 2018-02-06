#include <Integrators.h>

template <int dim>
MatrixIntegrator<dim>::MatrixIntegrator()
{}

template <int dim>
void MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto n_blocks = dinfo.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  typename std::remove_reference<decltype(info.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      const auto n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          ++c ;
        }
      auto &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs);
    }
}

template <int dim>
void MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                 dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const auto n_blocks = dinfo1.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  typename std::remove_reference<decltype(info1.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const auto deg1 = info1.fe_values(dinfo1.block_info->base_element(b)).get_fe().tensor_degree();
      const auto deg2 = info2.fe_values(dinfo2.block_info->base_element(b)).get_fe().tensor_degree();
      const auto &fev1 = info1.fe_values(dinfo1.block_info->base_element(b));
      const auto &fev2 = info2.fe_values(dinfo2.block_info->base_element(b));

      auto &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
      auto &RM12 = dinfo1.matrix(b*n_blocks + b,true).matrix;
      auto &RM21 = dinfo2.matrix(b*n_blocks + b,true).matrix;
      auto &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
      const auto n_quads = fev1.n_quadrature_points;
      const auto n_components = fev1.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev1.get_quadrature_points(),component,c);
          ++c ;
        }

      LocalIntegrators::Diffusion::ip_matrix<dim>
      (RM11,RM12,RM21,RM22,fev1,fev2,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}

template <int dim>
void MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                     typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto n_blocks = dinfo.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  typename std::remove_reference<decltype(info.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      const auto deg = info.fe_values(dinfo.block_info->base_element(b)).get_fe().tensor_degree();
      const auto n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          ++c ;
        }
      auto &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
      (M,fev,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator()
{
  this->use_cell = true;
#ifdef CG
  this->use_boundary = false;
  this->use_face = false;
#else
  this->use_boundary = true;
  this->use_face = true;
#endif
}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  auto &localdst = dinfo.vector(0);
  const auto n_blocks = localdst.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  const auto &Dsrc = info.gradients[0];
  typename std::remove_reference<decltype(info.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      const auto n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          ++c ;
        }
      AssertDimension(localdst.block(b).size(), fev.dofs_per_cell);
      dealii::VectorSlice<typename std::remove_reference<decltype(Dsrc)>::type> slice (Dsrc,b*n_components,n_components) ;
      LocalIntegrators::Diffusion::cell_residual<dim>(localdst.block(b), fev, slice, coeffs);
    }
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  auto &localdst1 = dinfo1.vector(0);
  auto &localdst2 = dinfo2.vector(0);
  const auto n_blocks = localdst1.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  const auto &Dsrc1 = info1.gradients[0];
  const auto &Dsrc2 = info2.gradients[0];
  const auto &src1 = info1.values[0];
  const auto &src2 = info2.values[0];

  typename std::remove_reference<decltype(info1.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const auto &fev1 = info1.fe_values(dinfo1.block_info->base_element(b));
      const auto &fev2 = info2.fe_values(dinfo2.block_info->base_element(b));
      const auto deg1 = fev1.get_fe().tensor_degree();
      const auto deg2 = fev2.get_fe().tensor_degree();

      const auto n_quads = fev1.n_quadrature_points;
      const auto n_components = fev1.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev1.get_quadrature_points(),component,c);
          ++c ;
        }
      dealii::VectorSlice<typename std::remove_reference<decltype(src1)>::type> slice1 (src1,b*n_components,n_components) ;
      dealii::VectorSlice<typename std::remove_reference<decltype(Dsrc1)>::type> Dslice1 (Dsrc1,b*n_components,n_components) ;
      dealii::VectorSlice<typename std::remove_reference<decltype(src2)>::type> slice2 (src2,b*n_components,n_components) ;
      dealii::VectorSlice<typename std::remove_reference<decltype(Dsrc2)>::type> Dslice2 (Dsrc2,b*n_components,n_components) ;
      LocalIntegrators::Diffusion::ip_residual<dim>
      (localdst1.block(b),localdst2.block(b),
       fev1,fev2,
       slice1,Dslice1,
       slice2,Dslice2,
       coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));
    }
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  auto &localdst = dinfo.vector(0);
  const auto n_blocks = localdst.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  const auto &Dsrc = info.gradients[0];
  const auto &src = info.values[0];
  typename std::remove_reference<decltype(info.values[0])>::type bdata_values;
  typename std::remove_reference<decltype(info.values[0])>::type coeffs;
  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      const auto deg = fev.get_fe().tensor_degree();
      const auto n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          ++c ;
        }
      bdata_values.resize(n_components);
      for (auto &component : bdata_values)
        component.resize(n_quads,0.);
      dealii::VectorSlice<typename std::remove_reference<decltype(src)>::type> slice (src,b*n_components,n_components) ;
      dealii::VectorSlice<typename std::remove_reference<decltype(Dsrc)>::type> Dslice (Dsrc,b*n_components,n_components) ;
      LocalIntegrators::Diffusion::nitsche_residual<dim>
      (localdst.block(b),
       fev,
       slice, Dslice,
       bdata_values, coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

// RHS INTEGRATOR
template <int dim>
RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components)
  : exact_solution(n_components)
{
  this->use_cell = true;
#ifdef CG
  this->use_boundary = false;
#else
  this->use_boundary = true;
#endif
  this->use_face = true;
}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  auto &result = dinfo.vector(0);
  const auto n_blocks = result.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  typename std::remove_reference<decltype(info.values[0][0])>::type exact_laplacian;
  typename std::remove_reference<decltype(info.gradients[0][0])>::type exact_gradients;
  typename std::remove_reference<decltype(info.values[0][0])>::type coeffs_values;
  typename std::remove_reference<decltype(info.gradients[0][0])>::type coeffs_gradients;
  typename std::remove_reference<decltype(info.values[0])>::type f;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      const auto n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      const auto &q_points = fev.get_quadrature_points();

      f.resize(n_components);
      exact_laplacian.resize(n_quads);
      exact_gradients.resize(n_quads);
      coeffs_values.resize(n_quads);
      coeffs_gradients.resize(n_quads);
      unsigned int c = 0 ;
      for (auto &component : f)
        {
          component.resize(n_quads);
          exact_solution.laplacian_list(q_points, exact_laplacian, c);
          exact_solution.gradient_list(q_points, exact_gradients, c);
          diffcoeff.value_list(q_points,coeffs_values,c);
          diffcoeff.gradient_list(q_points,coeffs_gradients,c);
          for (unsigned int q = 0 ; q<component.size() ; ++q)
            component[q] = coeffs_gradients[q]*exact_gradients[q]+coeffs_values[q]*exact_laplacian[q];
          ++c ;
        }

      dealii::LocalIntegrators::L2::L2(result.block(b),fev,f,-1.);
    }
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
#ifdef CG
  (void) dinfo;
  (void) info;
#else
  auto &result = dinfo.vector(0);
  const auto n_blocks = result.n_blocks();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

  std::vector<double> coeffs;
  std::vector<double> boundary_values;

  for (unsigned int b=0; b<n_blocks; ++b)
    {
      const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
      auto &local_vector = dinfo.vector(0).block(b);
      const auto deg = fev.get_fe().tensor_degree();
      const auto penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();
      boundary_values.resize(fev.n_quadrature_points);
      coeffs.resize(fev.n_quadrature_points);
      const auto &q_points = fev.get_quadrature_points();
      diffcoeff.value_list(q_points,coeffs,b);
      exact_solution.value_list(q_points, boundary_values, b);

      for (unsigned int k=0; k<fev.n_quadrature_points; ++k)
        for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
          local_vector(i) += coeffs[k]
                             * (fev.shape_value(i,k) * penalty * boundary_values[k]
                                - (fev.normal_vector(k) * fev.shape_grad(i,k)) * boundary_values[k])
                             * fev.JxW(k);
    }
#endif // CG OFF
}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &,
			      dealii::MeshWorker::DoFInfo<dim> &,
			      typename dealii::MeshWorker::IntegrationInfo<dim> &,
			      typename dealii::MeshWorker::IntegrationInfo<dim> &) const
{}

#ifndef HEADER_IMPLEMENTATION
#include "Integrators.inst"
#endif

