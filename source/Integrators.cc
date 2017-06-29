#include <Integrators.h>

#ifndef MATRIXFREE
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
      ::LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs);
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
{}

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

// // RHS INTEGRATOR
// template <int dim>
// RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components)
//   : exact_solution(n_components)
// {
//   this->use_cell = true;
// #ifdef CG
//   this->use_boundary = false;
// #else
//   this->use_boundary = true;
// #endif
//   this->use_face = false;
// }

// template <int dim>
// void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
// {
//   auto &result = dinfo.vector(0);
//   const auto n_blocks = result.n_blocks();
//   Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

//   typename std::remove_reference<decltype(info.values[0][0])>::type exact_laplacian;
//   typename std::remove_reference<decltype(info.gradients[0][0])>::type exact_gradients;
//   typename std::remove_reference<decltype(info.values[0][0])>::type coeffs_values;
//   typename std::remove_reference<decltype(info.gradients[0][0])>::type coeffs_gradients;
//   typename std::remove_reference<decltype(info.values[0])>::type f;

//   for (unsigned int b=0; b<n_blocks; ++b)
//     {
//       const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
//       const auto n_quads = fev.n_quadrature_points;
//       const auto n_components = fev.get_fe().n_components();
//       const auto &q_points = fev.get_quadrature_points();

//       f.resize(n_components);
//       exact_laplacian.resize(n_quads);
//       exact_gradients.resize(n_quads);
//       coeffs_values.resize(n_quads);
//       coeffs_gradients.resize(n_quads);
//       unsigned int c = 0 ;
//       for (auto &component : f)
//         {
//           component.resize(n_quads);
//           exact_solution.laplacian_list(q_points, exact_laplacian, c);
//           exact_solution.gradient_list(q_points, exact_gradients, c);
//           diffcoeff.value_list(q_points,coeffs_values,c);
//           diffcoeff.gradient_list(q_points,coeffs_gradients,c);
//           for (unsigned int q = 0 ; q<component.size() ; ++q)
//             component[q] = coeffs_gradients[q]*exact_gradients[q]+coeffs_values[q]*exact_laplacian[q];
//           ++c ;
//         }

//       dealii::LocalIntegrators::L2::L2(result.block(b),fev,f,-1.);
// #ifdef CG
//       //we need to do the same thing as for matrix integrator
//       auto &M = dinfo.matrix(b*n_blocks + b).matrix;
//       LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs_values);
// #endif // CG
//     }
// }
// template <int dim>
// void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
// {
// #ifndef CG
//   auto &result = dinfo.vector(0);
//   const auto n_blocks = result.n_blocks();
//   Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));

//   std::vector<double> coeffs;
//   std::vector<double> boundary_values;

//   for (unsigned int b=0; b<n_blocks; ++b)
//     {
//       const auto &fev = info.fe_values(dinfo.block_info->base_element(b));
//       auto &local_vector = dinfo.vector(0).block(b);
//       const auto deg = fev.get_fe().tensor_degree();
//       const auto penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();
//       boundary_values.resize(fev.n_quadrature_points);
//       coeffs.resize(fev.n_quadrature_points);
//       const auto &q_points = fev.get_quadrature_points();
//       diffcoeff.value_list(q_points,coeffs,b);
//       exact_solution.value_list(q_points, boundary_values, b);

//       for (unsigned int k=0; k<fev.n_quadrature_points; ++k)
//         for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
//           local_vector(i) += coeffs[k]
//                              * (fev.shape_value(i,k) * penalty * boundary_values[k]
//                                 - (fev.normal_vector(k) * fev.shape_grad(i,k)) * boundary_values[k])
//                              * fev.JxW(k);
//     }
// #endif // CG OFF
// }

#else // MATRIXFREE OFF
template <int dim>
void MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto &fev = info.fe_values(0);
  auto &M = dinfo.matrix(0).matrix;
  dealii::LocalIntegrators::Laplace::cell_matrix<dim>(M,fev);
}

template <int dim>
void MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                 dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                 typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  const auto deg1 = info1.fe_values(0).get_fe().tensor_degree();
  const auto deg2 = info2.fe_values(0).get_fe().tensor_degree();
  const auto &fev1 = info1.fe_values(0);
  const auto &fev2 = info2.fe_values(0);

  auto &RM11 = dinfo1.matrix(0,false).matrix;
  auto &RM12 = dinfo1.matrix(0,true).matrix;
  auto &RM21 = dinfo2.matrix(0,true).matrix;
  auto &RM22 = dinfo2.matrix(0,false).matrix;
  const auto penalty = dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2);
  dealii::LocalIntegrators::Laplace::ip_matrix<dim>
  (RM11,RM12,RM21,RM22,fev1,fev2,penalty);
}

template <int dim>
void MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                     typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto &fev = info.fe_values(0);
  const auto deg = info.fe_values(0).get_fe().tensor_degree();

  auto &M = dinfo.matrix(0).matrix;
  const auto penalty = dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg);
  dealii::LocalIntegrators::Laplace::nitsche_matrix<dim>
  (M,fev,penalty);
}

// MatrixFree Integrator
template <int dim, int fe_degree, int n_q_points_1d, int n_comp, typename number>
MFIntegrator<dim,fe_degree,n_q_points_1d,n_comp,number>::MFIntegrator()
{}

template <int dim, int fe_degree, int n_q_points_1d, int n_comp, typename number>
void
MFIntegrator<dim,fe_degree,n_q_points_1d,n_comp,number>::cell(const dealii::MatrixFree<dim,number>       &data,
    LA::MPI::Vector                            &dst,
    const LA::MPI::Vector                      &src,
    const std::pair<unsigned int,unsigned int> &cell_range) const
{
  dealii::FEEvaluation<dim,fe_degree,n_q_points_1d,n_comp,number> phi (data);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);
      phi.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
        phi.submit_gradient (phi.get_gradient(q), q);
      phi.integrate (false,true);
      phi.distribute_local_to_global (dst);
    }
}

template <int dim, int fe_degree, int n_q_points_1d, int n_comp, typename number>
void
MFIntegrator<dim,fe_degree,n_q_points_1d,n_comp,number>::face(const dealii::MatrixFree<dim,number>       &data,
    LA::MPI::Vector                            &dst,
    const LA::MPI::Vector                      &src,
    const std::pair<unsigned int,unsigned int> &face_range) const
{
  dealii::FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_comp,number> fe_eval(data,true);
  dealii::FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_comp,number> fe_eval_neighbor(data,false);
  for (unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);
      dealii::VectorizedArray<number> sigmaF =
        (fe_eval.get_normal_volume_fraction() +
         fe_eval_neighbor.get_normal_volume_fraction()) *
        (number)(std::max(fe_degree,1) * (fe_degree + 1.0)) * 0.5;

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          dealii::VectorizedArray<number> avg_jump_value = (fe_eval.get_value(q)-
                                                            fe_eval_neighbor.get_value(q)) * 0.5;
          dealii::VectorizedArray<number> mean_valgrad =
            fe_eval.get_normal_gradient(q) +
            fe_eval_neighbor.get_normal_gradient(q);
          mean_valgrad = avg_jump_value * 2. * sigmaF -
                         mean_valgrad * 0.5;
          fe_eval.submit_normal_gradient(-avg_jump_value,q);
          fe_eval_neighbor.submit_normal_gradient(-avg_jump_value,q);
          fe_eval.submit_value(mean_valgrad,q);
          fe_eval_neighbor.submit_value(-mean_valgrad,q);
        }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
}

template <int dim, int fe_degree, int n_q_points_1d, int n_comp, typename number>
void
MFIntegrator<dim,fe_degree,n_q_points_1d,n_comp,number>::boundary(const dealii::MatrixFree<dim,number>       &data,
    LA::MPI::Vector                            &dst,
    const LA::MPI::Vector                      &src,
    const std::pair<unsigned int,unsigned int> &face_range) const
{
  dealii::FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_comp,number> fe_eval(data, true);
  for (unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      dealii::VectorizedArray<number> sigmaF =
        (fe_eval.get_normal_volume_fraction()) *
        (number)(std::max(1,fe_degree) * (fe_degree + 1.0));

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          dealii::VectorizedArray<number> jump_value = fe_eval.get_value(q);
          dealii::VectorizedArray<number> mean_valgrad = -fe_eval.get_normal_gradient(q);
          mean_valgrad += jump_value * sigmaF * 2.0;
          fe_eval.submit_normal_gradient(-jump_value,q);
          fe_eval.submit_value(mean_valgrad,q);
        }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
}

template <int dim>
RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components)
  :
  ref_rhs(n_components),
  ref_solution(n_components)
{
  this->use_cell = true;
  this->use_boundary = true;
  this->use_face = false;
}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto &fev = info.fe_values(0);
  auto &local_dst = dinfo.vector(0).block(0);
  const auto n_quads = fev.n_quadrature_points;
  const auto &q_points = fev.get_quadrature_points();

  std::vector<double> rhs_values;
  rhs_values.resize(n_quads);
  ref_rhs.value_list(q_points, rhs_values);

  for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
    {
      double rhs_val = 0;
      for (unsigned int q=0; q<n_quads; ++q)
        rhs_val += (fev.shape_value(i,q) * rhs_values[q]) * fev.JxW(q);
      local_dst(i) += rhs_val;
    }
}

template <int dim>
void
RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                             typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto &fev = info.fe_values(0);
  auto &local_dst = dinfo.vector(0).block(0);
  const auto deg = fev.get_fe().tensor_degree();
  const auto penalty = 2. * deg * (deg+1) * dinfo.face->measure() / dinfo.cell->measure();
  const auto n_quads = fev.n_quadrature_points;
  const auto &q_points = fev.get_quadrature_points();

  std::vector<double> solution_values;
  solution_values.resize(n_quads);
  ref_solution.value_list(q_points, solution_values);

  for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
    {
      double rhs_val = 0;
      for (unsigned int q=0; q<n_quads; ++q)
        rhs_val +=
          ( penalty * fev.shape_value(i,q)
            - (fev.normal_vector(q) * fev.shape_grad(i,q)) )
          * solution_values[q] * fev.JxW(q);
      local_dst(i) += rhs_val;
    }
}

#endif // MATRIXFREE
#include "Integrators.inst"
