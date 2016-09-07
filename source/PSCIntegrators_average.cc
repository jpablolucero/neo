#include <PSCIntegrators_average.h>

// MATRIX INTEGRATOR
template <int dim>
PSCMatrixIntegrator<dim>::PSCMatrixIntegrator()
{}

template <int dim>
void PSCMatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
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
          this->diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          ++c ;
        }
      auto &M = dinfo.matrix(b*n_blocks + b).matrix;
      LocalIntegrators::Diffusion::cell_matrix<dim>(M,fev,coeffs);
    }
}

template <int dim>
void PSCMatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
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


      dealii::FullMatrix<double> &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
      const unsigned int n_quads = fev1.n_quadrature_points;
      const auto n_components = fev1.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          this->diffcoeff.value_list(fev1.get_quadrature_points(),component,c);
          // we want to average boundary and face contributions
          for (unsigned int i=0; i<fev1.get_quadrature_points().size(); ++i)
            component[i]*=.5;
          ++c;
        }

      //face contributions
      //These are unused
      dealii::FullMatrix<double> M21(dinfo1.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> M12(dinfo2.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
      LocalIntegrators::Diffusion::ip_matrix<dim>
      (RM11,M12,M21,RM22,fev1,fev2,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo2,deg1,deg2));

      //boundary contributions
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
      (RM11,fev1,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo1,dinfo1,deg1,deg1));
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
      (RM22,fev2,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo2,dinfo2,deg2,deg2));
    }
}

template <int dim>
void PSCMatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                        typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto n_blocks = dinfo.block_info->local().size();
  Assert(n_blocks>0, dealii::ExcMessage("BlockInfo not initialized!"));
  typename std::remove_reference<decltype(info.values[0])>::type coeffs;

  for (unsigned int b=0; b<n_blocks; ++b )
    {
      const dealii::FEValuesBase<dim> &fev = info.fe_values(dinfo.block_info->base_element(b));
      const unsigned int deg = info.fe_values(dinfo.block_info->base_element(b)).get_fe().tensor_degree();

      dealii::FullMatrix<double> &RM11 = dinfo.matrix(b*n_blocks + b,false).matrix;
      const unsigned int n_quads = fev.n_quadrature_points;
      const auto n_components = fev.get_fe().n_components();
      coeffs.resize(n_components);
      unsigned int c = 0;
      for (auto &component : coeffs)
        {
          component.resize(n_quads);
          this->diffcoeff.value_list(fev.get_quadrature_points(),component,c);
          // we want to average boundary and face contributions
          for (unsigned int i=0; i<fev.get_quadrature_points().size(); ++i)
            component[i]*=.5;
          ++c;
        }

      //face contributions
      //These are unused
      dealii::FullMatrix<double> M21(dinfo.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> M12(dinfo.matrix(b*n_blocks+b,true).matrix.n());
      dealii::FullMatrix<double> M22(dinfo.matrix(b*n_blocks+b,false).matrix.n());;
      LocalIntegrators::Diffusion::ip_matrix<dim>
      (RM11,M12,M21,M22,fev,fev,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));

      //boundary contributions
      LocalIntegrators::Diffusion::nitsche_matrix<dim>
      (RM11,fev,coeffs,
       dealii::LocalIntegrators::Laplace::compute_penalty(dinfo,dinfo,deg,deg));
    }
}

template class PSCMatrixIntegrator<2>;
template class PSCMatrixIntegrator<3>;
