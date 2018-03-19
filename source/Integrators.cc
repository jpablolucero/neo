#include <Integrators.h>

extern double eps ;
extern double abs_rate ;

// MATRIX INTEGRATOR
template <int dim>
MatrixIntegrator<dim>::MatrixIntegrator():
  angles("../include/integrators_data/transport/angles/D2P1K1.angles")
{}

template <int dim>
void MatrixIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                                typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  const auto n_blocks = dinfo.block_info->local().size();
  const auto &feT = info.fe_values(dinfo.block_info->base_element(n_blocks-1));
  const auto &pre = info.values[1];
  typedef dealii::VectorSlice<typename std::remove_reference<decltype(pre)>::type> SrcType ;

  auto &TT = dinfo.matrix((n_blocks-1)*n_blocks + (n_blocks-1),false).matrix;

  XS<dim> xsdata ;

  for (unsigned int b=0; b<n_blocks-1; ++b)
    {
      const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
      const auto n_comps = feV.get_fe().n_components();
      auto &M = dinfo.matrix(b*n_blocks + b,false).matrix;
      auto &TM = dinfo.matrix((n_blocks-1)*n_blocks + b,false).matrix;
      auto &MT = dinfo.matrix(b*n_blocks+(n_blocks-1),false).matrix;
      SrcType Tpreslice (pre,(n_blocks-1)*n_comps,1) ;

      const auto total = xsdata.total(feV.get_quadrature_points(),n_comps,b) ;
      const auto abs = xsdata.absorption(feV.get_quadrature_points(),angles.get_weights(),n_comps,
                                         n_blocks-1,b,1.,
					 static_cast<double>(100 - abs_rate) / 100. / static_cast<double>(n_blocks-1));

      auto Dplanck = [&](double T)->double 
      	{
      	  return (b==0) ? Dplanck_integral(xsdata.grid[b],T) : 
      	  Dplanck_integral(xsdata.grid[b],T) - Dplanck_integral(xsdata.grid[b-1],T) ;
      	};

      LocalIntegrators::Transport::cell_matrix<dim>(M,feV,angles.get_points());
      LocalIntegrators::Transport::total_matrix<dim>(M,feV,total);
      LocalIntegrators::Transport::Demission_matrix<dim>(MT,feV,Tpreslice,feT,abs,Dplanck) ;
      LocalIntegrators::Transport::T_absorption_matrix<dim>(TM,feT,feV,angles.get_weights(),abs);
      LocalIntegrators::Transport::DT_matrix<dim>(TT,feT,Tpreslice,angles.get_weights(),abs,Dplanck) ;
      for (unsigned int bin=0; bin<n_blocks-1; ++bin)
        {
          auto &M = dinfo.matrix(b*n_blocks + bin,false).matrix;
          const auto scattering = xsdata.scattering(feV.get_quadrature_points(),n_comps,bin,b,
                                                    static_cast<double>(100 - abs_rate) / 100. /(static_cast<double>(n_blocks-1)));
	  
	  LocalIntegrators::Transport::redistribution_matrix<dim>(M,feV,angles.get_weights(),scattering);
        }
    }
}

template <int dim>
void MatrixIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                               dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                               typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  {
    const auto n_blocks = dinfo1.block_info->local().size();
    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV1 = info1.fe_values(dinfo1.block_info->base_element(b));
        const auto &feV2 = info2.fe_values(dinfo2.block_info->base_element(b));

        auto &RM11 = dinfo1.matrix(b*n_blocks + b,false).matrix;
        auto &RM12 = dinfo1.matrix(b*n_blocks + b,true).matrix;
        auto &RM21 = dinfo2.matrix(b*n_blocks + b,true).matrix;
        auto &RM22 = dinfo2.matrix(b*n_blocks + b,false).matrix;
        const auto n_components = feV1.get_fe().n_components();
        const auto n_quads = feV1.n_quadrature_points;

	auto xs1 = xsdata.scattering(feV1.get_quadrature_points(),n_components,0,b);
        auto xs2 = xsdata.scattering(feV2.get_quadrature_points(),n_components,0,b);
	
        LocalIntegrators::Transport::ip_matrix<dim>
        (RM11,RM12,RM21,RM22,feV1,feV2,angles.get_points(),angles.get_weights(),
         xs1,xs2,
         dinfo1.face->diameter(),dinfo2.face->diameter());
      }
  }
}

template <int dim>
void MatrixIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  {
    const auto n_blocks = dinfo.block_info->local().size();
    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
        const auto n_components = feV.get_fe().n_components();
        const auto n_quads = feV.n_quadrature_points;
        auto &M = dinfo.matrix(b*n_blocks + b,false).matrix;

        auto xs1 = xsdata.scattering(feV.get_quadrature_points(),n_components,0,b);

        LocalIntegrators::Transport::boundary<dim>
        (M,feV,angles.get_points(),angles.get_weights(),
         xs1,dinfo.face->diameter());
      }
  }
}

// RESIDUAL INTEGRATOR
template <int dim>
ResidualIntegrator<dim>::ResidualIntegrator():
  angles("../include/integrators_data/transport/angles/D2P1K1.angles")
{}

template <int dim>
void ResidualIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  auto &localdst = dinfo.vector(0);
  const auto n_blocks = localdst.n_blocks();
  const auto &src = info.values[0];
  const auto &pre = info.values[1];
  typedef dealii::VectorSlice<typename std::remove_reference<decltype(src)>::type> SrcType ;

  XS<dim> xsdata ;

  for (unsigned int b=0; b<n_blocks-1; ++b)
    {
      const auto &feT = info.fe_values(dinfo.block_info->base_element(n_blocks-1));
      const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
      const auto n_comps = feV.get_fe().n_components();
      SrcType sliceout (src,b*n_comps,n_comps) ;
      SrcType Tslice (src,(n_blocks-1)*n_comps,1) ;
      SrcType Tpreslice (pre,(n_blocks-1)*n_comps,1) ;

      const auto total = xsdata.total(feV.get_quadrature_points(),n_comps,b) ;
      const auto abs = xsdata.absorption(feV.get_quadrature_points(),angles.get_weights(),n_comps,
					 n_blocks-1,b,1.,static_cast<double>(100 - abs_rate) / 100. /static_cast<double>(n_blocks-1));

      auto Dplanck = [&](double T)->double 
      	{
      	  return (b==0) ? Dplanck_integral(xsdata.grid[b],T) : 
      	  Dplanck_integral(xsdata.grid[b],T) - Dplanck_integral(xsdata.grid[b-1],T) ;
      	};

      LocalIntegrators::Transport::cell_residual<dim>(localdst.block(b),feV,sliceout,angles.get_points()) ;
      LocalIntegrators::Transport::total_residual<dim>(localdst.block(b),feV,sliceout,total) ;
      LocalIntegrators::Transport::Demission_residual<dim>(localdst.block(b),feV,Tpreslice,Tslice,abs,Dplanck) ;
      LocalIntegrators::Transport::T_absorption_residual<dim>(localdst.block(n_blocks-1),feT,sliceout,angles.get_weights(),abs) ;
      LocalIntegrators::Transport::DT_residual<dim>(localdst.block(n_blocks-1),feT,Tpreslice,Tslice,angles.get_weights(),abs,Dplanck);

      for (unsigned int bin=0; bin<n_blocks-1; ++bin )
        {
          const auto scattering = xsdata.scattering(feV.get_quadrature_points(),n_comps,bin,b,
                                                    static_cast<double>(100 - abs_rate) / 100. /static_cast<double>(n_blocks-1));
          SrcType slicein (src,bin*n_comps,n_comps) ;
          LocalIntegrators::Transport::redistribution_residual<dim>(localdst.block(b),feV,slicein,
                                                                    angles.get_weights(),scattering);
        }
    }
}

template <int dim>
void ResidualIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                                   dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                                   typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  {
    auto &localdst1 = dinfo1.vector(0);
    auto &localdst2 = dinfo2.vector(0);
    const auto n_blocks = localdst1.n_blocks();
    const auto &src1 = info1.values[0];
    const auto &src2 = info2.values[0];

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV1 = info1.fe_values(dinfo1.block_info->base_element(b));
        const auto &feV2 = info2.fe_values(dinfo2.block_info->base_element(b));
        const auto n_components = feV1.get_fe().n_components();
        const auto n_quads = feV1.n_quadrature_points;

        dealii::VectorSlice<typename std::remove_reference<decltype(src1)>::type> slice1 (src1,b*n_components,n_components) ;
        dealii::VectorSlice<typename std::remove_reference<decltype(src2)>::type> slice2 (src2,b*n_components,n_components) ;

        auto xs1 = xsdata.scattering(feV1.get_quadrature_points(),n_components,0,b);
        auto xs2 = xsdata.scattering(feV2.get_quadrature_points(),n_components,0,b);

        LocalIntegrators::Transport::ip_residual<dim>
        (localdst1.block(b),localdst2.block(b),
         feV1,feV2,slice1,slice2,angles.get_points(),angles.get_weights(),xs1,xs2,
         dinfo1.face->diameter(),dinfo2.face->diameter());
      }
  }
}

template <int dim>
void ResidualIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                       typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  {
    auto &localdst = dinfo.vector(0);
    const auto n_blocks = localdst.n_blocks();
    const auto &src = info.values[0];

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
        const auto n_components = feV.get_fe().n_components();
        const auto n_quads = feV.n_quadrature_points;

	auto xs1 = xsdata.scattering(feV.get_quadrature_points(),n_components,0,b);

        dealii::VectorSlice<typename std::remove_reference<decltype(src)>::type> slice (src,b*n_components,n_components) ;

        LocalIntegrators::Transport::boundary<dim>
        (localdst.block(b),feV,slice,angles.get_points(),angles.get_weights(),
         xs1,dinfo.face->diameter());
      }
  }
}

// RHS INTEGRATOR
template <int dim>
RHSIntegrator<dim>::RHSIntegrator(unsigned int n_components):
  angles("../include/integrators_data/transport/angles/D2P1K1.angles")
{}

template <int dim>
void RHSIntegrator<dim>::cell(dealii::MeshWorker::DoFInfo<dim> &dinfo, typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  {
    auto &localdst = dinfo.vector(0);
    const auto n_blocks = localdst.n_blocks();
    const auto &src = info.values[0];
    typedef dealii::VectorSlice<typename std::remove_reference<decltype(src)>::type> SrcType ;

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feT = info.fe_values(dinfo.block_info->base_element(n_blocks-1));
        const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
        const auto n_comps = feV.get_fe().n_components();
        SrcType sliceout (src,b*n_comps,n_comps) ;
        SrcType Tslice (src,(n_blocks-1)*n_comps,1) ;

        const auto total = xsdata.total(feV.get_quadrature_points(),n_comps,b) ;
        const auto abs = xsdata.absorption(feV.get_quadrature_points(),angles.get_weights(),n_comps,
                                           n_blocks-1,b,1.,static_cast<double>(100 - abs_rate) / 100. /static_cast<double>(n_blocks-1));

	auto planck = [&](double T)->double 
	  {
	    return (b==0) ? planck_integral(xsdata.grid[b],T) : 
	    ( planck_integral(xsdata.grid[b],T) - planck_integral(xsdata.grid[b-1],T) ) ;
	  };

        LocalIntegrators::Transport::cell_residual<dim>(localdst.block(b),feV,sliceout,angles.get_points()) ;
        LocalIntegrators::Transport::total_residual<dim>(localdst.block(b),feV,sliceout,total) ;
        LocalIntegrators::Transport::emission_residual<dim>(localdst.block(b),feV,Tslice,abs,planck) ;
        LocalIntegrators::Transport::T_absorption_residual<dim>(localdst.block(n_blocks-1),feT,sliceout,angles.get_weights(),abs) ;
        LocalIntegrators::Transport::T_residual<dim>(localdst.block(n_blocks-1),feT,Tslice,angles.get_weights(),abs,planck) ;
        for (unsigned int bin=0; bin<n_blocks-1; ++bin )
          {
            const auto scattering = xsdata.scattering(feV.get_quadrature_points(),n_comps,bin,b,
                                                      static_cast<double>(100 - abs_rate) / 100. /static_cast<double>(n_blocks-1));
            SrcType slicein (src,bin*n_comps,n_comps) ;
            LocalIntegrators::Transport::redistribution_residual<dim>(localdst.block(b),feV,slicein,
                                                                      angles.get_weights(),scattering);
          }
      }
  }
  {
    auto &result = dinfo.vector(0);
    const auto n_blocks = result.n_blocks();

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
        const auto n_components = feV.get_fe().n_components();

        auto f = xsdata.total(feV.get_quadrature_points(),n_components,b,1./static_cast<double>(n_blocks-1)) ;
        for (auto &component : f)
          for (auto &point : component)
            point = 1000. * eps / static_cast<double>(n_blocks-1);

        dealii::LocalIntegrators::L2::L2(result.block(b),feV,f,-1.);
      }
  }
}

template <int dim>
void RHSIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> &dinfo1,
                              dealii::MeshWorker::DoFInfo<dim> &dinfo2,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &info1,
                              typename dealii::MeshWorker::IntegrationInfo<dim> &info2) const
{
  {
    auto &localdst1 = dinfo1.vector(0);
    auto &localdst2 = dinfo2.vector(0);
    const auto n_blocks = localdst1.n_blocks();
    const auto &src1 = info1.values[0];
    const auto &src2 = info2.values[0];

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV1 = info1.fe_values(dinfo1.block_info->base_element(b));
        const auto &feV2 = info2.fe_values(dinfo2.block_info->base_element(b));
        const auto n_components = feV1.get_fe().n_components();
        const auto n_quads = feV1.n_quadrature_points;

        dealii::VectorSlice<typename std::remove_reference<decltype(src1)>::type> slice1 (src1,b*n_components,n_components) ;
        dealii::VectorSlice<typename std::remove_reference<decltype(src2)>::type> slice2 (src2,b*n_components,n_components) ;

        auto xs1 = xsdata.scattering(feV1.get_quadrature_points(),n_components,0,b);
        auto xs2 = xsdata.scattering(feV2.get_quadrature_points(),n_components,0,b);

        LocalIntegrators::Transport::ip_residual<dim>
        (localdst1.block(b),localdst2.block(b),
         feV1,feV2,slice1,slice2,angles.get_points(),angles.get_weights(),xs1,xs2,
         dinfo1.face->diameter(),dinfo2.face->diameter());
      }
  }
}

template <int dim>
void RHSIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> &dinfo,
                                  typename dealii::MeshWorker::IntegrationInfo<dim> &info) const
{
  {
    auto &localdst = dinfo.vector(0);
    const auto n_blocks = localdst.n_blocks();
    const auto &src = info.values[0];

    XS<dim> xsdata ;

    for (unsigned int b=0; b<n_blocks-1; ++b)
      {
        const auto &feV = info.fe_values(dinfo.block_info->base_element(b));
        const auto n_components = feV.get_fe().n_components();
        const auto n_quads = feV.n_quadrature_points;

        auto xs1 = xsdata.scattering(feV.get_quadrature_points(),n_components,0,b);

        dealii::VectorSlice<typename std::remove_reference<decltype(src)>::type> slice (src,b*n_components,n_components) ;

        LocalIntegrators::Transport::boundary<dim>
        (localdst.block(b),feV,slice,angles.get_points(),angles.get_weights(),
         xs1,dinfo.face->diameter());
      }
  }
}

#ifndef HEADER_IMPLEMENTATION
#include "Integrators.inst"
#endif

