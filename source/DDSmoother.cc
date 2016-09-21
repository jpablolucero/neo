#include <DDSmoother.h>

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::DDSmoother()
  :
  dealii::Subscriptor(),
  level(0)
{}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
template <typename GlobalOperatorType>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::initialize(const GlobalOperatorType & /*global_operator*/,
    const AdditionalData     &addit_data_)
{
  // timer->enter_subsection("DDSmoother::initialize");
  // TODO we need fe_degree template parameter
  Assert(addit_data_.level != dealii::numbers::invalid_unsigned_int, dealii::ExcInternalError());

  this->addit_data = addit_data_;
  level = addit_data.level;
  auto &cell_partition_data = addit_data.matrixfree_data->get_task_info().cell_partition_data;

  // SAME_DIAGONAL
  if ( same_diagonal && (cell_partition_data.front()<cell_partition_data.back()) )
    {
      dealii::Triangulation<dim>             local_triangulation;
      dealii::DoFHandler<dim>                local_dof_handler (local_triangulation);
      auto                                   &fe = addit_data.matrixfree_data->get_dof_handler(0).get_fe();
      const dealii::QGauss<1>                quad (fe_degree+1);
      SameDiagIntegrator<dim,fe_degree,fe_degree+1,1,number>   integrator;
      dealii::ConstraintMatrix               dummy_constraints;

      double h = 1./std::pow(2.,level);
      dealii::GridGenerator::hyper_cube (local_triangulation,0.,h);
      local_dof_handler.distribute_dofs (fe);

      dealii::MatrixFree<dim,number>         local_mf_data;
      typename dealii::MatrixFree<dim,double>::AdditionalData mfaddit_data;
      mfaddit_data.tasks_parallel_scheme = dealii::MatrixFree<dim,double>::AdditionalData::none;
#ifndef CG
      mfaddit_data.build_face_info = true;
#endif // CG 
      dummy_constraints.close();
      local_mf_data.reinit (*addit_data.mapping,local_dof_handler,dummy_constraints,quad,mfaddit_data);

      // // debug output
      // auto &cell_partition_data = mf_data.get_task_info().cell_partition_data;
      // for( auto &part : cell_partition_data )
      //    std::cout << part << " | " ;
      // std::cout << std::endl;

      // timer->enter_subsection("DDSmoother::__assemble(sd)");
      dealii::FullMatrix<number> matrix(local_dof_handler.n_dofs());
      std::pair<int,int> column_level_pair;
      for (int i=0; i<local_dof_handler.n_dofs(); ++i)
        {
          column_level_pair = std::make_pair(i,level);
          dealii::Vector<number> dst(local_dof_handler.n_dofs());
          local_mf_data.loop(&SameDiagIntegrator<dim,fe_degree,fe_degree+1,1,number>::cell,
                             &SameDiagIntegrator<dim,fe_degree,fe_degree+1,1,number>::face,
                             &SameDiagIntegrator<dim,fe_degree,fe_degree+1,1,number>::boundary,
                             &integrator,
                             dst,
                             column_level_pair);
          for (int j=0; j<local_dof_handler.n_dofs(); ++j)
            matrix(j,i) = dst(j);
        }
      // timer->leave_subsection();

      // timer->enter_subsection("DDSmoother::__invert(sd)");
      matrix.gauss_jordan();
      // timer->leave_subsection();
      // std::cout << "MATRIXFREE:- SAME DIAG matrix on level: " << level << std::endl;
      // matrix.print_formatted(std::cout);
      // std::cout << std::endl;

      // timer->enter_subsection("DDSmoother::__write(sd)");
      single_inverse.reset(new dealii::AlignedVector<dealii::VectorizedArray<number> >);
      for (int i=0; i<local_dof_handler.n_dofs(); ++i)
        for (int j=0; j<local_dof_handler.n_dofs(); ++j)
          single_inverse->push_back(dealii::make_vectorized_array(matrix(i,j)));
      // timer->leave_subsection();
    } // if same_diagonal
  else if ( !same_diagonal )
    AssertThrow(false,dealii::ExcNotImplemented());

  // timer->leave_subsection();
}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::smooth (const dealii::MatrixFree<dim,number>             &data,
    VectorType  &dst,
    const VectorType  &src,
    const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  // // debug output
  // std::cout << "src: " << std::endl;
  // src.print(std::cout);
  // std::cout << std::endl;
  dealii::FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);

      dealii::VectorizedArray<number> local_vector[phi.tensor_dofs_per_cell];
      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        local_vector[i] = dealii::make_vectorized_array(0.);

      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
          local_vector[i] += phi.begin_dof_values()[j] * ((*single_inverse)[i*phi.dofs_per_cell+j]);

      // debug output
      // std::cout << "dof values after mult" << std::endl;
      // for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
      //  std::cout << "[" << local_vector[i][0] << "][" << phi.begin_dof_values()[i][1] << "]" << std::endl;
      // std::cout << std::endl;

      for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_vector[i];

      phi.distribute_local_to_global(dst);
    }
  // // debug output
  // std::cout << "dst: " << std::endl;
  // dst.print(std::cout);
  // std::cout << std::endl;
}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::clear()
{}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::vmult (VectorType       &dst,
    const VectorType &src) const
{
#if PARALLEL_LA ==3
  dst = 0;
  dst.compress(dealii::VectorOperation::insert);
  vmult_add(dst, src);
  dst.compress(dealii::VectorOperation::add);

  // // debug output
  // std::cout << "dst before relax: " << std::endl;
  // dst.print(std::cout);
  // std::cout << std::endl;

  dst *= addit_data.relaxation;
  dst.compress(dealii::VectorOperation::add);

  // // debug output
  // std::cout << "dst after relax: " << std::endl;
  // dst.print(std::cout);
  // std::cout << std::endl;

  AssertIsFinite(dst.l2_norm());
#else
  AssertThrow(false, dealii::ExcNotImplemented());
#endif // PARALLEL_LA
}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::Tvmult (VectorType       &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::vmult_add (VectorType       &dst,
    const VectorType &src) const
{
  timer->enter_subsection("DDSmoother::smooth ("+ dealii::Utilities::int_to_string(level)+ ")");
#if PARALLEL_LA == 3
  Assert(dst.partitioners_are_globally_compatible(*addit_data.matrixfree_data->get_dof_info(0).vector_partitioner),
         dealii::ExcInternalError());
  Assert(src.partitioners_are_globally_compatible(*addit_data.matrixfree_data->get_dof_info(0).vector_partitioner),
         dealii::ExcInternalError());

  addit_data.matrixfree_data->cell_loop
  (&DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::smooth,
   this,
   dst,
   src);
#else
  AssertThrow(false, dealii::ExcNotImplemented());
#endif // PARALLEL_LA
  timer->leave_subsection();
}

template <int dim, int fe_degree, typename VectorType, typename number, bool same_diagonal>
void
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::Tvmult_add (VectorType       &/*dst*/,
    const VectorType &/*src*/) const
{
  // TODO use transpose of local inverses
  AssertThrow(false, dealii::ExcNotImplemented());
}

#include "DDSmoother.inst"
