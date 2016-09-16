#ifndef MGTRANSFERMF_H
#define MGTRANSFERMF_H

// Is it possible to use this or something similar for Trilinos
namespace dealii
{
  template <int dim, typename LOPERATOR>
  class MGTransferMF : public dealii::MGTransferMatrixFree<dim, typename LOPERATOR::value_type>
  {
  public:
    MGTransferMF(const MGLevelObject<LOPERATOR> &op)
      :
      mg_operator (op)
    {};

    // Overload of copy_to_mg from MGLevelGlobalTransfer
    template <class InVector, int spacedim>
    void
    copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof,
                MGLevelObject<dealii::parallel::distributed::Vector<typename LOPERATOR::value_type> > &dst,
                const InVector &src) const
    {
      for (unsigned int level=dst.min_level();
           level<=dst.max_level(); ++level)
        mg_operator[level].initialize_dof_vector(dst[level]);
      dealii::MGLevelGlobalTransfer
      <dealii::parallel::distributed::Vector<typename LOPERATOR::value_type> >::copy_to_mg(mg_dof, dst, src);
    }

  private:
    const MGLevelObject<LOPERATOR> &mg_operator;
  };
}

#endif // MGTRANSFERMF_H
