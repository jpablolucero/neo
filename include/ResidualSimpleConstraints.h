#ifndef RESIDUALSIMPLECONSTRAINTS_H
#define RESIDUALSIMPLECONSTRAINTS_H

#include <deal.II/meshworker/simple.h>

template <typename VectorType>
class ResidualSimpleConstraints : private dealii::MeshWorker::Assembler::ResidualSimple<VectorType>
{
public:
  /**
   * Initialize with an AnyData object holding the result of assembling.
   *
   * Assembling currently writes into the first vector of
   * <tt>results</tt>.
   */
  void initialize(dealii::AnyData &results);

  /**
   * Initialize the constraints.
   */
  void initialize(const dealii::ConstraintMatrix &constraints);

  /**
  * Initialize the local data in the DoFInfo object used later for
  * assembling.
  *
  * The info object refers to a cell if <code>!face</code>, or else to an
  * interior or boundary face.
  */
  template <class DOFINFO>
  void initialize_info(DOFINFO &info, bool face) const;

  /**
   * Assemble the local residuals into the global residuals.
   *
   * Values are added to the previous contents. If constraints are active,
   * ConstraintMatrix::distribute_local_to_global() is used.
   */
  template <class DOFINFO>
  void assemble(const DOFINFO &info);

  /**
   * Assemble both local residuals into the global residuals.
   */
  template<class DOFINFO>
  void assemble(const DOFINFO &info1,
                const DOFINFO &info2);
private:

  /**
   * The global residal vectors filled by assemble().
   */
  dealii::AnyData residuals;
  /**
   * A pointer to the object containing constraints.
   */
  dealii::SmartPointer<const dealii::ConstraintMatrix,ResidualSimpleConstraints<VectorType> > constraints;
};


template <typename VectorType>
inline void
ResidualSimpleConstraints<VectorType>::initialize(dealii::AnyData &results)
{
  residuals = results;
}


template <typename VectorType>
inline void
ResidualSimpleConstraints<VectorType>::initialize(const dealii::ConstraintMatrix &c)
{
  constraints = &c;
}


template <typename VectorType >
template <class DOFINFO>
inline void
ResidualSimpleConstraints<VectorType>::initialize_info(DOFINFO &info, bool face) const
{
  info.initialize_vectors(residuals.size());
  info.initialize_matrices(1, face);
}



template <typename VectorType>
template <class DOFINFO>
inline void
ResidualSimpleConstraints<VectorType>::assemble(const DOFINFO &info)
{
  Assert(!info.level_cell, dealii::ExcMessage("Cell may not access level dofs"));

  for (unsigned int k=0; k<residuals.size(); ++k)
    {
      VectorType *v = residuals.entry<VectorType *>(k);
      const dealii::FullMatrix<double> &local_matrix= info.matrix(k, false).matrix;
      for (unsigned int i=0; i != info.vector(k).n_blocks(); ++i)
        {
          const std::vector<dealii::types::global_dof_index>  &ldi = info.vector(k).n_blocks()==1?
                                                                     info.indices:
                                                                     info.indices_by_block[i];

          if (constraints !=0)
            constraints->distribute_local_to_global(info.vector(k).block(i), ldi, *v, local_matrix);
          else
            v->add(ldi, info.vector(k).block(i));
        }
    }
}



template <typename VectorType>
template <class DOFINFO>
inline void
ResidualSimpleConstraints<VectorType>::assemble(const DOFINFO &info1, const DOFINFO &info2)
{
  //Nothing different is done for flux terms
  dealii::MeshWorker::Assembler::ResidualSimple<VectorType>::assemble(info1, info2);
}
#endif
