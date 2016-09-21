#ifndef DDSMOOTHER_H
#define DDSMOOTHER_H

#include <deal.II/base/timer.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <functional>

#include <GenericLinearAlgebra.h>
#include <MfreeOperator.h>
#include <Integrators.h>

template <int dim, int fe_degree, typename VectorType=LA::MPI::Vector, typename number=double, bool same_diagonal=true>
class DDSmoother final : public dealii::Subscriptor
{
public:
  //  typedef typename dealii::LAPACKFullMatrix<double> LAPACKMatrix;
  class AdditionalData;

  DDSmoother ();
  DDSmoother (const DDSmoother &) = delete ;
  DDSmoother &operator = (const DDSmoother &) = delete;

  // interface for MGSmootherPrecondition but global_operator is not used
  template <typename GlobalOperatorType>
  void initialize(const GlobalOperatorType &global_operator,
                  const AdditionalData &data);
  void clear();

  void vmult(VectorType &dst, const VectorType &src) const;

  void Tvmult(VectorType &dst, const VectorType &src) const;

  void vmult_add(VectorType &dst, const VectorType &src) const;

  void Tvmult_add(VectorType &dst, const VectorType &src) const;

  static dealii::TimerOutput *timer;

protected:
  AdditionalData addit_data;

private:
  void smooth (const dealii::MatrixFree<dim,number>             &data,
               VectorType                                       &dst,
               const VectorType                                 &src,
               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  unsigned int level;
  std::shared_ptr<dealii::AlignedVector<dealii::VectorizedArray<number> > > single_inverse ;
};

template <int dim, int fe_degree, typename VectorType, class number, bool same_diagonal>
class DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::AdditionalData
{
public:
  AdditionalData()
    :
    level(-1),
    relaxation(1.0),
    matrixfree_data(0),
    mapping(0)
  {}

  unsigned int                              level;
  double                                    relaxation;
  const dealii::MatrixFree<dim,number>      *matrixfree_data;
  const dealii::Mapping<dim>                *mapping;
};

template <int dim, int fe_degree, typename VectorType, class number, bool same_diagonal>
dealii::TimerOutput *
DDSmoother<dim,fe_degree,VectorType,number,same_diagonal>::timer;

#ifdef HEADER_IMPLEMENTATION
#include <DDSmoother.cc>
#endif

#endif // DDSMOOTHER_H
