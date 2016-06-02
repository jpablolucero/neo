#ifndef MFPSCPRECONDITIONER_H
#define MFPSCPRECONDITIONER_H

#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/distributed/tria.h>

#include <functional>

#include <GenericLinearAlgebra.h>
#include <DDHandler.h>
#include <MFOperator.h>

template <int dim=2, typename VectorType=LA::MPI::Vector, class number=double>
class MFPSCPreconditioner final
{
public:
  typedef typename dealii::FullMatrix<double> Matrix;
  class AdditionalData;

  MFPSCPreconditioner();
  MFPSCPreconditioner (const MFPSCPreconditioner &) = delete ;
  MFPSCPreconditioner &operator = (const MFPSCPreconditioner &) = delete;

  // interface for MGSmootherPrecondition but global_operator is not used
  template <class GlobalOperatorType>
  void initialize(const GlobalOperatorType &global_operator,
                  const AdditionalData &data);
  void clear();

  void vmult(VectorType &dst, const VectorType &src) const;

  void Tvmult(VectorType &dst, const VectorType &src) const;

  void vmult_add(VectorType &dst, const VectorType &src) const;

  void Tvmult_add(VectorType &dst, const VectorType &src) const;

  static dealii::TimerOutput *timer;

protected:
  AdditionalData data;
};

template <int dim, typename VectorType, class number>
class MFPSCPreconditioner<dim, VectorType, number>::AdditionalData
{
public:
  AdditionalData() : ddh(0), weight(1.0) {}

  const DDHandlerBase<dim> *ddh;
  double weight;
};

template <int dim, typename VectorType, class number>
template <class GlobalOperatorType>
void MFPSCPreconditioner<dim, VectorType, number>::initialize(const GlobalOperatorType & /*global_operator*/,
    const AdditionalData &data)
{
  Assert(data.ddh != 0, dealii::ExcInternalError());
  this->data = data;
}

template <int dim, typename VectorType, class number>
void MFPSCPreconditioner<dim, VectorType, number>::clear()
{}

template <int dim, typename VectorType, class number>
dealii::TimerOutput *
MFPSCPreconditioner<dim, VectorType, number>::timer;

#ifdef HEADER_IMPLEMENTATION
#include <MFPSCPreconditioner.cc>
#endif

#endif
