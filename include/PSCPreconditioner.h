#ifndef PSCPRECONDITIONER_H
#define PSCPRECONDITIONER_H

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
class PSCPreconditioner final
{
public:
  typedef typename dealii::FullMatrix<double> Matrix;
  class AdditionalData;

  PSCPreconditioner();
  PSCPreconditioner (const PSCPreconditioner &) = delete ;
  PSCPreconditioner &operator = (const PSCPreconditioner &) = delete;

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
class PSCPreconditioner<dim, VectorType, number>::AdditionalData
{
public:
  AdditionalData() : ddh(0), weight(1.0) {}

  std::map<typename dealii::DoFHandler<dim>::level_cell_iterator,const Matrix * > local_matrices;
  const DDHandlerBase<dim> *ddh;
  double weight;
};

template <int dim, typename VectorType, class number>
template <class GlobalOperatorType>
void PSCPreconditioner<dim, VectorType, number>::initialize(const GlobalOperatorType & /*global_operator*/,
                                                            const AdditionalData &data)
{
  Assert(data.ddh != 0, dealii::ExcInternalError());
//  Assert(data.local_inverses.size() == data.ddh->size(),
//         dealii::ExcDimensionMismatch(data.local_inverses.size(),data.ddh->size()));
  this->data = data;
  // from the DDHandler we get the patches and the corresponding cells
  // use these to construct the matrices we want to smoothen with
  // from the local matrices)
//  std::vector<typename dealii::DoFHandler<dim>::level_cell_iterator> &patches = data.ddh->subdomain_to_global_map;
}

template <int dim, typename VectorType, class number>
void PSCPreconditioner<dim, VectorType, number>::clear()
{}

template <int dim, typename VectorType, class number>
dealii::TimerOutput *
PSCPreconditioner<dim, VectorType, number>::timer;

#ifdef HEADER_IMPLEMENTATION
#include <PSCPreconditioner.cc>
#endif

#endif
