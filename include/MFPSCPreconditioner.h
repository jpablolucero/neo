#ifndef MFPSCPRECONDITIONER_H
#define MFPSCPRECONDITIONER_H

#ifndef MATRIXFREE

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

private:
  unsigned int level;

  MatrixIntegrator<dim,false>  matrix_integrator;
  std::shared_ptr<DDHandlerBase<dim> > ddh;
};

template <int dim, typename VectorType, class number>
class MFPSCPreconditioner<dim, VectorType, number>::AdditionalData
{
public:
  AdditionalData() : dof_handler(0), level(-1), weight(1.0), mapping(0), patch_type(cell_patches) {}

  dealii::DoFHandler<dim> *dof_handler;
  unsigned int level;
  double weight;
  const dealii::Mapping<dim> *mapping;

  enum PatchType
  {
    cell_patches,
    vertex_patches
  };
  PatchType patch_type;
};

template <int dim, typename VectorType, class number>
template <class GlobalOperatorType>
void MFPSCPreconditioner<dim, VectorType, number>::initialize(const GlobalOperatorType & /*global_operator*/,
    const AdditionalData &data)
{
  Assert(data.dof_handler != 0, dealii::ExcInternalError());
  Assert(data.level != -1, dealii::ExcInternalError());
  Assert(data.mapping != 0, dealii::ExcInternalError());

  this->data = data;
  level = data.level;
  const dealii::DoFHandler<dim> &dof_handler = *(data.dof_handler);

  if (data.patch_type == AdditionalData::PatchType::cell_patches)
    ddh.reset(new DGDDHandlerCell<dim>());
  else
    ddh.reset(new DGDDHandlerVertex<dim>());
  ddh->initialize(dof_handler, level);
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

#endif // MATRIXFREE OFF
#endif // MFPSCPRECONDITIONER_H
