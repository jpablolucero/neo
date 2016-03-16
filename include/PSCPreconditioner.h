#ifndef PSCPRECONDITIONER_H
#define PSCPRECONDITIONER_H

#include <DDHandler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

template <int dim, class number>
class PSCPreconditioner
{
  public:
    typedef typename dealii::FullMatrix<number> Matrix;
    class AdditionalData;

    PSCPreconditioner();

    // interface for MGSmootherPrecondition but global_operator is not used
    template <class GlobalOperatorType>
      void initialize(const GlobalOperatorType& global_operator,
                      const AdditionalData& data);
    void clear();
      
    void vmult(dealii::Vector<number> &dst,
               const dealii::Vector<number> &src) const;
    void Tvmult(dealii::Vector<number> &dst,
                const dealii::Vector<number> &src) const;
    void vmult_add(dealii::Vector<number> &dst,
                   const dealii::Vector<number> &src) const;
    void Tvmult_add(dealii::Vector<number> &dst,
                    const dealii::Vector<number> &src) const;

  protected:
    AdditionalData data;
};

template <int dim, class number>
class PSCPreconditioner<dim, number>::AdditionalData
{
  public:
    AdditionalData() : local_inverses(0), ddh(0), weight(1.0) {}

    std::vector<const Matrix*> local_inverses;
    const DDHandlerBase<dim, number>* ddh;
    double weight;
};

template <int dim, class number>
template <class GlobalOperatorType>
void PSCPreconditioner<dim, number>::initialize(
    const GlobalOperatorType& /*global_operator*/,
    const AdditionalData& data)
{
  assert(data.ddh != 0);
  assert(data.local_inverses.size() == data.ddh->size());
  this->data = data;
}

template <int dim, class number>
void PSCPreconditioner<dim, number>::clear()
{}

#endif
