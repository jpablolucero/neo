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

private:
  std::vector<Matrix> patch_inverses;
};

template <int dim, typename VectorType, class number>
class PSCPreconditioner<dim, VectorType, number>::AdditionalData
{
public:
  AdditionalData() : ddh(0), weight(1.0) {}

  std::map<int, const Matrix * > local_matrices;
  const DDHandlerBase<dim> *ddh;
  double weight;
};

template <int dim, typename VectorType, class number>
template <class GlobalOperatorType>
void PSCPreconditioner<dim, VectorType, number>::initialize(const GlobalOperatorType & /*global_operator*/,
                                                            const AdditionalData &data)
{
  Assert(data.ddh != 0, dealii::ExcInternalError());
  this->data = data;
  // from the DDHandler we get the patches and the corresponding cells
  // use these to construct the matrices we want to smoothen with
  // from the local matrices)
  const unsigned int n_dofs_per_cell = data.ddh->get_dofh().get_fe().n_dofs_per_cell();
  for (unsigned int i=0; i<data.ddh->subdomain_to_global_map.size(); ++i)
    {
      for (unsigned int j=0; j<data.ddh->subdomain_to_global_map[i].size(); ++j)
        {
          const typename dealii::DoFHandler<dim>::level_cell_iterator cell
            = data.ddh->subdomain_to_global_map[i][j];
          const std::vector<unsigned int> local_to_patch
          (data.ddh->all_to_unique[i].begin()+n_dofs_per_cell*j,
           data.ddh->all_to_unique[j].begin()+n_dofs_per_cell*(j+1));
          data.local_matrices.at(cell->index())->scatter_matrix_to(local_to_patch,
                                                                   local_to_patch,
                                                                   patch_inverses[i]);
        }
      //invert patch_matrix
      patch_inverses[i].gauss_jordan();
    }
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
