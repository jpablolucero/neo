#ifndef GENERIC_LINEAR_ALGEBRA_H
#define GENERIC_LINEAR_ALGEBRA_H

#include <deal.II/base/config.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#if PARALLEL_LA==0
namespace dealii
{
  namespace LinearAlgebraDealII
  {
    using namespace dealii;
    typedef SolverCG<Vector<double> > SolverCG;
    namespace MPI
    {
      typedef Vector<double> Vector;
      typedef BlockVector<double> BlockVector;

      typedef SparseMatrix<double>                          SparseMatrix;
      typedef types::global_dof_index                       SparseMatrixSizeType ;

      typedef PreconditionSSOR<SparseMatrix > PreconditionSSOR;
    }
  }
}

#elif PARALLEL_LA==1

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_block_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

namespace dealii
{
  namespace LinearAlgebraPETSc
  {
    using namespace dealii;

    typedef PETScWrappers::SparseMatrix SparseMatrix;

    typedef PETScWrappers::SolverCG SolverCG;
    typedef PETScWrappers::SolverGMRES SolverGMRES;
    typedef PETScWrappers::SolverBicgstab SolverBicgstab;
    typedef PETScWrappers::SolverCGS SolverCGS;
    typedef PETScWrappers::SolverTFQMR SolverTFQMR;
    typedef PETScWrappers::SparseDirectMUMPS SolverDirect;

    typedef PETScWrappers::PreconditionBoomerAMG PreconditionAMG;
    typedef PETScWrappers::PreconditionICC PreconditionIC;
    typedef PETScWrappers::PreconditionILU PreconditionILU;
    typedef PETScWrappers::PreconditionJacobi PreconditionJacobi;
    typedef PETScWrappers::PreconditionSSOR PreconditionSSOR;
    typedef PETScWrappers::PreconditionSOR PreconditionSOR;
    typedef PETScWrappers::PreconditionBlockJacobi PreconditionBlockJacobi;
    typedef PETScWrappers::PreconditionNone PreconditionIdentity;

    namespace MPI
    {
      typedef PETScWrappers::MPI::Vector Vector;
      typedef PETScWrappers::MPI::BlockVector BlockVector;

      typedef PETScWrappers::MPI::SparseMatrix SparseMatrix;
      typedef PETScWrappers::MPI::BlockSparseMatrix BlockSparseMatrix;

      typedef dealii::BlockDynamicSparsityPattern BlockSparsityPattern;

      typedef PETScWrappers::SparseMatrix::size_type      SparseMatrixSizeType ;
    }
  }
}

#else//DEAL_II_USE_TRILINOS

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_solver.h>

namespace dealii
{
  namespace LinearAlgebraTrilinos
  {
    using namespace dealii;

    typedef TrilinosWrappers::SolverCG SolverCG;

    typedef TrilinosWrappers::SolverCG SolverCG;
    typedef TrilinosWrappers::SolverGMRES SolverGMRES;
    typedef TrilinosWrappers::SolverBicgstab SolverBicgstab;
    typedef TrilinosWrappers::SolverCGS SolverCGS;
    typedef TrilinosWrappers::SolverTFQMR SolverTFQMR;
    typedef TrilinosWrappers::SolverDirect SolverDirect;

    typedef TrilinosWrappers::PreconditionAMG PreconditionAMG;
    typedef TrilinosWrappers::PreconditionIC PreconditionIC;
    typedef TrilinosWrappers::PreconditionILU PreconditionILU;
    typedef TrilinosWrappers::PreconditionJacobi PreconditionJacobi;
    typedef TrilinosWrappers::PreconditionSSOR PreconditionSSOR;
    typedef TrilinosWrappers::PreconditionSOR PreconditionSOR;
    typedef TrilinosWrappers::PreconditionBlockJacobi PreconditionBlockJacobi;
    typedef TrilinosWrappers::PreconditionIdentity PreconditionIdentity;

    namespace MPI
    {
      typedef TrilinosWrappers::MPI::Vector Vector;
      typedef TrilinosWrappers::MPI::BlockVector BlockVector;

      typedef TrilinosWrappers::SparseMatrix SparseMatrix;
      typedef TrilinosWrappers::BlockSparseMatrix BlockSparseMatrix;

      typedef TrilinosWrappers::BlockSparsityPattern BlockSparsityPattern;

      typedef TrilinosWrappers::SparseMatrix::const_iterator SparseMatrixConstIterator;
      typedef TrilinosWrappers::SparseMatrix::size_type      SparseMatrixSizeType ;
    }
  }
}
#endif//PARALLEL_LA

namespace LA
{
#if PARALLEL_LA == 0
  using namespace dealii::LinearAlgebraDealII;
#elif PARALLEL_LA == 1
  using namespace dealii::LinearAlgebraPETSc;
#else
  using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#endif//GENERIC_LINEAR_ALGEBRA_H
