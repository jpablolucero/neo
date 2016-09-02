#include <PSCIntegrators.h>

#ifdef MATRIXFREE
template <int dim>
PSCMatrixIntegrator<dim>::PSCMatrixIntegrator()
  :
  MatrixIntegrator<dim>::MatrixIntegrator()
{}

#else // MATRIXFREE OFF
template <int dim>
PSCMatrixIntegrator<dim>::PSCMatrixIntegrator()
{}

#endif // MATRIXFREE

template class PSCMatrixIntegrator<2>;
template class PSCMatrixIntegrator<3>;
