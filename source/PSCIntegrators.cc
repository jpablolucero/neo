#include <PSCIntegrators.h>

// MATRIX INTEGRATOR
template <int dim>
PSCMatrixIntegrator<dim>::PSCMatrixIntegrator()
  :
  MatrixIntegrator<dim>::MatrixIntegrator()
{}

template class PSCMatrixIntegrator<2>;
template class PSCMatrixIntegrator<3>;
