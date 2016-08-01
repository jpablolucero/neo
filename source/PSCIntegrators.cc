#include <PSCIntegrators.h>

// MATRIX INTEGRATOR
template <int dim,bool same_diagonal>
PSCMatrixIntegrator<dim,same_diagonal>::PSCMatrixIntegrator()
{}

template class PSCMatrixIntegrator<2,false>;
template class PSCMatrixIntegrator<3,false>;
template class PSCMatrixIntegrator<2,true>;
template class PSCMatrixIntegrator<3,true>;
