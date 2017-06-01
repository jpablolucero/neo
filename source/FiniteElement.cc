#include <FiniteElement.h>

template <int dim>
FiniteElement<dim>::FiniteElement(unsigned int degree):
  mapping (),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1)
#else
  fe(dealii::FE_DGQ<dim>(degree),1)
#endif
{}

template class FiniteElement<2>;
template class FiniteElement<3>;
