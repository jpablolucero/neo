#include <FiniteElement.h>

extern std::unique_ptr<dealii::ConditionalOStream> pcout ;

template <int dim>
FiniteElement<dim>::FiniteElement(unsigned int degree):
  mapping (),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1)
#else
  fe(dealii::FE_DGQ<dim>(degree),1)
#endif
{
#ifdef CG
  *pcout<< "Using FE_Q elements" << std::endl;
#else
  *pcout<< "Using FE_DGQ elements" << std::endl;
#endif //CG
}

template class FiniteElement<2>;
template class FiniteElement<3>;
