#include <FiniteElement.h>

extern std::unique_ptr<dealii::ConditionalOStream> pcout ;

template <int dim>
FiniteElement<dim>::FiniteElement(unsigned int degree):
  mapping (degree + 1),
#ifdef CG
  fe(dealii::FE_Q<dim>(degree),1)
#else
  fe(dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree),4),10,
     dealii::FESystem<dim>(dealii::FE_DGQ<dim>(degree),1),1)
#endif
{}

template class FiniteElement<2>;
template class FiniteElement<3>;
