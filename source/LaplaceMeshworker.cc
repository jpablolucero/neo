#include <LaplaceMeshworker.h>

int main ()
{
  dealii::deallog.depth_console (2);
  MyLaplace<2> dgmethod;
  dgmethod.run ();
  return 0;
}
