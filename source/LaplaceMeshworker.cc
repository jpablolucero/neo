#include <LaplaceMeshworker.h>

int main ()
{
  std::ofstream logfile("deallog");
  dealii::deallog.attach(logfile);
  dealii::deallog.depth_console (2);

  MyLaplace<2,false> dgmethod;
  dgmethod.run ();
  return 0;
}
