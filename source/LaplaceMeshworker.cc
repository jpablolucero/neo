#include <LaplaceMeshworker.h>

int main ()
{
  std::ofstream logfile("deallog");
  dealii::deallog.attach(logfile);
  dealii::deallog.depth_console (2);

  // stupid change!!!!!
  // Coefficient<2> coeff;
  // dealii::Point<2> p1(0.5,0.5);
  // dealii::Point<2> p2(1.0,1.0);
  // dealii::Point<2> p3(1.0,0.0);
  // std::vector<dealii::Point<2> > pvector;
  // pvector.push_back(p1);
  // pvector.push_back(p2);
  // pvector.push_back(p3);
  // std::vector<double> valvector(3);
  // coeff.value_list(pvector, valvector);
  // for (unsigned int i=0; i<pvector.size(); ++i)
  //   dealii::deallog << "value_" << i << " = " << valvector[i] << std::endl;
  MyLaplace<2,true> dgmethod;
  dgmethod.run ();
  return 0;
}
