#include <LaplaceMeshworker.h>

int main (int argc, char *argv[])
{
  try
    {
#if PARALLEL_LA == 1
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
#else
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
#endif
      std::ofstream logfile("deallog");
      dealii::deallog.attach(logfile);
      dealii::deallog.depth_console (2);
      MyLaplace<2> dgmethod;
      dgmethod.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}



