#include <deal.II/base/utilities.h>
#include <MyLaplace.h>

int main (int argc, char *argv[])
{
  try
    {
#if PARALLEL_LA == 1
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
#else
#if MAXTHREADS==0
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
#else
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, MAXTHREADS);
#endif
#if PARALLEL_LA == 0
      const unsigned int n_proc = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      AssertThrow (n_proc==1,
                   dealii::ExcMessage("If no parallel linear algebra is used, only one MPI process may be used!"));
#endif
#endif
      MPI_Comm mpi_communicator (MPI_COMM_WORLD);
      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      std::ofstream logfile("deallog");
      dealii::deallog.attach(logfile);
      //lots of output only on the first process
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
        dealii::deallog.depth_console (2);
      dealii::TimerOutput timer (mpi_communicator, pcout,
                                 dealii::TimerOutput::never,
                                 dealii::TimerOutput::wall_times);
      {
        MyLaplace<2,true,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      /*{
        MyLaplace<2,true,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<2,true,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<2,true,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }*/
      {
        MyLaplace<2,false,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      /*{
        MyLaplace<2,false,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<2,false,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<2,false,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,true,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,true,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,true,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,true,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }*/
      /*{
        MyLaplace<3,false,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }*/
      /*{
        MyLaplace<3,false,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,false,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }
      {
        MyLaplace<3,false,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.run ();
      }*/
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



