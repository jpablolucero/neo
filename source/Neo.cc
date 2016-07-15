#include <Neo.h>

#ifndef BENCHMARKS
int main (int argc, char *argv[])
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
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    dealii::deallog.depth_console (2);
  dealii::TimerOutput timer (mpi_communicator, pcout,
                             dealii::TimerOutput::never,
                             dealii::TimerOutput::wall_times);

  const unsigned int smooth_steps = 1;


//  for (unsigned int l=2; l<7; ++l)
  {
    Simulator<2,false,2> dgmethod(timer, mpi_communicator, pcout);
    dgmethod.n_levels = 5; //atoi(argv[1]) ;
    dgmethod.smoothing_steps = smooth_steps;
    dgmethod.run ();
  }
  /*  for (unsigned int l=2; l<7; ++l)
      {
        Simulator<3,false,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }*/
  /*  for (unsigned int l=7; l<10; ++l)
      {
        Simulator<3,false,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=7; l<10; ++l)
      {
        Simulator<3,true,1> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=7; l<10; ++l)
      {
        Simulator<2,false,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=7; l<10; ++l)
      {
        Simulator<2,true,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,false,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,true,2> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=6; l<9; ++l)
      {
        Simulator<2,false,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<2,true,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,false,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,true,3> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=6; l<9; ++l)
      {
        Simulator<2,false,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<2,true,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,false,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }
    for (unsigned int l=4; l<7; ++l)
      {
        Simulator<3,true,4> dgmethod(timer, mpi_communicator, pcout);
        dgmethod.n_levels = l ;
        dgmethod.smoothing_steps = smooth_steps;
        dgmethod.run ();
      }*/
  return 0;
}
#endif



