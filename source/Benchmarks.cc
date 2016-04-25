#include <Benchmarks.h>

#ifdef BENCHMARKS

void Neo2d(benchmark::State &state)
{
  while (state.KeepRunning())
    {
      dealii::MultithreadInfo::set_thread_limit(state.range_x());
      MPI_Comm mpi_communicator (MPI_COMM_WORLD);
      std::ofstream   fout("/dev/null");
      std::cout.rdbuf(fout.rdbuf());
      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      std::ofstream logfile("deallog");
      dealii::deallog.attach(logfile);
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
        dealii::deallog.depth_console (0);
      dealii::TimerOutput timer (mpi_communicator, pcout,
                                 dealii::TimerOutput::never,
                                 dealii::TimerOutput::wall_times);
      Simulator<2,false,1> dgmethod(timer, mpi_communicator, pcout);
      dgmethod.n_levels = state.range_y();
      dgmethod.run ();
    }
}

void Neo3d(benchmark::State &state)
{
  while (state.KeepRunning())
    {
      dealii::MultithreadInfo::set_thread_limit(state.range_x());
      MPI_Comm mpi_communicator (MPI_COMM_WORLD);
      std::ofstream   fout("/dev/null");
      std::cout.rdbuf(fout.rdbuf());
      dealii::ConditionalOStream pcout(std::cout,
                                       dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      std::ofstream logfile("deallog");
      dealii::deallog.attach(logfile);
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
        dealii::deallog.depth_console (0);
      dealii::TimerOutput timer (mpi_communicator, pcout,
                                 dealii::TimerOutput::never,
                                 dealii::TimerOutput::wall_times);
      Simulator<2,false,1> dgmethod(timer, mpi_communicator, pcout);
      dgmethod.n_levels = state.range_y();
      dgmethod.run ();
    }
}

BENCHMARK(Neo2d)
->Threads(1)
->ArgPair(1,2)->ArgPair(2,2)->ArgPair(4,2)->ArgPair(8,2)->ArgPair(16,2)
->ArgPair(1,4)->ArgPair(2,4)->ArgPair(4,4)->ArgPair(8,4)->ArgPair(16,4)
->ArgPair(1,6)->ArgPair(2,6)->ArgPair(4,6)->ArgPair(8,6)->ArgPair(16,6)
->UseRealTime();

BENCHMARK(Neo3d)
->Threads(1)
->ArgPair(1,2)->ArgPair(2,2)->ArgPair(4,2)->ArgPair(8,2)->ArgPair(16,2)
->ArgPair(1,4)->ArgPair(2,4)->ArgPair(4,4)->ArgPair(8,4)->ArgPair(16,4)
->UseRealTime();

BENCHMARK_MAIN()

#endif



