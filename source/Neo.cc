#include <Neo.h>

#ifndef BENCHMARKS

std::unique_ptr<dealii::TimerOutput>        timer ;
std::unique_ptr<MPI_Comm>                   mpi_communicator ;
std::unique_ptr<dealii::ConditionalOStream> pcout ;

int main (int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  mpi_communicator.reset(new MPI_Comm(MPI_COMM_WORLD));
  pcout.reset(new dealii::ConditionalOStream(std::cout,dealii::Utilities::MPI::this_mpi_process(*mpi_communicator) == 0));
  std::ofstream logfile("deallog");
  dealii::deallog.attach(logfile);
  if (dealii::Utilities::MPI::this_mpi_process(*mpi_communicator)==0)
    dealii::deallog.depth_console (3);
  timer.reset(new dealii::TimerOutput (*mpi_communicator, *pcout,dealii::TimerOutput::never,dealii::TimerOutput::wall_times));
  const unsigned int d = 2 ;
  const unsigned int fe_degree = 1 ;
  typedef MFOperator<d,fe_degree,double> SystemMatrixType;
  typedef GMGPreconditioner<d>  Precond;
  Simulator<SystemMatrixType,dealii::parallel::distributed::Vector<double>,Precond,d,fe_degree> dgmethod;
  dgmethod.n_levels = (argc > 1) ? atoi(argv[1]) : 2 ;
  dgmethod.min_level=0;
  dgmethod.smoothing_steps = (argc > 2) ? atoi(argv[2]) : 1 ;
  dgmethod.run ();
  return 0;
}
#endif



