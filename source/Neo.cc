#include <Neo.h>

#ifndef BENCHMARKS

std::unique_ptr<dealii::TimerOutput>        timer ;
std::unique_ptr<MPI_Comm>                   mpi_communicator ;
std::unique_ptr<dealii::ConditionalOStream> pcout ;

double eps = 1. ;
double abs_rate = 0 ;
double max_energy = 1. ;
double flux_guess = 1. ;
double temp_guess = 1. ;

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
  typedef dealii::parallel::distributed::Vector<double> VectorType;
  typedef dealii::parallel::distributed::Vector<double> VectorType;
  typedef MFOperator<d,fe_degree,double> SystemMatrixType;
  typedef PSCPreconditioner<d,SystemMatrixType,dealii::TrilinosWrappers::SparseMatrix> Smoother;
  typedef GMGPreconditioner<d,VectorType,double,false,fe_degree,Smoother,SystemMatrixType,Smoother>  Precond;
  Simulator<SystemMatrixType,VectorType,Precond,d,fe_degree> dgmethod;
  dgmethod.n_levels = (argc > 1) ? atoi(argv[1]) : 2 ;
  eps = (argc > 2) ? std::pow(10.,atoi(argv[2])) : 1. ;
  abs_rate = (argc > 3) ? atof(argv[3]) : 0 ;
  max_energy = (argc > 4) ? atof(argv[4]) : 0 ;
  flux_guess = (argc > 5) ? atof(argv[5]) : 1. ;
  temp_guess = (argc > 6) ? atof(argv[6]) : 1. ;
  dgmethod.aspin = (argc > 7) ? atoi(argv[7]) : 0 ; 
  dealii::deallog << "eps = " << eps << std::endl ;
  dealii::deallog << "abs_rate = " << abs_rate << std::endl ;
  dealii::deallog << "max_energy = " << max_energy << std::endl ;
  dealii::deallog << "flux_guess = " << flux_guess << std::endl ;
  dealii::deallog << "temp_guess = " << temp_guess << std::endl ;
  dgmethod.run_non_linear ();
  return 0;
}
#endif



