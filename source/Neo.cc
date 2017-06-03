#include <Neo.h>

#ifndef BENCHMARKS

std::unique_ptr<dealii::TimerOutput>        timer ;
std::unique_ptr<MPI_Comm>                   mpi_communicator ;
std::unique_ptr<dealii::ConditionalOStream> pcout ;

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
  mpi_communicator.reset(new MPI_Comm(MPI_COMM_WORLD));
  pcout.reset(new dealii::ConditionalOStream(std::cout,dealii::Utilities::MPI::this_mpi_process(*mpi_communicator) == 0));
#if PARALLEL_LA == 0
  *pcout<< "Using deal.II (serial) linear algebra" << std::endl;
#elif PARALLEL_LA == 1
  *pcout<< "Using PETSc parallel linear algebra" << std::endl;
#elif PARALLEL_LA == 2
  *pcout<< "Using Trilinos parallel linear algebra" << std::endl;
#else
  *pcout<< "Using deal.II parallel linear algebra" << std::endl;
#endif // PARALLEL_LA
  std::ofstream logfile("deallog");
  dealii::deallog.attach(logfile);
  if (dealii::Utilities::MPI::this_mpi_process(*mpi_communicator)==0)
    dealii::deallog.depth_console (3);
  timer.reset(new dealii::TimerOutput (*mpi_communicator, *pcout,dealii::TimerOutput::never,dealii::TimerOutput::wall_times));
  const unsigned int d = 2 ;
  const unsigned int fe_degree = 1 ;
#ifdef MATRIXFREE
  typedef MfreeOperator<d,fe_degree,fe_degree+1,double> SystemMatrixType;
  *pcout << "Using deal.II's MatrixFree objects" << std::endl;
#else
  typedef MFOperator<d,fe_degree,double> SystemMatrixType;
  *pcout << "Using MeshWorker-based matrix-free implementation" << std::endl;
#endif // MATRIXFREE
#ifdef MG   
  typedef GMGPreconditioner<d>  Precond;
#else // MG OFF
  typedef dealii::PreconditionIdentity          Precond;
#endif // MG
  
  Simulator<SystemMatrixType,LA::MPI::Vector,Precond,d,fe_degree> dgmethod;
  dgmethod.n_levels = (argc > 1) ? atoi(argv[1]) : 2 ;
  dgmethod.min_level=0;
  dgmethod.smoothing_steps = (argc > 2) ? atoi(argv[2]) : 1 ;
  dgmethod.run ();
  return 0;
}
#endif



