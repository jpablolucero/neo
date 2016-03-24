#include <GlobalTimer.h>

dealii::TimerOutput global_timer(MPI_COMM_WORLD,
                                 std::cout, dealii::TimerOutput::never,
                                 dealii::TimerOutput::cpu_and_wall_times);
