#include <GlobalTimer.h>

dealii::TimerOutput global_timer(
    std::cout, dealii::TimerOutput::never,
    dealii::TimerOutput::wall_times);
