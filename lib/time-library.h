#include <time.h>
#define NS_PER_SECOND 1e9

void sub_timespec ( struct timespec t1, struct timespec t2, struct timespec *td);
double simple_sub_timespec ( struct timespec t1 , struct timespec t2 );