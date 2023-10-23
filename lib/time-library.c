#include "time-library.h"


void sub_timespec ( struct timespec t1, struct timespec t2, struct timespec *td) {

	td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
	td->tv_sec = t2.tv_sec - t1.tv_sec;

	
	if (td->tv_sec > 0 && td->tv_nsec < 0 ){
		td->tv_nsec += 	NS_PER_SECOND;
		td->tv_sec--;
	}
	if (td->tv_sec < 0 && td->tv_nsec > 0 ){
                td->tv_nsec -=  NS_PER_SECOND;
                td->tv_sec++;
        }

}

double simple_sub_timespec ( struct timespec t1 , struct timespec t2 ) {

	double td1, td2 ;

	td1 = t1 . tv_sec + ( t1 . tv_nsec /( double ) NS_PER_SECOND ) ;
	td2 = t2 . tv_sec + ( t2 . tv_nsec /( double ) NS_PER_SECOND ) ;

	return ( td2 - td1 ) ;
}