#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

    srand(time(NULL));
    double x;
    double y;
    int positive = 0;
    const int probesNumber = 10000000;

    clock_t start = clock();

    for(int i = 0; i < probesNumber; i++) {

        x = 2*((double) rand() / RAND_MAX ) - 1;
        y = 2*((double) rand() / RAND_MAX ) - 1;

        if( x*x + y*y < 1.0)
            positive++;

    }

    clock_t end = clock();
    float seconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Monte Carlo - Pi estimation is: %lf", 4. *(double)positive/(double)probesNumber);
    printf("Running time of algorithm: %lf", seconds);
    return 0;

}