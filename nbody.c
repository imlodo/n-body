#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SOFTENING 1e-9f // Softening factor 1×10^-9 is used to avoid values that are too large
#define DT 0.01f        // delta time (simulation time) of 0.01 seconds
#define SEED 69
#define MASTER 0
#define EXPECTED_ARGUMENT 4
#define PRINT_ARGUMENT 1
#define NUMBER_OF_BODIES_ARGUMENT 2
#define NUMBER_OF_ITERATIONS_ARGUMENT 3
#define BODY_FLOAT 6
#define BODY_NO_DIFFERENCE 0
#define PRINT_REQUIRED 1
#define PRINT_NOT_REQUIRED 0
#define EXECUTION_TIME_REQUIRED 1
#define EXECUTION_TIME_NOT_REQUIRED 0
#define NO_EXECUTION_TIME 0

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *bodies, int numberOfBodies);
void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop, int independentStart, int independentStop);
void updatePositions(Body *bodies, float dt, int start, int stop);
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs);
void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, double executionTime, int isExecutionTimeRequired, int isPrintRequired);
void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd);
void printHowToUse();

int main(int argc, char **argv)
{
    int numberOfTasks, rank, isPrintRequired, numberOfBodies, iterations, bytes;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != EXPECTED_ARGUMENT ||
        (strcmp(argv[PRINT_ARGUMENT], "print") != 0 && strcmp(argv[PRINT_ARGUMENT], "notPrint") != 0))
    {
        if (rank == MASTER)
        {
            printHowToUse();
        }
        MPI_Finalize();
        return -1;
    }

    isPrintRequired = (strcmp(argv[PRINT_ARGUMENT], "print") == 0) ? PRINT_REQUIRED : PRINT_NOT_REQUIRED;
    numberOfBodies = atoi(argv[NUMBER_OF_BODIES_ARGUMENT]);
    iterations = atoi(argv[NUMBER_OF_ITERATIONS_ARGUMENT]);
    srand(SEED);
    bytes = numberOfBodies * sizeof(Body);
    float *buffer = (float *)malloc(bytes);
    Body *bodies = (Body *)buffer;

    if (rank == MASTER)
    {
        randomizeBodies(buffer, BODY_FLOAT * numberOfBodies);
        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations,
                           NO_EXECUTION_TIME, EXECUTION_TIME_NOT_REQUIRED, isPrintRequired);
    }

    MPI_Datatype MPI_BODY;
    MPI_Datatype oldTypes[1] = {MPI_FLOAT};
    int blocksCount[1] = {BODY_FLOAT};
    MPI_Aint offset[1] = {0};
    MPI_Type_create_struct(1, blocksCount, offset, oldTypes, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    int *bodiesPerProcess = (int *)malloc(numberOfTasks * sizeof(int));
    int *displs = (int *)malloc(numberOfTasks * sizeof(int));
    buildBodiesPerProcessAndDispls(numberOfBodies, numberOfTasks, bodiesPerProcess, displs);
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Request *requests = (MPI_Request *)malloc(numberOfTasks * sizeof(MPI_Request));

    if (requests == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for MPI requests.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    int startTime = MPI_Wtime();
    Body *recvBuffer = (Body *)malloc(bodiesPerProcess[rank] * sizeof(Body));
    MPI_Scatterv(bodies, bodiesPerProcess, displs, MPI_BODY, recvBuffer,
                 bodiesPerProcess[rank], MPI_BODY, MASTER, MPI_COMM_WORLD);
    memcpy(&bodies[displs[rank]], recvBuffer, bodiesPerProcess[rank] * sizeof(Body));
    free(recvBuffer);

    for (int iteration = 0; iteration < iterations; iteration++)
    {
        for (int process = MASTER; process < numberOfTasks; process++)
        {
            MPI_Ibcast(&bodies[displs[process]], bodiesPerProcess[process],
                       MPI_BODY, process, MPI_COMM_WORLD, &requests[process]);
        }

        // Each process computes its share of bodies
        int independentStop = displs[rank] + bodiesPerProcess[rank];
        bodyForce(bodies, DT, displs[rank], independentStop, displs[rank], independentStop);

        for (int waitedProcess = MASTER; waitedProcess < numberOfTasks; waitedProcess++)
        {
            if (waitedProcess != rank)
            {
                MPI_Wait(&requests[waitedProcess], &status);
                int dependentStop = displs[waitedProcess] + bodiesPerProcess[waitedProcess];
                bodyForce(bodies, DT, displs[waitedProcess], dependentStop, displs[rank], independentStop);
            }
        }
        
        updatePositions(bodies, DT, displs[rank], independentStop);
    }

    Body *gatherBuffer = NULL;
    if (rank == MASTER)
    {
        gatherBuffer = (Body *)malloc(numberOfBodies * sizeof(Body));
    }

    MPI_Gatherv(&bodies[displs[rank]], bodiesPerProcess[rank], MPI_BODY,
                gatherBuffer, bodiesPerProcess, displs, MPI_BODY, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
    {
        memcpy(bodies, gatherBuffer, numberOfBodies * sizeof(Body));
        free(gatherBuffer);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int finishTime = MPI_Wtime();
    int executionTime = finishTime - startTime;
    if (rank == MASTER)
    {
        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations,
                           executionTime, EXECUTION_TIME_REQUIRED, isPrintRequired);
    }

    free(bodiesPerProcess);
    free(displs);
    MPI_Type_free(&MPI_BODY);

    MPI_Finalize();

    return 0;
}

void randomizeBodies(float *bodies, int numberOfBodies)
{
    for (int body = 0; body < numberOfBodies; body++)
    {
        bodies[body] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop,
               int independentStart, int independentStop)
{
    for (int i = independentStart; i < independentStop; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = dependentStart; j < dependentStop; j++)
        {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[i].vx += dt * Fx;
        bodies[i].vy += dt * Fy;
        bodies[i].vz += dt * Fz;
    }
}

void updatePositions(Body *bodies, float dt, int start, int stop)
{
    for (int body = start; body < stop; body++)
    {
        bodies[body].x += bodies[body].vx * dt;
        bodies[body].y += bodies[body].vy * dt;
        bodies[body].z += bodies[body].vz * dt;
    }
}

void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks,
                                    int *bodiesPerProcess, int *displs)
{
    int rest = numberOfBodies % numberOfTasks;
    int bodiesDifference = numberOfBodies / numberOfTasks;
    int startPosition = 0;

    for (int process = MASTER; process < numberOfTasks; process++)
    {
        if (rest > BODY_NO_DIFFERENCE)
        {
            bodiesPerProcess[process] = bodiesDifference + 1;
            rest--;
        }
        else
        {
            bodiesPerProcess[process] = bodiesDifference;
        }

        displs[process] = startPosition;
        startPosition += bodiesPerProcess[process];
    }
}

void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations,
                        double executionTime, int isExecutionTimeRequired, int isPrintRequired)
{
    if (isPrintRequired == 1)
    {
        printBodies(bodies, numberOfBodies, numberOfTasks, iterations, isExecutionTimeRequired);
    }

    if (isExecutionTimeRequired == 1)
    {
        printf("With %d processor, %d bodies and %d iterations the time of execution is %0.2f seconds\n",
               numberOfTasks, numberOfBodies, iterations, executionTime);

        FILE *file = fopen("./nBodyExecutionTime.txt", "a");
        fprintf(file,
                "With %d processor, %d bodies and %d iterations the time of execution is %0.2f seconds\n",
                numberOfTasks, numberOfBodies, iterations, executionTime);
        fclose(file);
    }
}

void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd)
{
    FILE *file = fopen("./bodies.txt", "a");

    if (isEnd == 1)
    {
        fprintf(file, "Bodies at the end with %d processors and %d iterations:\n",
                numberOfTasks, iterations);
    }
    else
    {
        fprintf(file, "Bodies at the beginning with %d processors %d iterations:\n",
                numberOfTasks, iterations);
    }

    for (int body = 0; body < numberOfBodies; body++)
    {
        fprintf(file, "Body[%d][%f, %f, %f, %f, %f, %f]\n", body,
                bodies[body].x, bodies[body].y, bodies[body].z,
                bodies[body].vx, bodies[body].vy, bodies[body].vz);
    }

    fprintf(file, "\n");
    fclose(file);
}

void printHowToUse()
{
    printf("How to use? Execute: mpirun -np <processors_count> ./nbody [print | notPrint] <body_count> <iterations>\n");
    printf("---> Example: mpirun -np 4 ./nbody print 12 3\n");
    printf("---> The program will run the nBody problem with 4 processors, 12 bodies, and 3 iterations.\n");
    printf("---> Use 'print' to display the bodies' positions or 'notPrint' to skip printing.\n");
}
