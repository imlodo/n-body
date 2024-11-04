#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SOFTENING 1e-9f // Softening factor 1×10^-9 is used to avoid values that are too large
#define DT 0.01f        // delta time (simulation time) of 0.01 seconds
#define SEED 69         // Custom seed for rand functions
#define MASTER 0        // Master node id
#define CORRECTLY_INVOKED 1
#define NOT_CORRECTLY_INVOKED 0
#define EXPECTED_ARGUMENT 4  // Aggiornato a 4, poiché abbiamo solo 4 argomenti ora
#define PRINT_ARGUMENT 1     // Il primo argomento passato è "print" o "notPrint"
#define NUMBER_OF_BODIES_ARGUMENT 2  // Il secondo è il numero di corpi
#define NUMBER_OF_ITERATIONS_ARGUMENT 3  // Il terzo è il numero di iterazioni
#define BODY_FLOAT 6
#define BODY_NO_DIFFERENCE 0
#define PRINT_REQUIRED 1
#define PRINT_NOT_REQUIRED 0
#define EXECUTION_TIME_REQUIRED 1
#define EXECUTION_TIME_NOT_REQUIRED 0
#define NO_EXECUTION_TIME 0

// Data structure representing the individual particle
typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

// Declaration of functions
void randomizeBodies(float *bodies, int numberOfBodies);
void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop, int independentStart, int independentStop);
void updatePositions(Body *bodies, float dt, int start, int stop);
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs);
void printTimeAndBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, double executionTime, int isExecutionTimeRequired, int isPrintRequired);
void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd);
void printHowToUse();

int main(int argc, char **argv)
{
    // Inizializzazione dell'ambiente MPI
    MPI_Init(NULL, NULL);

    int numberOfTasks, rank;

    // Ottiene il numero di processi MPI realmente usati
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);

    // Ottiene il rank del processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Gestione dei parametri di input
    // Arresta l'esecuzione se non sono forniti correttamente
    int isCorrectlyInvoked = CORRECTLY_INVOKED;
    if (argc != EXPECTED_ARGUMENT)
    {
        isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;
    }

    int isPrintRequired;
    if (isCorrectlyInvoked != NOT_CORRECTLY_INVOKED)
    {
        if (strcmp(argv[PRINT_ARGUMENT], "print") == 0)
        {
            isPrintRequired = PRINT_REQUIRED;
        }
        else if (strcmp(argv[PRINT_ARGUMENT], "notPrint") == 0)
        {
            isPrintRequired = PRINT_NOT_REQUIRED;
        }
        else
        {
            isCorrectlyInvoked = NOT_CORRECTLY_INVOKED;
        }
    }

    if (isCorrectlyInvoked == NOT_CORRECTLY_INVOKED)
    {
        if (rank == MASTER)
        {
            printHowToUse();
        }

        MPI_Finalize();

        return 0;
    }

    int numberOfBodies = atoi(argv[NUMBER_OF_BODIES_ARGUMENT]);
    int iterations = atoi(argv[NUMBER_OF_ITERATIONS_ARGUMENT]);
    srand(SEED);

    // Inizializzazione casuale dei corpi
    int bytes = numberOfBodies * sizeof(Body);
    float *buffer = (float *)malloc(bytes);
    Body *bodies = (Body *)buffer;
    if (rank == MASTER)
    {
        randomizeBodies(buffer, BODY_FLOAT * numberOfBodies);

        printTimeAndBodies(bodies, numberOfBodies, numberOfTasks, iterations,
                           NO_EXECUTION_TIME, EXECUTION_TIME_NOT_REQUIRED, isPrintRequired);
    }

    // Creazione del datatype MPI custom per i corpi
    MPI_Datatype MPI_BODY;
    MPI_Datatype oldTypes[1] = {MPI_FLOAT};
    int blocksCount[1] = {BODY_FLOAT};
    MPI_Aint offset[1] = {0};
    MPI_Type_create_struct(1, blocksCount, offset, oldTypes, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);

    // Contiene il numero di corpi da inviare a ciascuno dei processi
    // e gli spostamenti dove inizia ciascun segmento.
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

    // Scatter dei dati
    Body *recvBuffer = (Body *)malloc(bodiesPerProcess[rank] * sizeof(Body));
    MPI_Scatterv(bodies, bodiesPerProcess, displs, MPI_BODY, recvBuffer,
                bodiesPerProcess[rank], MPI_BODY, MASTER, MPI_COMM_WORLD);
    memcpy(&bodies[displs[rank]], recvBuffer, bodiesPerProcess[rank] * sizeof(Body));
    free(recvBuffer);

    // Calcola per il numero di iterazioni richieste
    for (int iteration = 0; iteration < iterations; iteration++)
    {
        for (int process = MASTER; process < numberOfTasks; process++)
        {
            MPI_Ibcast(&bodies[displs[process]], bodiesPerProcess[process],
                       MPI_BODY, process, MPI_COMM_WORLD, &requests[process]);
        }

        // Ogni processo calcola la propria parte di corpi
        int independentStop = displs[rank] + bodiesPerProcess[rank];
        bodyForce(bodies, DT, displs[rank], independentStop, displs[rank], independentStop);

        for (int waitedProcess = MASTER; waitedProcess < numberOfTasks; waitedProcess++)
        {
            if (waitedProcess != rank)
            {
                // Aspetta il processo con lo stesso rank di waitedProcess
                MPI_Wait(&requests[waitedProcess], &status);

                // Calcola sui propri particolari rispetto ai particolari
                // del processo con lo stesso rank di waitedProcess.
                int dependentStop = displs[waitedProcess] + bodiesPerProcess[waitedProcess];
                bodyForce(bodies, DT, displs[waitedProcess], dependentStop, displs[rank], independentStop);
            }
        }

        // Alla fine integra le posizioni
        updatePositions(bodies, DT, displs[rank], independentStop);
    }

    // Raccoglie tutti i calcoli dagli slave al master
    Body *gatherBuffer = NULL;
    if (rank == MASTER) {
        gatherBuffer = (Body *)malloc(numberOfBodies * sizeof(Body));
    }
    
    MPI_Gatherv(&bodies[displs[rank]], bodiesPerProcess[rank], MPI_BODY,
                gatherBuffer, bodiesPerProcess, displs, MPI_BODY, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
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

    // Finalizza l'ambiente MPI
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

    // Si basa sul fatto che il resto è sempre minore del divisore
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
    // Se è richiesto il tempo di esecuzione, allora è la fine del calcolo
    if (isPrintRequired == 1)
    {
        printBodies(bodies, numberOfBodies, numberOfTasks, iterations, isExecutionTimeRequired);
    }

    if (isExecutionTimeRequired == 1)
    {
        printf("Con %d processori, %d corpi e %d iterazioni il tempo di esecuzione è %0.2f secondi\n",
               numberOfTasks, numberOfBodies, iterations, executionTime);

        FILE *file = fopen("./nBodyExecutionTime.txt", "a");
        fprintf(file,
                "Con %d processori, %d corpi e %d iterazioni il tempo di esecuzione è %0.2f secondi\n\n",
                numberOfTasks, numberOfBodies, iterations, executionTime);
        fclose(file);
    }
}

void printBodies(Body *bodies, int numberOfBodies, int numberOfTasks, int iterations, int isEnd)
{
    FILE *file = fopen("./bodies.txt", "a");

    if (isEnd == 1)
    {
        fprintf(file, "Corpi alla fine con %d processori e %d iterazioni:\n",
                numberOfTasks, iterations);
    }
    else
    {
        fprintf(file, "Corpi all'inizio con %d processori e %d iterazioni:\n",
                numberOfTasks, iterations);
    }

    for (int body = 0; body < numberOfBodies; body++)
    {
        fprintf(file, "Corpo[%d][%f, %f, %f, %f, %f, %f]\n", body,
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
