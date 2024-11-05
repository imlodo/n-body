<header align = "center" style = "text-align: center">
	<span><h1>Soluzione al problema n-body utilizzando MPI</h1></span>
	Soluzione parallelizzata del problema N-Body utilizzando MPI <br>
 <h5>MPI repository: https://www.microsoft.com/en-us/download/details.aspx?id=57467. </h5>
	<h5>
		Antonio Lodato <br>
		a.lodato23@studenti.unisa.it <br>
  Repository GitHub: https://github.com/imlodo/n-body
	</h5>
</header>

### Descrizione del problema

Il problema N-Body è una questione centrale in numerosi settori della scienza, che vanno dalla fisica alla chimica, passando per l'ingegneria aerospaziale. Il problema si occupa di prevedere la dinamica di un sistema composto da N entità interagenti, le cui forze mutuali e movimenti sono determinati dalle leggi della fisica.

Il programma per risolvere il problema N-Body simula le interazioni fisiche tra le particelle, prendendo come input le loro posizioni e velocità iniziali. L'output del programma può essere configurato per mostrare le posizioni e le velocità delle particelle a intervalli specifici oppure le loro condizioni al termine del periodo simulato. Questo consente agli utenti di ottenere dati precisi per analisi successive o per confermare teorie scientifiche.

È essenziale che il software mantenga una coerenza assoluta nei risultati prodotti, indipendentemente dal numero di processori usati nella simulazione. Questa uniformità assicura che le simulazioni siano affidabili e che i loro risultati siano validi universalmente, permettendo agli scienziati di confrontare direttamente i dati simulati con osservazioni reali o altre simulazioni. Per cui dato uno stesso input al programma deve fornire lo stesso output, indipendentemente dal numero di processori utilizzati nel calcolo.

### Descrizione Soluzione

Come punto di partenza si è adottata una soluzione quadraticamente dipendente dal numero di particelle. Per la componente matematica del calcolo delle forze, il programma fa riferimento alla soluzione sequenziale di Harris, disponibile qui: [soluzione n-body di Harris](https://github.com/harrism/mini-nbody/blob/master/nbody.c).

Il programma è progettato per simulare la dinamica di un numero specificato di corpi \( <body_count> \) per un determinato numero di iterazioni \( <iterations> \). Il processo MASTER inizializza casualmente un array di corpi in base agli input e distribuisce una porzione di questo array a ciascuno dei processi SLAVE \( <processors_count>-1 \), assegnando loro la responsabilità di quel segmento.

Sia il MASTER che ogni SLAVE contribuiscono al carico di lavoro computazionale, simulando le forze che agiscono sul sottoinsieme di particelle di loro competenza. Una volta completati i calcoli per tutte le iterazioni specificate, ogni SLAVE invia i propri risultati al MASTER.

Per l'inizializzazione è stato impiegato un algoritmo deterministico per garantire una distribuzione casuale delle configurazioni dei corpi.

**Sinossi del codice**:
- Vengono definiti costanti come il fattore di ammorbidimento, il delta temporale e il seme casuale per la coerenza della simulazione.
- La funzione `main` inizia con l'impostazione dell'ambiente MPI, la validazione degli input e l'inizializzazione dei corpi.
- Viene creato e confermato il tipo di dato MPI per le strutture dei corpi.
- La funzione `randomizeBodies` genera posizioni e velocità iniziali per i corpi.
- La funzione `bodyForce` calcola le forze gravitazionali tra i corpi.
- Le posizioni sono aggiornate in base alle forze calcolate nella funzione `updatePositions`.
- Le operazioni collettive di MPI (`MPI_Scatterv`, `MPI_Gatherv` e `MPI_Ibcast`) gestiscono la distribuzione e la raccolta dei dati tra i processi.
- I tempi e gli stati finali dei corpi possono essere stampati se specificato, con opzioni controllate tramite argomenti da linea di comando.

L'implementazione mira a un'esecuzione parallela efficiente utilizzando MPI, concentrando l'attenzione sulla corretta gestione della memoria e minimizzando il sovraccarico di comunicazione tra i processi.

## Dettagli dell'implementazione

### Definizione della struttura per ogni particella

La rappresentazione di ogni particella è definita attraverso una struttura `Body`, che comprende le coordinate spaziali \(x, y, z\) e le componenti di velocità \(vx, vy, vz\):

```c
typedef struct { 
    float x, y, z, vx, vy, vz; 
} Body;
```

### Inizializzazione

L'inizializzazione dell'array di corpi viene eseguita tramite la funzione `randomizeBodies`, che assegna valori casuali alle coordinate e alle velocità dei corpi. La funzione utilizza `rand()` per generare numeri casuali normalizzati tra -1 e 1:

```c
void randomizeBodies(float *bodies, int numberOfBodies) {
    for (int body = 0; body < numberOfBodies; body++)
        bodies[body] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}
```

La memoria per l'array di `Body` viene allocata come segue, con un casting esplicito del buffer di tipo `float` alla struttura `Body`:

```c
int bytes = numberOfBodies * sizeof(Body);
float *buffer = (float*) malloc(bytes);
Body *bodies = (Body*) buffer;
```

### Definizione di un tipo MPI per la struttura

Per facilitare le operazioni di comunicazione MPI con la struttura `Body`, viene creato un tipo di dato derivato MPI:

```c
MPI_Datatype MPI_BODY;
MPI_Datatype oldTypes[1] = {MPI_FLOAT};
int blocksCount[1] = {BODY_FLOAT}; // BODY_FLOAT = 6, corrispondente ai componenti di Body
MPI_Aint offset[1] = {0};

MPI_Type_create_struct(1, blocksCount, offset, oldTypes, &MPI_BODY);
MPI_Type_commit(&MPI_BODY);
```

Dopo l'uso, il tipo viene liberato con `MPI_Type_free()`:

```c
MPI_Type_free(&MPI_BODY);
```

### Distribuzione dei dati ai processi

La distribuzione iniziale dei corpi ai vari processi avviene tramite `buildBodiesPerProcessAndDispls()`, che calcola come dividere equamente i corpi fra i processi, considerando anche il caso in cui il numero di corpi non sia divisibile uniformemente:

```c
void buildBodiesPerProcessAndDispls(int numberOfBodies, int numberOfTasks, int *bodiesPerProcess, int *displs) {
    int rest = numberOfBodies % numberOfTasks;
    int bodiesDifference = numberOfBodies / numberOfTasks;
    int startPosition = 0;

    for (int process = MASTER; process < numberOfTasks; process++) {
        bodiesPerProcess[process] = (rest > 0) ? (bodiesDifference + 1) : bodiesDifference;
        rest--;
        displs[process] = startPosition;
        startPosition += bodiesPerProcess[process];
    }
}
```

### Comunicazione e calcolo delle forze

Durante la simulazione, ciascun processo calcola le forze agendo sui corpi di sua competenza e aggiorna le posizioni. La comunicazione delle posizioni aggiornate fra i processi avviene utilizzando `MPI_Ibcast()`, e il calcolo delle forze e l'aggiornamento delle posizioni sono gestiti rispettivamente dalle funzioni `bodyForce()` e `updatePositions()`.

Le forze vengono calcolate tenendo conto della distanza fra i corpi, usando un fattore di ammorbidimento per evitare divisioni per zero:

```c
void bodyForce(Body *bodies, float dt, int dependentStart, int dependentStop, int independentStart, int independentStop) {
    for (int i = independentStart; i < independentStop; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        for (int j = dependentStart; j < dependentStop; j++) {
            float dx = bodies[j].x - bodies[i].x, dy = bodies[j].y - bodies[i].y, dz = bodies[j].z - bodies[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }
        bodies[i].vx += dt * Fx;
        bodies[i].vy += dt * Fy;
        bodies[i].vz += dt * Fz;
    }
}
```

Infine, le posizioni dei corpi vengono aggiornate in base alle velocità:

```c
void updatePositions(Body *bodies, float dt, int start, int stop) {
    for (int body = start; body < stop; body++) {
        bodies[body].x += bodies[body].vx * dt;
        bodies[body].y += bodies[body].vy * dt;
        bodies[body].z += bodies[body].vz * dt;
    }
}
```
## Esecuzione

### Prerequisiti

Prima di procedere con il download e l'esecuzione del progetto, è necessario assicurarsi che i seguenti strumenti siano installati sul sistema:

- GCC (GNU Compiler Collection)
- CMake
- MPI (implementazione specifica come OpenMPI o MPICH)

Questi componenti possono essere installati attraverso il gestore di pacchetti della distribuzione Linux utilizzata. Per esempio, su sistemi basati su Ubuntu, si può utilizzare il seguente comando:

```bash
sudo apt-get install build-essential cmake mpich
```

### Download del progetto

Per scaricare il progetto, l'utente deve clonare il repository GitHub utilizzando il comando:

```bash
git clone https://github.com/imlodo/n-body
```

### Compilazione del progetto

Dopo aver clonato il repository, si deve navigare nella directory del progetto e seguire i passaggi per compilare il codice:

1. **Creazione di una cartella build**: è raccomandato creare una cartella build per mantenere separati i file sorgente dai file di compilazione:

   ```bash
   mkdir build
   cd build
   ```

2. **Configurazione del progetto con CMake**: questo passaggio genera i Makefiles necessari per la compilazione:

   ```bash
   cmake ..
   ```

3. **Compilazione del progetto**: questo comando costruisce effettivamente l'eseguibile:

   ```bash
   cmake --build .
   ```

### Esecuzione del programma

Una volta compilato il progetto, l'eseguibile può essere eseguito con MPI. È importante navigare nella cartella che contiene l'eseguibile, che potrebbe trovarsi direttamente nella cartella `build` o in una sottocartella come `Debug`, a seconda della configurazione di CMake.

```bash
cd Debug  # Modificare questo passaggio se la cartella di destinazione è diversa
mpiexec -np 12 ./nbody print 10500 100
```

Il comando sopra indicato esegue il programma `nbody` utilizzando 12 processi paralleli. Il programma simula 10500 corpi per 100 iterazioni e stampa i risultati.

### Nota

Si raccomanda di adattare i comandi alla struttura delle cartelle e alle configurazioni specifiche del progetto. Il comando `mpiexec` potrebbe variare leggermente in base alla specifica implementazione MPI installata (ad esempio, potrebbe essere necessario usare `mpirun`).
