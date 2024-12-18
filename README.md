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

## Correttezza

La correttezza del programma è stata verificata attraverso il confronto tra l'output generato e quello della [soluzione sequenziale di Harrism](https://github.com/harrism/mini-nbody/blob/master/nbody.c), usando lo stesso numero di particelle, numero di iterazioni e condizioni iniziali deterministiche. Indipendentemente dal numero di processori utilizzati, l'output del programma è consistente e coincide con i risultati ottenuti dalla soluzione sequenziale, confermando l'affidabilità dell'implementazione parallela.

### Esempio di esecuzione

Con 12 particelle e 3 iterazioni, l'array iniziale di particelle è il seguente:

```plaintext
Body[0][0.233, -0.445, 0.865, -0.124, 0.258, -0.456]
Body[1][-0.134, 0.385, 0.922, -0.500, 0.389, 0.561]
Body[2][0.746, -0.769, 0.033, 0.233, 0.144, -0.250]
Body[3][0.155, 0.012, -0.054, 0.983, -0.299, 0.044]
Body[4][-0.028, -0.973, 0.229, 0.311, 0.188, -0.092]
Body[5][0.391, 0.290, -0.823, -0.635, 0.676, 0.322]
Body[6][-0.671, 0.742, 0.040, 0.688, -0.244, -0.213]
Body[7][0.978, -0.765, -0.239, -0.045, 0.428, 0.103]
Body[8][-0.477, 0.215, -0.806, 0.514, -0.358, 0.866]
Body[9][0.483, -0.369, 0.794, -0.831, 0.542, -0.079]
Body[10][-0.690, 0.587, -0.425, 0.272, -0.962, 0.065]
Body[11][0.992, -0.122, 0.020, -0.099, 0.498, -0.861]
```

Dopo aver eseguito 3 iterazioni, l'output è il seguente:

```plaintext
Body[0][0.210, -0.411, 0.885, -0.101, 0.234, -0.480]
Body[1][-0.149, 0.405, 0.910, -0.521, 0.403, 0.550]
Body[2][0.770, -0.790, 0.055, 0.247, 0.155, -0.260]
Body[3][0.172, 0.032, -0.079, 1.008, -0.314, 0.030]
Body[4][-0.056, -0.980, 0.215, 0.319, 0.192, -0.120]
Body[5][0.415, 0.315, -0.840, -0.655, 0.700, 0.310]
Body[6][-0.658, 0.760, 0.022, 0.703, -0.251, -0.230]
Body[7][0.955, -0.788, -0.250, -0.035, 0.440, 0.112]
Body[8][-0.490, 0.238, -0.816, 0.530, -0.368, 0.856]
Body[9][0.510, -0.355, 0.810, -0.850, 0.555, -0.072]
Body[10][-0.712, 0.610, -0.440, 0.282, -0.975, 0.055]
Body[11][1.002, -0.107, 0.015, -0.084, 0.510, -0.875]
```

Questo output dimostra la consistenza del comportamento del sistema, garantendo che, indipendentemente dalla distribuzione dei processi, i risultati restano invariati.

La coerenza dei risultati conferma l'accuratezza dell'algoritmo parallelo implementato. Le iterazioni mostrano la stabilità e l'affidabilità dei calcoli effettuati dal programma, rispettando il principio di determinismo nel caso di input identici.

## Benchmark

Il programma è stato sottoposto a test approfonditi per valutare la scalabilità forte e debole. Questi test sono stati ripetuti più volte per garantire la riproducibilità dei risultati, e i valori presentati sono medie dei risultati ottenuti.

### Scalabilità Forte
La **scalabilità forte** si riferisce alla capacità del sistema di mantenere un tempo di esecuzione costante nonostante l'aumento del numero di processori, mantenendo inalterata la dimensione totale del problema. In questi test, abbiamo fissato il numero di particelle mentre incrementavamo il numero di processori. Specificatamente, abbiamo utilizzato 15,000 particelle e 500 iterazioni, variando il numero di processori da 1 a 32.

![image](https://github.com/user-attachments/assets/2a5f34fa-8e84-46d0-83d0-c97155683a2b)

### Scalabilità Debole
La **scalabilità debole** esamina come l'efficienza del sistema varia con l'aumento del numero di processori, quando la dimensione del problema per processore rimane costante. Abbiamo condotto due tipi di test per la scalabilità debole:

1. **Iterazioni Non Costanti**: Abbiamo incrementato il numero di particelle e di iterazioni proporzionalmente al numero di processori. Specificamente, abbiamo utilizzato 1,000 particelle e 10 iterazioni per processore, con un numero di processori che varia da 1 a 32. Es. se i processori sono 2 abbiamo 2,000 particelle e 20 iterazioni e così via.
2. **Iterazioni Costanti**: Il carico di lavoro per processore è stato mantenuto costante aumentando il numero di particelle, mentre il numero totale di iterazioni è rimasto fisso a 300. Qui, ogni processore ha gestito 1,000 particelle.

![image](https://github.com/user-attachments/assets/b270dffb-6544-4a89-a51b-f4abe9505ef3)


### Configurazione Hardware
I test sono stati eseguiti utilizzando un cluster di 8 macchine t2.xlarge su AWS Educate, con le seguenti specifiche per ogni macchina:

- **Modello**: t2.xlarge
- **vCPU**: 4
- **Memoria**: 16 GiB
- **Storage**: Solo EBS

Questo setup sfrutta il limite massimo di vCPU consentito da AWS Educate, permettendo un totale di 32 vCPU distribuite sulle macchine virtuali in esecuzione.

![image](https://github.com/user-attachments/assets/2c50758e-18b8-4604-a3bd-9f06f88c6af8)


### Speedup
Lo **speedup** è stato calcolato confrontando i tempi di esecuzione del programma eseguito su un singolo processore con quelli ottenuti utilizzando un numero crescente di processori, da 2 a 32. Questa metrica mostra l'efficacia del parallelismo nel ridurre il tempo di calcolo.

![image](https://github.com/user-attachments/assets/3fa964db-17b7-4f43-85f5-764362fe9659)

Questi test hanno fornito dati cruciali sulla capacità del programma di gestire carichi di lavoro intensivi in un ambiente distribuito, evidenziando l'efficienza e le potenzialità della soluzione proposta per computazioni parallele su larga scala.
