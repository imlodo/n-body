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

### Implementazione Soluzione
