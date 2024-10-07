
| **Termine**                    | **Definizione**                                                                                                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Decomposizione del dominio** | Tecnica di parallelizzazione che divide i dati in parti elaborate simultaneamente da processori diversi.                                                                                                      |
| **Decomposizione funzionale**  | Tecnica di parallelizzazione che divide il problema in task indipendenti eseguiti da processori diversi.                                                                                                      |
| **Data parallelism**           | Tipo di parallelismo dove la stessa operazione viene eseguita simultaneamente su diversi dati da processori differenti.                                                                                       |
| **Task parallelism**           | Tipo di parallelismo dove diversi task indipendenti vengono eseguiti simultaneamente da processori diversi.                                                                                                   |
| **Load balancing**             | Distribuzione equa del lavoro tra i processori per evitare colli di bottiglia e massimizzare l'efficienza.                                                                                                    |
| **Tempo di esecuzione**        | Tempo totale impiegato da un programma per completare l'esecuzione. Include il tempo di calcolo, il tempo di comunicazione e il tempo di inattività.                                                          |
| **Tempo di calcolo**           | Tempo impiegato da un processore per eseguire le operazioni sui dati.                                                                                                                                         |
| **Tempo di comunicazione**     | Tempo impiegato per lo scambio di dati tra i processori.                                                                                                                                                      |
| **Tempo di inattività**        | Tempo in cui un processore rimane inattivo in attesa di dati o istruzioni.                                                                                                                                    |
| **Latenza**                    | Ritardo temporale tra l'invio e la ricezione di un messaggio.                                                                                                                                                 |
| **Larghezza di banda**         | Quantità di dati che possono essere trasmessi in un determinato periodo di tempo.                                                                                                                             |
| **Sovrapposizione**            | Tecnica che esegue simultaneamente la comunicazione e il calcolo per ridurre il tempo di inattività e migliorare le prestazioni.                                                                              |
| **Speedup**                    | Rapporto tra il tempo di esecuzione di un programma seriale e quello della sua versione parallela. Misura l'accelerazione ottenuta con la parallelizzazione.                                                  |
| **Efficienza**                 | Rapporto tra lo speedup e il numero di processori utilizzati. Misura l'efficacia nell'utilizzo delle risorse di calcolo.                                                                                      |
| **Legge di Amdahl**            | Formula che calcola lo speedup massimo ottenibile in base alla frazione di codice intrinsecamente sequenziale. Evidenzia i limiti della parallelizzazione quando una parte del codice non è parallelizzabile. |
# Problem Decomposition
Esistono due tipi principali di decomposizione dei problemi:
- **Decomposizione del dominio**
- **Decomposizione funzionale**

## Decomposizione del dominio
Nella **decomposizione del dominio** (o "data parallelism"):
- I dati vengono divisi in parti di dimensioni simili e assegnati a diversi processori.
- Ogni processore lavora solo sulla porzione di dati che gli è stata assegnata.
- I processori possono comunicare periodicamente per scambiarsi dati.
#### Vantaggi del data parallelism
- **Flusso di controllo unico**: Un algoritmo di data parallelism esegue una sequenza di istruzioni elementari sui dati, dove ogni istruzione inizia solo dopo la fine della precedente.
- **Single-Program-Multiple-Data (SPMD)**: Lo stesso codice viene eseguito su tutti i processori, che lavorano in modo indipendente su grandi porzioni di dati, con scambi di dati minimi.

## Decomposizione funzionale
La **decomposizione del dominio** non è sempre l'approccio più efficiente:
- Quando i tempi di elaborazione dei dati differiscono tra i processori, le prestazioni del codice sono limitate dal processo più lento, mentre gli altri rimangono inattivi.
  
In questi casi, la **decomposizione funzionale** o **"task parallelism"** è preferibile:
- Il problema viene suddiviso in un numero elevato di task più piccoli, assegnati ai processori non appena disponibili.
- I processori più veloci ricevono semplicemente più lavoro da svolgere.
#### Vantaggi del task parallelism
- Di solito è implementato nel **paradigma client-server**:
  - Un processo master assegna i task ai processi slave.
  - Può essere implementato a qualsiasi livello del programma, per esempio, facendo eseguire più input in parallelo o assegnando task specifici a ciascun processore.

## Problematiche della programmazione parallela
Lo scopo principale di un programma parallelo è ottenere migliori prestazioni rispetto alla versione seriale. Alcuni aspetti da considerare:

- **Load balancing**
- **Minimizzazione della comunicazione**
- **Sovrapposizione tra comunicazione e computazione**

## Load Balancing
- L'obiettivo è dividere equamente il lavoro tra i processori.
- È semplice quando tutti i processori eseguono le stesse operazioni su diverse parti di dati, ma più complesso in presenza di grandi variazioni nei tempi di elaborazione.
#### Tipi di Load Balancing
- **Task/Program Partitioning**: Dividere il lavoro in modo che ogni processore esegua la stessa quantità di operazioni.
- **Data Partitioning**: Dividere i dati tra i processori riducendo al minimo l'interazione tra di essi.

## Minimizzare la comunicazione
- Il **tempo di esecuzione totale** è un fattore chiave nella programmazione parallela.
- Il tempo di esecuzione è composto da:
  - **Tempo di calcolo**
  - **Tempo di inattività** (idle time)
  - **Tempo di computazione**: è il tempo trascorso eseguendo calcoli sui dati.
  - **Idle time**: è il tempo che un processo trascorre aspettando i dati da altri processori.
  - **Tempo di comunicazione**: è il tempo necessario ai processi per inviare e ricevere messaggi. 
	   - Si misura in termini di **latenza** (tempo necessario per iniziare la comunicazione) e **larghezza di banda** (velocità di trasmissione dei dati).
	   - I programmi seriali non usano la comunicazione tra processi, quindi è importante minimizzare questo tempo per migliorare le prestazioni.

## Sovrapposizione tra Comunicazione e Computazione
- Per ridurre l'**idle time**, si può sovrapporre la comunicazione e la computazione:
  - Un processo può essere occupato con nuovi task mentre attende il completamento della comunicazione.
  - Ciò è possibile utilizzando tecniche di comunicazione **non-bloccanti** e computazione **non specifica ai dati**.
  - Tuttavia, nella pratica, è difficile implementare efficacemente questa interleaving.

## Metriche di Prestazione nei Sistemi Paralleli

### Tempo di esecuzione
- **Ts (tempo seriale)**: il tempo che un algoritmo sequenziale impiega dall'inizio alla fine.
- **Tp (tempo parallelo)**: il tempo che un calcolo parallelo impiega dall'inizio fino alla conclusione dell'ultimo processore.

### Speedup (S)
- Lo **speedup** è il rapporto tra il tempo di esecuzione seriale e quello parallelo:
  
$$  S = \frac{Ts}{Tp}$$
  
  - Esempio: somma di _n_ numeri, **Tp = O(logn)** e **Ts = O(n)**. Lo speedup è **S = O(n/logn)**.
  - Lo **speedup** non può superare il numero di processori, **S ≤ p**.
  - Lo **speedup superlineare** si verifica quando le prestazioni parallele superano quelle seriali, ad esempio a causa di vantaggi hardware o algoritmi più efficienti in parallelo.

### Efficienza (E)
- L'**efficienza** è il rapporto tra lo speedup e il numero di processori:
  
$$  E = \frac{S}{p}$$
  
  - Misura quanto efficacemente ogni processore viene utilizzato. Esempio: somma di _n_ numeri su _n_ processori, $Tp = O(log(n)), \ E = O(\frac{1}{log(n)}).$

## Legge di Amdahl
- **Legge di Amdahl**: se una frazione **f** di un calcolo è intrinsecamente sequenziale, il massimo speedup ottenibile su _p_ processori è:
  
$$  S ≤ \frac{1}{f + \frac{1-f}{p}}$$
  
  - Per esempio, se **f = 10%**, lo **speedup** massimo è **S ≤ 10** man mano che **p** tende a infinito.
  - La legge assume che la frazione sequenziale di un programma sia costante. Tuttavia, in molti casi, con l'aumentare della dimensione del problema, la parte sequenziale si riduce. 

### Considerazioni su Amdahl
- Sebbene Amdahl suggerisca di limitare l'uso di macchine parallele, in alcuni casi (come le previsioni meteorologiche), l'aumento delle dimensioni del problema migliora la precisione, consentendo ad esempio previsioni più accurate.


## FAQ sulla Programmazione Parallela

### 1. Quali sono i principali tipi di decomposizione dei problemi nella programmazione parallela?

Esistono due tipi principali di decomposizione dei problemi: decomposizione del dominio e decomposizione funzionale.

- **Decomposizione del dominio (o "data parallelism")**: I dati vengono divisi in parti di dimensioni simili e assegnati a diversi processori. Ogni processore esegue lo stesso codice sulla propria porzione di dati, comunicando solo occasionalmente per scambiarsi informazioni.
- **Decomposizione funzionale (o "task parallelism")**: Il problema viene suddiviso in un numero elevato di task più piccoli, assegnati ai processori non appena disponibili. Questo approccio è vantaggioso quando i tempi di elaborazione dei dati variano significativamente, in quanto i processori più veloci possono gestire più lavoro.

### 2. Quali sono i vantaggi del data parallelism rispetto al task parallelism?

Il data parallelism offre alcuni vantaggi specifici:

- **Flusso di controllo unico**: Le istruzioni vengono eseguite in sequenza, semplificando la programmazione e il debug.
- **Single-Program-Multiple-Data (SPMD)**: Lo stesso codice viene eseguito su tutti i processori, riducendo la complessità.

Tuttavia, se i tempi di elaborazione dei dati sono eterogenei, il task parallelism potrebbe essere più efficiente.

### 3. Quali sfide si incontrano nella programmazione parallela?

Tra le sfide principali della programmazione parallela troviamo:

- **Load balancing**: Distribuire equamente il lavoro tra i processori per evitare colli di bottiglia.
- **Minimizzare la comunicazione**: Ridurre il tempo speso dai processori per comunicare tra loro, che può rallentare l'esecuzione.
- **Sovrapposizione tra comunicazione e computazione**: Cercare di far lavorare i processori su nuovi task mentre aspettano i dati da altri processori.

### 4. Come si misura l'efficienza di un programma parallelo?

Le metriche principali sono:

- **Speedup (S)**: Rapporto tra il tempo di esecuzione seriale e quello parallelo. Un alto speedup indica un miglioramento significativo delle prestazioni.
- **Efficienza (E)**: Rapporto tra lo speedup e il numero di processori. Misura quanto efficacemente ogni processore viene utilizzato.

### 5. Cos'è la legge di Amdahl e cosa implica?

La legge di Amdahl afferma che il massimo speedup ottenibile da un programma parallelo è limitato dalla porzione di codice che deve essere eseguita sequenzialmente. Se una frazione significativa del codice non può essere parallelizzata, il guadagno di prestazioni sarà limitato, anche con un numero elevato di processori.

### 6. La legge di Amdahl significa che la programmazione parallela è inutile?

No. Anche se la legge di Amdahl evidenzia i limiti della parallelizzazione, in molti casi i benefici superano le limitazioni. Ad esempio, in problemi complessi come le previsioni meteorologiche, l'aumento delle dimensioni del problema (e quindi la parte parallelizzabile) può portare a risultati più accurati, giustificando l'utilizzo di sistemi paralleli.

### 7. Quali sono le componenti del tempo di esecuzione in un programma parallelo?

Il tempo di esecuzione totale è composto da:

- **Tempo di calcolo**: Tempo dedicato all'esecuzione delle operazioni sui dati.
- **Tempo di inattività (idle time)**: Tempo in cui un processore rimane inattivo in attesa di dati da altri processori.
- **Tempo di comunicazione**: Tempo necessario per la trasmissione dei dati tra i processori.

### 8. Come si può ridurre il tempo di inattività in un programma parallelo?

Una tecnica per ridurre l'idle time è la sovrapposizione tra comunicazione e computazione. Utilizzando metodi di comunicazione non-bloccanti, un processore può iniziare un nuovo task mentre attende il completamento di una comunicazione precedente. Questo approccio, tuttavia, può essere complesso da implementare in modo efficiente.

### Quiz


1. **Descrivere la differenza tra decomposizione del dominio e decomposizione funzionale.**
2. **Quali sono i vantaggi del data parallelism?**
3. **In quali situazioni la decomposizione funzionale è preferibile alla decomposizione del dominio?**
4. **Definire il concetto di load balancing e spiegare la sua importanza nella programmazione parallela.**
5. **Quali sono i tre componenti principali del tempo di esecuzione di un programma parallelo?**
6. **Spiegare come la minimizzazione della comunicazione può migliorare le prestazioni di un programma parallelo.**
7. **Descrivere la tecnica di sovrapposizione tra comunicazione e computazione e i suoi benefici.**
8. **Definire lo speedup e spiegare come viene calcolato.**
9. **Qual è la differenza tra speedup ed efficienza?**
10. **Spiegare la Legge di Amdahl e le sue implicazioni per la programmazione parallela.**

### Chiave di Risposta del Quiz

1. **Decomposizione del dominio** divide i dati in parti elaborate simultaneamente, mentre la **decomposizione funzionale** divide il problema in task indipendenti.
2. Il **data parallelism** offre un flusso di controllo unico, semplifica la programmazione (SPMD) e minimizza la comunicazione tra i processori.
3. La **decomposizione funzionale** è preferibile quando i tempi di elaborazione dei dati variano significativamente, evitando che i processori più veloci rimangano inattivi.
4. **Load balancing** significa distribuire equamente il lavoro tra i processori, evitando colli di bottiglia e massimizzando l'efficienza.
5. I tre componenti del tempo di esecuzione sono: **tempo di calcolo**, **tempo di comunicazione** e **tempo di inattività**.
6. Minore comunicazione significa meno tempo speso nello scambio di dati tra i processori, riducendo il tempo di esecuzione totale.
7. La **sovrapposizione** esegue la comunicazione e il calcolo simultaneamente, riducendo il tempo di inattività e migliorando le prestazioni.
8. **Speedup** è l'accelerazione ottenuta con la parallelizzazione, calcolato come il rapporto tra il tempo di esecuzione seriale e quello parallelo: **S = Ts/Tp**.
9. **Speedup** misura l'accelerazione totale, mentre l'**efficienza** misura l'utilizzo efficace di ciascun processore (**E = S/p**).
10. La **Legge di Amdahl** afferma che lo speedup è limitato dalla porzione di codice non parallelizzabile, evidenziando i limiti della parallelizzazione.

### Domande per Saggio

1. Discutere i vantaggi e gli svantaggi della decomposizione del dominio e della decomposizione funzionale, fornendo esempi concreti di situazioni in cui ciascun approccio sarebbe più appropriato.
2. Analizzare l'importanza del load balancing nella programmazione parallela. Descrivere diverse tecniche di load balancing e discutere i fattori da considerare nella scelta della tecnica più adatta a un determinato problema.
3. Spiegare come la comunicazione tra processi può influenzare le prestazioni di un programma parallelo. Descrivere diverse tecniche per ridurre al minimo la comunicazione e discutere i compromessi tra prestazioni e complessità di implementazione.
4. Confrontare e contrastare lo speedup e l'efficienza come metriche di performance per i sistemi paralleli. Discutere l'importanza di considerare entrambe le metriche durante la progettazione e l'implementazione di algoritmi paralleli.
5. Discutere le implicazioni della Legge di Amdahl per la progettazione e l'implementazione di algoritmi paralleli. Analizzare i limiti della legge e discutere le strategie per superare tali limiti e ottenere prestazioni elevate su sistemi con un gran numero di processori.