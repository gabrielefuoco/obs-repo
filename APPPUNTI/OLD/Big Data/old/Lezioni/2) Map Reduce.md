
| Termine | **Definizione** |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MapReduce** | Un modello di programmazione per l'elaborazione di grandi insiemi di dati su cluster di computer. |
| **Divide et Impera** | Una strategia di risoluzione dei problemi che consiste nel suddividere un problema in sottoproblemi più piccoli, risolvere i sottoproblemi e quindi combinare le soluzioni per ottenere la soluzione al problema originale. |
| **Scalabilità Orizzontale** | La capacità di un sistema di aumentare le prestazioni aggiungendo più macchine al cluster. |
| **Server Commodity** | Server economici e facilmente reperibili. |
| **Mapper** | Una funzione che elabora una coppia chiave-valore di input e produce un insieme di coppie chiave-valore intermedie. |
| **Reducer** | Una funzione che elabora un insieme di coppie chiave-valore intermedie con la stessa chiave e produce un insieme di coppie chiave-valore di output. |
| **Shuffle and Sort** | La fase di MapReduce che raggruppa le coppie chiave-valore intermedie per chiave e le ordina. |
| **Combiner** | Una funzione opzionale che può essere utilizzata per aggregare i dati intermedi prima che vengano inviati ai reducer. |
| **Partizionamento** | Il processo di divisione dei dati in sottoinsiemi che vengono elaborati dai reducer. |
| **Clustering** | Il processo di raggruppamento di un insieme di oggetti in gruppi (cluster) in base alla loro somiglianza. |
| **Algoritmo K-means** | Un algoritmo di clustering che mira a partizionare gli oggetti in k cluster, dove ogni oggetto appartiene al cluster con la media più vicina. |

L'idea chiave di **MapReduce** è basata sul concetto di "divide et impera" per affrontare problemi di big data, suddividendo un grande problema in sotto-problemi più piccoli che possono essere eseguiti in parallelo. 

### Divide et impera

- **Scomposizione del problema**: Dividere il problema principale in compiti più piccoli, assegnandoli a worker che possono essere thread, core, processori o macchine in un cluster.
- **Esecuzione parallela**: I worker eseguono i compiti in parallelo, utilizzando al meglio le risorse disponibili.
- **Combinazione dei risultati**: I risultati intermedi dei worker vengono raccolti e combinati per ottenere la soluzione finale.
- **Coordinamento**: Gestire la sincronizzazione tra i worker, la condivisione di dati parziali e i fallimenti.

### Scalabilità orizzontale

- **Preferenza per molti server commodity**: È più efficiente usare molti server a basso costo piuttosto che pochi server high-end.
- **I/O lento rispetto alla velocità di elaborazione**: Il vero collo di bottiglia nei big data è spesso l'I/O, non l'elaborazione.
- **Condivisione vs. nessuna condivisione**: Mentre la condivisione di dati globali è complessa e presenta problemi di sincronizzazione e deadlock, la "nessuna condivisione" evita tali complicazioni, permettendo ai worker di operare in modo indipendente.

## Map Reduce: Panoramica

- **Modello di programmazione** creato per elaborare grandi quantità di dati su migliaia di server, proposto da Google nel 2004.
- Utilizza un approccio **senza condivisione** per gestire la distribuzione dei compiti.
- **Applicazioni comuni**: indicizzazione web, ordinamento distribuito, grafo inverso dei link web, statistiche di accesso web.

### Vantaggi per il programmatore

- **Semplicità**: nasconde la complessità dell'esecuzione distribuita, come parallelizzazione, bilanciamento del carico e tolleranza ai guasti.
- Offre un'**API semplice** per sviluppare programmi senza gestire i dettagli a livello di sistema.

### Fasi del modello MapReduce

1. **Map**:
 - Elabora un grande insieme di record.
 - Estrae informazioni rilevanti da ciascun record.
 - Ordina e mescola i risultati intermedi.
2. **Reduce**:
 - Aggrega i risultati intermedi 
 - Genera l'output finale.

### Struttura delle funzioni

- **Map**: `map(k1, v1) -> [(k2, v2)]`
- **Reduce**: `reduce(k2, [v2]) -> [(k3, v3)]`
 - Input e output (k,v) sono coppie **chiave-valore**.
 - Le chiavi possono non essere uniche e vengono utilizzate per raggruppare i dati durante il processo di riduzione.

L'idea chiave è fornire un'astrazione funzionale che consente di gestire grandi quantità di dati in modo distribuito e parallelo.

### Fase **Map**

- Applica una funzione su coppie **chiave-valore** dell'input per produrre nuove coppie.
 - `map(in_key, in_value) -> list(out_key, intermediate_value)`
- La funzione **Map** viene eseguita in parallelo su diverse macchine, poiché i dati di input sono partizionati in frammenti.
- La libreria MapReduce raccoglie i valori intermedi con la stessa chiave e li passa alla funzione **Reduce**.

### Fase **Reduce**

- Combina i valori intermedi associati alla stessa chiave per produrre un nuovo output.
 - `reduce(out_key, list(intermediate_value)) -> list(out_key, out_value)`
- I **reducer** operano in parallelo su chiavi diverse, consentendo ulteriore parallelismo.

### Esempio di MapReduce

- **Map**: Applica una funzione che calcola il quadrato di numeri.
  ```lisp
  (map square [1,2,3,4]) ; restituisce [1,4,9,16]
  ```
- **Reduce**: Somma i risultati ottenuti.
  ```lisp
  (reduce + [1,4,9,16]) ; restituisce 30
  ```

### Programma MapReduce

- Un **job MapReduce** (programma) contiene:
 - Codice delle funzioni **Map** e **Reduce**.
 - Parametri di configurazione (input/output su file system distribuito).
 - L'input viene suddiviso in **task**: mapper e reducer.
 - Tutti i **mapper** devono completare prima che inizi la fase di **reduce**.
- L'output è memorizzato sul file system distribuito, e un programma può includere più round di map e reduce.
### Fasi della Computazione MapReduce

1. **Task Map**: Ricevono frammenti di dati da un file system distribuito.
2. **Trasformazione**: I task Map trasformano i dati in coppie chiave-valore, secondo la logica definita dall'utente nella funzione Map.
3. **Raccolta e Ordinamento**: Le coppie chiave-valore vengono raccolte da un controller master e ordinate per chiave.
4. **Distribuzione ai Task Reduce**: Le chiavi sono divise tra i task Reduce, in modo che tutte le coppie con la stessa chiave siano inviate allo stesso task Reduce.
5. **Esecuzione Reduce**: I task Reduce elaborano una chiave alla volta, combinando i valori associati secondo la logica definita dall'utente nella funzione Reduce.
6. **Output**: Le coppie chiave-valore di output vengono memorizzate sul file system distribuito.
7. **Fasi successive**: L'output dei reducer può essere usato come input per una successiva fase MapReduce.

- Tra le fasi **Map** e **Reduce** avviene un'operazione chiamata **shuffle and sort**, che:
 - Trasferisce e unisce i dati intermedi dai mapper ai reducer.
 - Ordina i dati per chiave in modo distribuito.
- Le chiavi intermedie sono temporanee, memorizzate localmente sul disco delle macchine del cluster.
#### Esempio 

- **Map**: I mapper generano coppie intermedie a partire dai dati di input.
- **Reduce**: I reducer elaborano tutte le coppie con la stessa chiave.
- **Barriera**: Tra map e reduce avviene un grande ordinamento e raggruppamento distribuito, necessario per sincronizzare le fasi.

## "Hello World" in MapReduce: WordCount

- **Problema:** Contare il numero di occorrenze di ogni parola in una grande collezione di documenti
- **Input:** Repository di documenti, ogni documento è un elemento
- **Map:** Legge un documento ed emette una sequenza di coppie chiave-valore dove:
- Le chiavi sono le parole dei documenti e i valori sono uguali a 1:
	- (w1, 1), (w2, 1), ..., (wn, 1)
- **Shuffle and sort:** Raggruppa per chiave e genera coppie della forma:
	- (w1, [1, 1, ..., 1]), ..., (wn, [1, 1, ..., 1])
- **Reduce:** Somma tutti i valori ed emette (w1, k), ..., (wn, l)
- **Output:** Coppie (w, m) dove:
	- w è una parola che appare almeno una volta tra tutti i documenti di input
	- m è il numero totale di occorrenze di w tra tutti i documenti
### WordCount: Map

```python
def Map(String key, String value):
    # key: nome del documento
    # value: contenuto del documento
    for each word w in value:
        EmitIntermediate(w, "1")
```

Il Map emette ogni parola nel documento con un valore associato uguale a "1"

### WordCount: Reduce

```python
def Reduce(String key, Iterator values):
    # key: una parola
    # values: una lista di conteggi
    int result = 0
    for each v in values:
        result += ParseInt(v)
    Emit(AsString(result))
```

Il Reduce somma tutti gli "1" emessi per una data parola

---
### Esempio: WordLengthCount

- **Problema**: Contare quante parole di diverse lunghezze ci sono in una collezione di documenti.
- **Input**: Collezione di documenti, ciascun documento è un elemento.
- **Map**: Trasforma un documento in coppie (lunghezza parola, parola) per ogni parola del documento.
- **Shuffle and Sort**: Raggruppa le parole per lunghezza.
- **Reduce**: Conta il numero di parole per ogni lunghezza e genera coppie (lunghezza, numero di parole).
- **Output**: Coppie (lunghezza, numero totale di parole con quella lunghezza).

### Esempio: Moltiplicazione Matrice-Vettore

**Dati:**

* Matrice sparsa *A* di dimensione *n x n*
* Vettore *x* di dimensione *n x 1*
**Problema:**

Calcolare il prodotto matrice-vettore *y = A x x*.

**Soluzione con MapReduce:**

1. **Map:**
 * Genera coppie (i, *a<sub>ij</sub>x<sub>j</sub>*) moltiplicando gli elementi della matrice *A* per quelli del vettore *x*.
2. **Reduce:**
 * Somma i valori per ogni riga *i* della matrice per ottenere *y<sub>i</sub> = Σ a<sub>ij</sub>x<sub>j</sub>*, generando coppie (i, *y<sub>i</sub>*).
**Soluzione quando il vettore non entra in memoria:**

* Dividere *x* e *A* in blocchi più piccoli (strisce o blocchi quadrati) per eseguire il calcolo in parallelo. 

## Panoramica dell'esecuzione di MapReduce

**Architettura master-worker**: Il master coordina i task map e reduce, assegnando i task ai worker e monitorando l'esecuzione.
- **Gestione dei guasti**: 
 - Se il master si guasta, l'intero job viene riavviato.
 - Se un worker map si guasta, i task mappati devono essere rieseguiti.
 - Se un worker reduce si guasta, i task vengono riassegnati ad altri worker.

### Ottimizzazione: Combiner

 Migliora le prestazioni riducendo i dati intermedi grazie a una fase di combinazione locale prima del reduce.
- Il **combiner** esegue una riduzione preliminare, riducendo il traffico di rete e migliorando l'efficienza.
- È applicabile quando la funzione reduce è **associativa** e **commutativa**.

### Shuffle and Sort

Avviene tra le fasi Map(+Combiner) e Reduce, ordinando e ridistribuendo i dati verso i reducer corretti.
- Ogni mapper ordina i propri dati e li scrive su disco, poi i reducer scaricano e uniscono i file ordinati.

### Ottimizzazione: Partizionamento

Utilizza un partizionatore per assegnare le coppie chiave-valore ai reducer, garantendo una distribuzione efficiente delle chiavi tra i reducer.

### Flussi di lavoro MapReduce

Problemi complessi possono richiedere più job MapReduce concatenati, con l'output di un job che diventa l'input del successivo. Tuttavia, la gestione di file intermedi rallenta le prestazioni.

## Clustering k-means in MapReduce

#### Clustering

- Processo di esaminare una collezione di "punti" e raggrupparli in "cluster" secondo una certa misura di distanza
- Esempi di utilizzo: 
 - Segmentazione dei clienti
 - Clustering di azioni di mercato
 - Riduzione della dimensionalità

#### Distanza tra punti

- **Distanza euclidea**: Radice della somma dei quadrati delle differenze tra coordinate.
	-$d(p, q) = √((p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2)$
- **Distanza di Manhattan**: Somma dei valori assoluti delle differenze tra coordinate.
	- $\Sigma_{i=1}|p_i - q_i|$

#### Distanza tra cluster

- Si utilizza la distanza tra i **centroidi** (il punto medio dei punti dati di un cluster), che raramente coincide con uno dei punti dati originali.

#### Algoritmo k-means

Noto algoritmo di clustering appartenente alla classe degli algoritmi di assegnazione dei punti. Assume uno spazio euclideo. Esistono diverse euristiche per questo algoritmo, noi consideriamo la più semplice,l'**Algorirmo di Lloyd**
1. Definire il numero di cluster desiderato, **k**.
2. Selezionare **k punti** iniziali come centroidi.
3. Per ogni punto dati, assegnarlo al centroide più vicino e ricalcolare i centroidi.
4. Ripetere finché i cluster non migliorano più.

## Esecuzione di un'iterazione di k-means con MapReduce

#### Classificazione dei punti

- Ogni punto dati viene assegnato al centroide del cluster più vicino, minimizzando la distanza euclidea:
 $$
 z_i \leftarrow \arg\min_{j} \|u_j - x_i\|^2_2
 $$
 Dove $z_i$ è l'assegnazione del punto $x_i$ al centroide $u_j$.

#### Ricalcolo dei centroidi

- I centroidi vengono aggiornati calcolando la media dei punti assegnati a ciascun cluster:
 $$
 u_j = \frac{1}{n_j} \sum_{i:z_i=j} x_i
 $$
 Dove $n_j$ è il numero di punti nel cluster $j$ e $x_i$ sono i punti assegnati.

---
### MapReduce per la classificazione dei punti

- **Map**: Dato l'insieme di centroidi $\{u_j\}$ e un punto $x_i$, la funzione Map assegna il punto al centroide più vicino e emette la coppia chiave-valore $(z_i, x_i)$, dove:
 - $z_i$ è il cluster più vicino.
 - Parallelizzato sui punti dati.

$$ \begin{aligned}
map([μ_1, μ_2, ..., μ_k], x_i): \\
 z_i ← argmin_j ||u_j - x_i||^2_2 \\
 emit(z_i, x_i)
 \end{aligned}$$

Il **centroide assegnato** $z_i$ è la **chiave**, e il **punto dati** $x_i$ è il **valore**.

---
### MapReduce per il ricalcolo dei centroidi

- **Reduce**: Per ogni cluster $j$, la funzione Reduce calcola la media dei punti $x_i$ assegnati al cluster e produce il nuovo centroide.

```python
reduce(j, x_in_cluster_j: [x_1, ..., x_n]):
    sum = 0
    count = 0
    for x in x_in_cluster_j:
        sum += x
        count += 1
    emit(j, sum / count)
```
Il **nuovo centroide** $u_j$ viene calcolato come la **media** dei punti assegnati al cluster $j$.

---
### Iterazioni di k-means con MapReduce

- k-means richiede **più iterazioni** per convergere. In ogni iterazione:
 - **Map**: Assegna ciascun punto al centroide più vicino.
 - **Reduce**: Ricalcola i nuovi centroidi basati sui punti assegnati.

#### Ottimizzazione delle iterazioni

- Ogni iterazione necessita di trasmettere i nuovi centroidi a tutti i mapper.
- **Distribuzione intelligente dei punti**: Ogni mapper gestisce più punti, riducendo il numero totale di mapper.
- La procedura viene ripetuta finché non si raggiunge la **convergenza** (i centroidi non cambiano più) o si raggiunge un numero massimo di iterazioni.
