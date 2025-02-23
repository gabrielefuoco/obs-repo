
## Esempi di MapReduce


![[|325](_page_1_Figure_1.jpeg)]

* I mapper vengono applicati a tutte le coppie chiave-valore di input per generare un numero arbitrario di coppie intermedie.
* I reducer vengono applicati a tutti i valori intermedi associati alla stessa chiave intermedia.
* Tra la fase di map e la fase di reduce si trova una barriera che comporta un ordinamento e un raggruppamento distribuito su larga scala.


## "Hello World" in MapReduce: WordCount

* **Problema:** contare il numero di occorrenze di ogni parola in una grande collezione di documenti.
* **Input:** repository di documenti, ogni documento è un elemento.
* **Map:** legge un documento ed emette una sequenza di coppie chiave-valore dove:
    * Le chiavi sono le parole del documento e i valori sono uguali a 1:  `(w1, 1), (w2, 1), ..., (wn, 1)`.
* **Shuffle e sort:** raggruppa per chiave e genera coppie del tipo `(w1, [1, 1, ..., 1]), ..., (wp, [1, 1, ..., 1])`.
* **Reduce:** somma tutti i valori ed emette `(wi, k), ..., (wj, l)`.
* **Output:** coppie `(w, m)` dove:
    * `w` è una parola che appare almeno una volta in tutti i documenti di input e `m` è il numero totale di occorrenze di `w` in tutti quei documenti.

### WordCount: Map

La fase di *Map* del processo WordCount associa ad ogni parola nel documento il valore "1".

```
Map(String key, String value) :
// key: nome del documento
// value: contenuto del documento
for each word w in value:
  EmitIntermediate(w, "1")
```

### WordCount: Reduce

La fase di *Reduce* somma tutti gli "1" emessi per ogni parola.  Il risultato è il conteggio delle occorrenze di ciascuna parola.

```
Reduce(String key, Iterator values): 
// key: una parola
// values: una lista di "1" (conteggi)
int result = 0;
for each v in values:
  result += ParseInt(v);
Emit(AsString(result));
```


## Esempio: WordLengthCount

* **Problema:** contare quante parole di determinate lunghezze esistono in una collezione di documenti.
* **Input:** un repository di documenti, ogni documento è un elemento.
* **Map:** legge un documento ed emette una sequenza di coppie chiave-valore dove la chiave è la lunghezza di una parola e il valore è la parola stessa: `(i, w1), ..., (j, wn)`.
* **Shuffle e sort:** raggruppa per chiave e genera coppie del tipo `(1, [w1, ..., wk]), ..., (n, [wr, ..., ws])`.
* **Reduce:** conta il numero di parole in ogni lista ed emette: `(1, l), ..., (p, m)`.
* **Output:** coppie `(l, n)`, dove `l` è una lunghezza e `n` è il numero totale di parole di lunghezza `l` nei documenti di input.


## Ottimizzazione: combinazione

* I task di reduce non possono iniziare prima che l'intera fase di map sia completata.
* Come migliorare le prestazioni? Eseguendo una mini fase di reduce sull'output map locale, spostando così parte del lavoro dei reducer sui mapper precedenti.
* La funzione reduce deve essere associativa e commutativa. I valori da combinare possono essere combinati in qualsiasi ordine, con lo stesso risultato.
* Esempio: l'addizione nel Reduce di WordCount. In questo caso applichiamo un combiner all'output Map locale.
* `combine (k2, [v2]) → [(k3, v3)]`.


In molti casi, la stessa funzione può essere utilizzata per la combinazione e per la riduzione finale. Lo shuffle e il sort sono comunque necessari!

**Vantaggi:**

* Riduzione della quantità di dati intermedi.
* Riduzione del traffico di rete.


## WordCount con combiner

* **Problema:** contare il numero di occorrenze di ogni parola in una grande collezione di documenti.
* **Input:** repository di documenti, ogni documento è un elemento.
* **Map:** legge un documento ed emette una sequenza di coppie chiave-valore dove:
    * Le chiavi sono le parole del documento e i valori sono uguali a 1: `(W1, 1), (W2, 1), ..., (Wp, 1)`.
* **Combiner:** raggruppa per chiave, somma tutti i valori ed emette: `(W1, i), ..., (Wn, j)`.
* **Shuffle e sort:** raggruppa per chiave e genera coppie del tipo `(w4, [p, ..., q]), ..., (wn, [r, ..., s])`.
* **Reduce:** somma tutti i valori ed emette `(w, k), ..., (wj, l)`.
* **Output:** coppie `(w, m)` dove:
    * `w` è una parola che appare almeno una volta in tutti i documenti di input e `m` è il numero totale di occorrenze di `w` in tutti quei documenti.

![[|301](_page_10_Figure_1.jpeg)]


## Ottimizzazione: partizionamento

* Come dividere lo spazio delle chiavi intermedie in modo personalizzato? Tramite un partizionatore.
* Assegna le coppie chiave-valore intermedie ai reducer.

![[|415](_page_11_Figure_4.jpeg)]


## Workflow MapReduce

Un singolo job MapReduce ha una gamma limitata di problemi risolvibili.  Compiti più complessi richiedono la concatenazione di più job.

### Esempio: Trovare gli URL più popolari in un file di log

Questo esempio illustra la necessità di workflow MapReduce.  Il processo si articola in due fasi:

1. **Determinazione del numero di visualizzazioni per ogni URL:**  Questa fase conta le occorrenze di ogni URL nel file di log.

2. **Ordinamento degli URL per popolarità:** Questa fase ordina gli URL in base al numero di visualizzazioni, dal più popolare al meno popolare.  I mapper di questa seconda fase invertono chiavi e valori, usando la frequenza di visualizzazione come chiave e l'URL come valore, per facilitare l'ordinamento.

I job MapReduce possono essere concatenati in workflow, dove l'output di un job diventa l'input del successivo. Tuttavia, questa concatenazione genera file intermedi su sistemi di file distribuiti, che vengono letti e scritti da ogni job. Questo processo di lettura e scrittura su disco genera un significativo calo delle prestazioni.



### Esempio: k-means in MapReduce

Il clustering è il processo di esame di una collezione di "punti" e di raggruppamento dei punti in "cluster" in base ad una qualche misura di distanza.

* **Esempi di analisi cluster:**
    * Segmentazione dei clienti: ricerca di similarità tra gruppi di clienti.
    * Clustering del mercato azionario: raggruppare le azioni in base alle performance.
    * Riduzione della dimensionalità di un dataset raggruppando osservazioni con valori simili.

![[Pasted image 20250223162415.png|328]]


### Distanza tra punti

Prima dobbiamo definire la distanza tra due punti dati. La più popolare è la distanza euclidea. La distanza tra i punti p e q è data da:

$$d(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$$

dove n è il numero di variabili indipendenti nello spazio.

* Un'altra distanza popolare è la distanza di Manhattan. Si usa la somma dei valori assoluti invece dei quadrati. Si basa sulla geografia stradale a griglia del distretto di Manhattan a New York:

$$d_{Manhattan}(p,q) = \sum_{i=1}^{n} |p_i - q_i|$$


### Distanza tra cluster

* **Distanza del centroide:** Distanza tra i centroidi dei cluster. Il centroide è il punto che ha la posizione media di tutti i punti dati in ogni coordinata. 
	* **Esempio:** per i punti (-1, 10, 3), (0, 5, 2) e (1, 20, 10), il centroide si trova in ((-1+0+1)/3, (10+5+20)/3, (3+2+10)/3) = (0, 35/3, 5).
* Nota: il centroide non deve essere, e raramente lo è, uno dei punti dati originali.


### Clustering k-means

K-means è un algoritmo di clustering ben noto, appartenente alla classe di algoritmi di clustering di assegnazione di punti. I punti sono considerati in un certo ordine e ognuno è assegnato al cluster in cui si adatta meglio. K-means presuppone uno spazio euclideo.


Esistono diversi algoritmi euristici per k-means. Consideriamo l'algoritmo di Lloyd, il primo e più semplice. Cerca di minimizzare la somma dei quadrati all'interno del cluster.

1. Specificare il numero desiderato di cluster k;
2. Inizialmente scegliere k punti dati che probabilmente si trovano in cluster diversi;
3. Rendere questi punti dati i centroidi dei loro cluster;
4. Ripetere:
    * Per ogni punto dati p rimanente:
        * Trovare il centroide a cui p è più vicino;
        * Aggiungere p al cluster di quel centroide;
        * Ricalcolare i centroidi del cluster;
    * Finché non viene apportato alcun miglioramento;


## MapReduce di 1 iterazione di k-means

* **Classifica:** assegna ogni punto al centroide del cluster più vicino.

$$z_i \leftarrow \arg\min_j \left\|\mu_j - x_i\right\|_2^2$$

* **Ricentra:** aggiorna i centroidi del cluster come media dei punti assegnati.

$$\mu_j = \frac{1}{n_j} \sum_{i:z_i=j} x_i$$

dove: μj: centroide per il cluster j; nj: numero di elementi nel cluster j.


### MapReduce di 1 iterazione di k-means

* **Classifica:** assegna ogni punto al centroide del cluster più vicino.

$$z_i \leftarrow \arg\min_j \left\|\mu_j - x_i\right\|_2^2$$

**Map:** dato (`{μj}`, `xi`), per ogni punto emetti (`zi`, `xi`). Parallelo sui punti dati.

* **Ricentra:** aggiorna i centroidi del cluster come media dei punti assegnati.

$$\mu_j = \frac{1}{n_j} \sum_{i:z_i=j} x_i$$

**Reduce:** media su tutti i punti nel cluster j (`z = j`). Parallelo sui centroidi del cluster.


## Fase di classificazione come Map

* **Classifica:** assegna ogni punto al centroide del cluster più vicino.

$$z_i \leftarrow \arg\min_j \left\|\mu_j - x_i\right\|_2^2$$

```
map([μ1, μ2, …, μk], xi)
zi ← arg min || uj – xi ||2
emit (zi, xi)
```

`zi` è l'ID del cluster (chiave); il punto dati `xi` è il valore.


### Fase di ricentramento come Reduce

* **Ricentra:** aggiorna i centroidi del cluster come media dei punti assegnati.

$$\mu_j = \frac{1}{n_j} \sum_{i:z_i=j} x_i$$

```
reduce(j, x_in_clusterj: [xi, ...])
sum = 0
count = 0
for x in x_in_clusterj:
  sum += x
  count += 1
emit (j, sum/count)
```

Reduce sui punti dati assegnati al cluster j (hanno chiave j). Emetti il nuovo centroide per il cluster j.


### Iterazioni multiple per k-means

K-means necessita di una versione iterativa di MapReduce. Ogni mapper deve ottenere un punto dati e tutti i centroidi del cluster. Ciò genera troppi mapper! Una migliore implementazione: ogni mapper ottiene molti punti dati.


Ad ogni nuova iterazione, è necessario trasmettere i nuovi centroidi all'intero cluster MapReduce e ripetere più fasi di Map e Reduce fino alla convergenza (o al numero massimo di passaggi).

