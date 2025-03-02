
## MapReduce: Esempi e Ottimizzazioni

MapReduce è un modello di programmazione per elaborare grandi dataset distribuiti.  Il processo si divide in tre fasi principali:

**1. Map:** I *mapper* applicano una funzione a ogni coppia chiave-valore di input, generando coppie chiave-valore intermedie.  ![[]]

**2. Shuffle e Sort:** Le coppie intermedie vengono raggruppate per chiave tramite un ordinamento e raggruppamento distribuito.

**3. Reduce:** I *reducer* elaborano i valori associati a ciascuna chiave intermedia, producendo l'output finale.


**Esempio 1: WordCount**

Questo esempio conta le occorrenze di ogni parola in un corpus di documenti.

* **Input:**  Un insieme di documenti.
* **Map:**  Per ogni documento, genera coppie `(parola, 1)`.  Esempio di codice:
```
Map(String key, String value) : 
  for each word w in value: 
    EmitIntermediate(w, "1") 
```
* **Shuffle e Sort:** Raggruppa le coppie per parola.
* **Reduce:** Somma gli "1" per ogni parola, generando coppie `(parola, conteggio)`. Esempio di codice:
```
Reduce(String key, Iterator values): 
  int result = 0; 
  for each v in values: 
    result += ParseInt(v); 
  Emit(AsString(result));
```
* **Output:** Coppie `(parola, conteggio)` indicando il numero di occorrenze di ogni parola.


**Esempio 2: WordLengthCount**

Questo esempio conta le parole in base alla loro lunghezza.

* **Input:** Un insieme di documenti.
* **Map:** Genera coppie `(lunghezza, parola)`.
* **Shuffle e Sort:** Raggruppa le coppie per lunghezza.
* **Reduce:** Conta le parole per ogni lunghezza, generando coppie `(lunghezza, numero di parole)`.
* **Output:** Coppie `(lunghezza, numero di parole)` indicando il numero di parole di ogni lunghezza.


**Ottimizzazione: Combinazione**

Per migliorare le prestazioni, si può introdurre una fase di *combinazione* tra Map e Reduce.  Questa fase esegue una mini-riduzione sull'output locale di ogni mapper, riducendo il carico sui reducer.  La funzione di combinazione deve essere associativa e commutativa.  Nel WordCount, ad esempio, la combinazione potrebbe sommare gli "1" localmente prima di inviarli al reducer.  La funzione `combine (k2, [v2]) → [(k3, v3)]`  spesso può essere la stessa funzione del reducer.  Questo permette ai task di reduce di iniziare prima del completamento della fase map, migliorando l'efficienza complessiva.

---

## MapReduce: WordCount, Ottimizzazione e Workflow

Questo documento descrive il funzionamento di MapReduce, focalizzandosi su ottimizzazioni e workflow per compiti complessi.

### WordCount con Combiner

Il problema del conteggio delle occorrenze di ogni parola in un corpus di documenti viene risolto tramite MapReduce.

* **Input:**  Collezione di documenti.
* **Map:**  Produce coppie (parola, 1) per ogni parola in ogni documento.
* **Combiner:**  Raggruppa le coppie per parola, sommando i valori (conteggi).
* **Shuffle e Sort:**  Raggruppa ulteriormente le coppie per parola, preparando l'input per il reducer.
* **Reduce:**  Somma i conteggi per ogni parola.
* **Output:** Coppie (parola, conteggio totale).  `![[]]` illustra il processo.


### Ottimizzazione: Partizionamento

Il partizionamento personalizzato dello spazio delle chiavi intermedie, tramite un partizionatore, assegna efficientemente le coppie chiave-valore ai reducer, migliorando le prestazioni. `![[]]` mostra un esempio di partizionamento.


### Workflow MapReduce

Compiti complessi richiedono workflow MapReduce, concatenando più job. L'output di un job diventa l'input del successivo.  Questo genera però file intermedi su sistemi di file distribuiti, causando un calo di prestazioni.

**Esempio: URL più popolari:**

Questo workflow comprende due fasi:

1. Conteggio delle visualizzazioni per ogni URL (simile al WordCount).
2. Ordinamento degli URL per popolarità (invertendo chiave e valore nel mapper per facilitare l'ordinamento).


### Esempio: k-means in MapReduce

Il k-means, un algoritmo di clustering, può essere implementato con MapReduce.  Il clustering raggruppa punti dati in base alla distanza (es. distanza euclidea:  $$d(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$$, o distanza di Manhattan).  Applicazioni includono la segmentazione di clienti e l'analisi del mercato azionario. `![[]]` mostra un esempio di clustering.

---

Il documento descrive l'implementazione dell'algoritmo di clustering k-means usando MapReduce.  L'algoritmo inizia definendo una distanza, ad esempio la distanza di Manhattan ($$d_{Manhattan}(p,q) = \sum_{i=1}^{n} |p_i - q_i|$$), e poi calcola il centroide di un cluster come la media delle coordinate dei punti che lo compongono.

K-means iterativamente assegna punti ai cluster più vicini (**fase di classificazione**) e poi ricalcola i centroidi basandosi sui punti assegnati (**fase di ricentramento**).  La fase di classificazione assegna ogni punto  `xi` al cluster `zi` con centroide più vicino, usando la formula: $$z_i \leftarrow \arg\min_j \left\|\mu_j - x_i\right\|_2^2$$. La fase di ricentramento aggiorna i centroidi  `μj` calcolando la media dei punti assegnati a ciascun cluster: $$\mu_j = \frac{1}{n_j} \sum_{i:z_i=j} x_i$$.

L'implementazione MapReduce di una singola iterazione di k-means suddivide queste due fasi. La **fase Map** (classificazione) assegna ogni punto al cluster più vicino, emettendo coppie (`zi`, `xi`). La **fase Reduce** (ricentramento) calcola la media dei punti per ogni cluster, emettendo i nuovi centroidi.  Il codice Map e Reduce è fornito per illustrare questo processo:

```
map([μ1, μ2, …, μk], xi)
zi ← arg min || uj – xi ||2
emit (zi, xi)
```

```
reduce(j, x_in_clusterj: [xi, ...])
sum = 0
count = 0
for x in x_in_clusterj:
    sum += x
    count += 1
emit (j, sum/count)
```

Per iterazioni multiple, invece di inviare tutti i centroidi ad ogni mapper, una soluzione più efficiente prevede che ogni mapper riceva molti punti dati.  L'algoritmo continua iterativamente fino a convergenza, ovvero fino a quando non si osservano miglioramenti significativi nella posizione dei centroidi.

---

L'algoritmo K-means, implementato con MapReduce, iterativamente affina i centroidi dei cluster.  Ad ogni iterazione, i nuovi centroidi calcolati vengono distribuiti a tutti i nodi del cluster MapReduce.  Il processo di Map e Reduce viene ripetuto fino al raggiungimento della convergenza, ovvero quando i centroidi non cambiano significativamente tra un'iterazione e l'altra, oppure fino al raggiungimento di un numero massimo di iterazioni predefinito.

---
