L'analisi delle regole associative è una tecnica di data mining che identifica relazioni significative tra attributi in un dataset. L'obiettivo è scoprire regole che indicano la presenza di un elemento in una transazione sulla base della presenza di altri elementi.

### Concetti chiave
* **Itemset frequenti:** insiemi di elementi che compaiono frequentemente insieme in molte transazioni.
* **Regole di associazione:** rappresentano le relazioni tra due itemset, indicando la co-occorrenza e non la casualità.
* **Variabile binaria asimmetrica:** modello utilizzato per gli item, dove la presenza di un item in una transazione è considerata più importante della sua assenza.

### Itemset e Support Count

**Scopo:** individuare itemset che tendono a comparire frequentemente insieme in un insieme di transazioni.

**Definizioni:**

* **Insieme di elementi (I):**  $I = \{i_1, i_2, \dots, i_d\}$
* **Insieme di transazioni (T):** $T = \{t_1, t_2, \dots, t_n\}$, dove ogni*ogni* $t_j$ contiene un sottoinsieme degli elementi di I.
* **Itemset (X):** sottinsieme degli elementi di I.

**Metriche chiave:**

* **Support count (σ(X)):** numero di transazioni che contengono X:
    $\sigma(X) = |\{t_i | X \subset t_i, t_j \in T\}|$
* **Support (s(X)):** frazione di transazioni che contengono X:
    $s(X) = \frac{\sigma(X)}{N}$

**Frequent Itemset:** un itemset X è considerato **Frequent Itemset** se $s(X) \ge minsup$, dove minsup è una soglia minima di supporto definita dall'utente.

**Obiettivo:** individuare tutti gli itemset frequenti, ovvero gli insiemi di elementi che ricorrono insieme in almeno una frazione minsup delle transazioni. Questi rappresentano le associazioni o pattern frequenti all'interno dei dati.

## Regole di Associazione

Le regole di associazione sono un concetto chiave nell'analisi di associazione, che mira a identificare relazioni frequenti tra insiemi di elementi (itemset) all'interno di un insieme di transazioni.

**Definizione:** Una regola di associazione è rappresentata come X → Y, dove X e Y sono insiemi disgiunti di elementi ($X \cap Y = \emptyset$). La forza di una regola è misurata attraverso due metriche:

* **Supporto:** La frazione di transazioni che contengono sia X che Y. Misura quanto spesso la regola è applicabile ai dati.
    $$S(X \cup Y) = \frac{\sigma(X \cup Y)}{N}$$
    Dove $\sigma(X \cup Y)$ è il numero di transazioni che contengono sia X che Y, e N è il numero totale di transazioni.

* **Confidenza:** La frequenza con cui Y compare nelle transazioni contenenti X. Valuta l'affidabilità della regola.
    $$c(X \to Y) = \frac{\sigma(X \cup Y)}{\sigma(X)}$$
    Dove $\sigma(X)$ è il numero di transazioni che contengono X.

**Scopo:** L'obiettivo è individuare regole forti, ovvero con supporto e confidenza elevati rispetto alle soglie fissate. Queste regole rappresentano pattern frequenti e affidabili nei dati.

#### Scoperta delle Regole di Associazione:

Il problema della scoperta delle regole di associazione consiste nel trovare, dato un insieme di transazioni D, tutte le regole X→Y che soddisfano:

* **Supporto:** $\text{Support} \ge \text{minsup}$
* **Confidenza:** $\text{Confidenza} \ge \text{minconf}$

Dove minsup e minconf sono soglie definite dall'utente.

**Approccio Brute-Force:**

Un approccio brute-force calcolerebbe supporto e confidenza per tutte le possibili regole. Il numero totale di regole R su un dataset con d elementi distinti è esponenziale:

$$R = 3^d - 2^{d+1} + 2$$

Dove il numero totale di itemset è pari a $2^d$.

Questo approccio è proibitivamente costoso.

**Approccio Efficiente:**

Gli algoritmi tipicamente scompongono il problema in due sotto-task:

1. **Generazione degli Itemset Frequenti:** Trovare tutti gli itemset che soddisfano minsup.
2. **Generazione delle Regole:** Per ogni itemset frequente L, generare regole $f \to (L-f)$ con f sottoinsieme non vuoto di L, aventi $\text{confidenza} \ge \text{minconf}$.

Questa decomposizione sfrutta il fatto che il supporto di X→Y è uguale a quello di X∪Y. Quindi, si possono subito escludere regole con itemset non frequente.

**Generazione degli Itemset Frequenti:**

La chiave è l'efficiente generazione degli itemset frequenti, ancora un problema complesso, svolto tipicamente con algoritmi ad hoc come Apriori.

**In sintesi:** Le formule permettono di quantificare il numero totale di regole potenziali e motivano l'approccio suddiviso in generazione degli itemset frequenti e successiva generazione delle regole.


### Frequente Itemset Generation

Approccio "brute-force" per trovare gli itemset frequenti in un insieme di transazioni, e le strategie per ridurre la sua complessità computazionale.

**Approccio "brute-force"**

* **Scopo:** Trovare tutti gli itemset frequenti in un insieme di transazioni.
* **Procedura:**
    1. Generare tutti gli itemset candidati, escludendo quelli vuoti o singoletti.
    2. Per ogni candidato, contare il suo supporto confrontandolo con ogni transazione.
* **Complessità:** $O(NMw)$, dove:
    * N è il numero di transazioni.
    * M è il numero di itemset candidati ($O(2^d)$ con d elementi distinti).
    * w è la dimensione massima di una transazione.
* **Problema:** La complessità esponenziale in d rende l'approccio inefficiente.

**Strategie per ridurre la complessità:**

1. **Ridurre M (numero di candidati):**
    * Utilizzare il principio di Apriori per escludere molti candidati senza calcolare il supporto.
2. **Ridurre N (numero di transazioni):**
    * Possibile per itemset di grandi dimensioni.
3. **Ridurre NM (numero totale di confronti):**
    * Usare strutture dati efficienti per memorizzare candidati e transazioni. 

## Generazione di Itemset Frequenti: Il Principio Apriori

### Misure di Validità delle Regole

Per valutare la validità di una regola associativa, si utilizzano due misure:

* **Supporto:** La frequenza con cui un itemset appare nell'insieme di dati.
* **Confidenza:** La probabilità che un itemset appaia dato che un altro itemset è già presente.

Il problema è che il numero di itemset possibili cresce esponenzialmente con il numero di elementi, rendendo la ricerca di regole associative computazionalmente costosa.

### Il Principio Apriori

Il principio Apriori afferma che se un itemset è frequente, allora tutti i suoi sottoinsiemi sono anch'essi frequenti. Questo principio si basa sulla proprietà di **antimonotonia** degli itemset: aumentando il numero di elementi in un itemset, il suo supporto diminuisce.

Formalmente, il principio Apriori può essere espresso come:

$\forall X,Y:(X⊆Y)⇒s(X)≥s(Y)$

Dove:

* $X$ e $Y$ sono itemset.
* $s(X)$ è il supporto di $X$.

#### Esempio
Consideriamo un insieme di transazioni con una soglia di supporto del 60%. L'algoritmo Apriori inizia generando itemset candidati di dimensione 1. Gli itemset che non raggiungono la soglia di supporto vengono scartati.

Nella successiva iterazione, vengono generati itemset candidati di dimensione 2 utilizzando solo gli itemset frequenti di dimensione 1. Questo processo continua fino a quando non vengono trovati tutti gli itemset frequenti.

L'utilizzo del principio Apriori riduce significativamente il numero di itemset candidati da considerare, migliorando l'efficienza dell'algoritmo.
#### Efficacia del Pruning

L'efficacia del pruning basato sul supporto di Apriori può essere dimostrata contando il numero di itemset candidati generati. In un esempio con 6 elementi, una strategia di forza bruta genererebbe 41 candidati. Con il principio Apriori, questo numero si riduce a 13, rappresentando una riduzione del 68%.


## Algoritmo Apriori

L'algoritmo Apriori è un algoritmo per la scoperta di itemset frequenti in un insieme di dati di transazioni. L'obiettivo è trovare tutti gli itemset che appaiono in almeno una certa percentuale di transazioni, definita come *minsup*.
**Descrizione dell'Algoritmo:**
L'algoritmo Apriori opera in modo iterativo, partendo da itemset di dimensione 1 e aumentando gradualmente la dimensione degli itemset fino a quando non vengono trovati tutti gli itemset frequenti.

**Fase 1: Inizializzazione**

* L'algoritmo inizia calcolando il supporto di ogni singolo item (1-itemset).
* Gli itemset che superano la soglia di supporto *minsup* vengono inclusi nell'insieme *F1* degli itemset frequenti di dimensione 1.

**Fase 2: Generazione e Pruning di Candidati**

* In ogni iterazione successiva, l'algoritmo genera candidati itemset di dimensione *k* (k-itemset) a partire dagli itemset frequenti di dimensione *k-1*.
* I candidati vengono poi sottoposti a un processo di pruning, in cui vengono eliminati tutti i candidati che non soddisfano il principio Apriori. Questo principio afferma che se un itemset è frequente, allora tutti i suoi sottoinsiemi devono essere anch'essi frequenti.

**Fase 3: Conteggio del Supporto**

* L'algoritmo scansiona l'insieme di dati per contare il supporto di ogni candidato itemset.

**Fase 4: Eliminazione dei Candidati Infrequenti**

* Gli itemset candidati che non raggiungono la soglia di supporto *minsup* vengono eliminati.

**Fase 5: Terminazione**

* L'algoritmo termina quando non vengono generati nuovi itemset frequenti.

### Pseudocodice
``` pseudo
k = 1
F_k = {i | i ∈ I and σ(i) ≥ N × minsup}
repeat
    k = k + 1
    Lk = candidate-gen(Fk−1)
    Lk = candidate-prune(Lk, Fk−1)
    for all transaction t ∈ T do
        Lt = subset(Lk, t)
        for all candidate itemset c ∈ Lt do
            σ(c) = σ(c) + 1
        end for
    end for
    Fk = {c | c ∈ Lk and σ(c) ≥ N × minsup}
until Fk = ∅
Result =  ∪ F_{k}

```


Dove:
* *N* è il numero totale di transazioni.
* *minsup* è la soglia di supporto minima.
* *I* è l'insieme di tutti gli item.
* *σ(i)* è il supporto dell'item *i*.
* *Fk* è l'insieme degli itemset frequenti di dimensione *k*.
* *Lk* è l'insieme dei candidati itemset di dimensione *k*.
* *candidate-gen()* è la funzione che genera i candidati itemset.
* *candidate-prune()* è la funzione che elimina i candidati infrequenti.
* *subset()* è la funzione che determina tutti i candidati itemset contenuti in una transazione.

## Generazione di Candidati con Forza Bruta e Pruning

L'approccio di forza bruta per generare itemset di una data cardinalità (ad esempio, 3) consiste nel creare tutti i possibili itemset di quella cardinalità. Per 6 item, questo significa generare $\begin{pmatrix} 6 \\ 3 \end{pmatrix}= 20$ itemset. 

Il numero di itemset da generare dipende dal numero totale di item (*n*) e dalla cardinalità desiderata (*k*): $\begin{pmatrix}n \\k\end{pmatrix}$.

Per ridurre il numero di itemset da verificare, si può applicare il **pruning**. Questo processo elimina gli itemset candidati che contengono almeno un itemset non frequente. In questo modo, si evita di verificare itemset che sicuramente non saranno frequenti. 

## Generazione di Candidati con il Principio Apriori

### Metodo $F_{k−1}×F_{k−2}$

Questo metodo genera itemset di cardinalità *k* combinando itemset frequenti di cardinalità *k −*1 e *k −*2. Il processo prevede:

1. **Calcolo degli itemset frequenti di cardinalità 1 e 2.**
2. **Merge:** Combinare gli itemset frequenti di cardinalità *k −*1 e *k −*2 se i loro prefissi sono identici.
3. **Pruning:** Eliminare gli itemset candidati che contengono almeno un itemset non frequente.
4. **Conteggio del supporto:** Verificare la frequenza degli itemset candidati rimanenti.

### Metodo $F_{k−1}×F_{k−1}$

Questo metodo ottimizza la generazione di candidati combinando due itemset di cardinalità *k −*1. Il processo prevede:

1. **Merge:** Combinare due itemset di cardinalità *k −*1 se i loro prefissi sono uguali.
2. **Pruning:** Eliminare gli itemset candidati che contengono almeno un itemset non frequente.
3. **Conteggio del supporto:** Verificare la frequenza degli itemset candidati rimanenti.

**Esempio:**

![[7) Regole associative-20241004203853657.png|404]]

In questo esempio, gli itemset *ABD* e *ACD* non vengono combinati perché i loro prefissi sono uguali solo per un elemento, non per due.

**Alternative:**
Esistono altre tecniche che considerano i suffissi o combinazioni di prefissi e suffissi. La scelta della tecnica più efficace dipende dal caso specifico.

**Principio generale:**
Due itemset possono essere uniti se differiscono di un solo elemento.

**Vantaggi:**
Queste tecniche sono più efficienti rispetto alla generazione di candidati con forza bruta, riducendo il numero di itemset da verificare.

## Generazione di Regole Associative

**Partizionamento e Confidenza**
La generazione di regole associative prevede la divisione di un itemset frequente in due parti:

- **Corpo**: Gli elementi che compaiono nel lato sinistro della regola.
- **Testa**: Gli elementi che compaiono nel lato destro della regola.

La confidenza di una regola è data dal rapporto tra la frequenza dell'itemset completo e la frequenza del corpo della regola.

**Esempio**:
Dato l'itemset frequente {A, B, C, D}, possiamo generare le seguenti regole:

![[7) Regole associative-20241005111740790.png]]

**Proprietà di Anti-Monotonicità**
La confidenza di una regola non è anti-monotona rispetto al numero di elementi nel corpo. Ad esempio, la confidenza di ABC →D non è necessariamente maggiore o minore della confidenza di AB →D.

Tuttavia, la confidenza è anti-monotona rispetto al numero di elementi nella testa. Questo significa che spostando elementi dalla testa al corpo, la confidenza diminuisce.

**Esempio**:
$c(ABC \to D) \ge c(AB \to CD) \ge c(A \to BCD)$

**Pruning delle Regole**
Il principio Apriori può essere applicato anche alla generazione di regole. Se una regola non è frequente, tutte le regole ottenute spostando elementi dalla testa al corpo non avranno una confidenza adeguata. Questo permette di eliminare un numero significativo di regole candidate.

**Complessità**
Nonostante il pruning, il problema della generazione di regole rimane esponenziale. Il numero di regole possibili cresce esponenzialmente con il numero di elementi nell'itemset.

## Complessità degli Algoritmi per le Regole Associative

Il processo di scoperta di regole associative presenta due sfide principali:

1. **Numero esponenziale di itemset e regole:** Il numero di possibili itemset e regole cresce esponenzialmente con la dimensione del dataset.
2. **Calcolo del supporto e della confidenza:** Il calcolo della frequenza di ogni itemset e della confidenza di ogni regola richiede un tempo significativo, soprattutto per dataset di grandi dimensioni.


La complessità degli algoritmi per le regole associative è influenzata da diversi fattori:

* **Soglia minima di supporto (e confidenza):** Più bassa è la soglia, più itemset diventano frequenti, aumentando la complessità.
* **Dimensionalità:** La dimensione del reticolo di itemset dipende dal numero di elementi nel dataset.
* **Dimensione del database:** Il calcolo del supporto richiede la scansione di tutte le transazioni.
* **Larghezza media delle transazioni:** Transazioni più grandi aumentano la complessità.
* **Strutture di memorizzazione:** L'utilizzo di strutture dati efficienti può migliorare le prestazioni.

### Calcolo del Supporto

La complessità del calcolo del supporto è data da *O*(*NMw*) confronti, dove:

* **N:** numero di transazioni
* **M:** numero di itemset candidati, pari a $O(2^d)$ dove *d* è la dimensione massima degli itemset
* **w:** larghezza massima della transazione

### Ottimizzazione con Strutture ad Hash

Le strutture ad hash possono migliorare il calcolo del supporto. Invece di confrontare ogni transazione con tutti gli itemset candidati, si può utilizzare una struttura ad accesso diretto per accedere solo ai bucket corrispondenti all'hash della transazione.

**Esempio:**

![[7) Regole associative-20241004203906101.png|424]]

![[7) Regole associative-20241004203918187.png|434]]

In questo esempio, un albero è utilizzato per organizzare gli itemset in base ai loro elementi. Ogni nodo dell'albero rappresenta un elemento, e i nodi foglia (bucket) contengono gli itemset che iniziano con gli elementi corrispondenti.

### Calcolo del Numero di Itemset Totali

Il numero di itemset totali contenuti nei bucket è dato dalla formula:

$$
\binom{k}{n} = \frac{k!}{n!(k-n)!}
$$

dove *k* è il numero di elementi e *n* è la dimensione degli itemset.


## Hash Tree per la Generazione di Regole Associative

### Struttura dell'Hash Tree

Un hash tree è una struttura dati ad albero che organizza gli itemset candidati in base ad una funzione di hash. Ogni nodo dell'albero rappresenta un valore di hash, e i figli di un nodo sono gli itemset che hanno quel valore di hash.

**Esempio:**

![[7) Regole associative-20241004204004920.png|414]]

* La funzione di hash utilizzata in questo esempio è *x*%3, che restituisce il resto della divisione di *x* per 3.
* I nodi dell'albero sono etichettati con i valori di hash possibili (1, 2, 0).
* Gli itemset sono organizzati in base al valore di hash del loro primo elemento.
* Se un nodo foglia contiene più itemset del limite massimo, viene suddiviso applicando la funzione di hash al secondo elemento degli itemset.

### Matching tra Hash Tree e Transazione

Per calcolare il supporto di un itemset, è necessario verificare se l'itemset è contenuto in una transazione. L'hash tree può essere utilizzato per velocizzare questo processo.

**Esempio:**

![[7) Regole associative-20241004204015804.png|495]]

* Si crea un albero che rappresenta la transazione, organizzando gli elementi della transazione in base al loro valore di hash.
* Si confrontano le due strutture ad albero (hash tree e albero della transazione) per trovare gli itemset che sono sottoinsiemi della transazione.
* Questo processo di matching limita il numero di confronti necessari, evitando di dover controllare tutti i possibili sottoinsiemi della transazione.

### Vantaggi dell'Hash Tree

L'utilizzo dell'hash tree offre diversi vantaggi:

* **Riduzione del numero di confronti:** L'hash tree permette di limitare il numero di confronti necessari per verificare se un itemset è contenuto in una transazione.
* **Miglioramento delle prestazioni:** L'hash tree velocizza il processo di calcolo del supporto, migliorando le prestazioni degli algoritmi per le regole associative.

## Rappresentazione Compatta degli Itemset Frequenti

### Problema della Complessità

Il numero di possibili itemset cresce esponenzialmente con la dimensione del dataset. Ad esempio, con 10 elementi per categoria e 3 categorie, il numero totale di itemset è dato da:
$$3 \times \sum_{k=0}^{10} \begin{pmatrix} 10 \\k{}\end{pmatrix} 10 \\k{}=3 \times(2^{10}-1)$$
Questo rende inefficiente il calcolo di tutti gli itemset.

### Itemset Massimali

Un itemset è **massimale** se nessuno dei suoi immediati superset è frequente. Ad esempio, se il supporto minimo è 5, l'itemset *F* è massimale perché non esiste un superset di *F* con supporto almeno 5.

### Vantaggi degli Itemset Massimali

* **Riduzione della complessità:** Gli itemset massimali sono un sottoinsieme molto limitato di tutti i possibili itemset.
* **Efficienza:** Calcolare solo gli itemset massimali riduce significativamente il tempo di calcolo.

### Principio apriori

Il principio apriori afferma che se un itemset è frequente, tutti i suoi sottoinsiemi sono anch'essi frequenti. Questo principio può essere utilizzato per eliminare gli itemset non massimali durante il processo di scoperta delle regole associative.

**Esempio:**

![[7) Regole associative-20241004204050883.png|474]]

Se un itemset è frequente (ad esempio, *F* nel reticolo), tutti i suoi sottoinsiemi (ad esempio, *E*, *F*, *J*) sono anch'essi frequenti. Quindi, non è necessario calcolare i sottoinsiemi di un itemset frequente, perché non possono essere massimali.

### Itemset Chiusi

Un itemset è **chiuso** se nessuno dei suoi immediati superset ha lo stesso valore di supporto. In altre parole, un itemset chiuso non può essere esteso aggiungendo altri elementi senza diminuire il suo supporto.

**Esempio:**

![[7) Regole associative-20241004204116719.png|462]]

* L'itemset {B} è chiuso perché nessun superset di {B} ha lo stesso supporto di {B}.
* L'itemset {A, B} è chiuso perché nessun superset di {A, B} ha lo stesso supporto di {A, B}.

### Valutazione delle Regole

Le regole associative vengono valutate utilizzando misure come:

* **Supporto:** La proporzione di transazioni che contengono l'itemset associato alla regola.
* **Confidenza:** La proporzione di transazioni che contengono l'itemset associato alla regola, tra tutte le transazioni che contengono l'antecedente della regola.

Queste misure sono **oggettive**, ma spesso vengono utilizzate anche misure **soggettive** che dipendono dal contesto.

### Problemi nella Valutazione delle Regole

* **Supporto non omogeneo:** La scelta della soglia di supporto può essere difficile, poiché un supporto troppo basso può generare un numero elevato di itemset frequenti, mentre un supporto troppo alto può far perdere regole importanti.
* **Pattern cross support:** Le regole possono contenere item con supporto molto diverso, rendendo la regola poco interessante per l'analisi.

## 15.2.1 Tabella di Contingenza

La tabella di contingenza è uno strumento utile per valutare le regole associative del tipo _X → Y_. La tabella contiene quattro valori che rappresentano il supporto di diverse combinazioni di X e Y

$$
\begin{aligned}
s_{XY}, \ \text{Support count di } X \text{ e } Y \\
s_{\overline{X}Y}, \ \text{Support count di } X \text{ e } \bar{Y} \\
s_{X\overline{Y}}, \ \text{Support count di } \bar{X} \text{ e } Y \\
s_{\overline{X}\overline{Y}},\ \text{Support count di } \bar{X} \text{ e } \bar{Y} 
\end{aligned}
$$

![[7) Regole associative-20241004204128505.png|321]]

Questi valori possono essere utilizzati per calcolare il supporto della regola _X → Y_:
$\text{supporto}_{xy}=\frac{S_{xy}}{N}$
dove _N_ è il numero totale di oggetti.

Un criterio importante per la valutazione delle regole è che la confidenza della regola _c(X → Y)_ sia maggiore del supporto di _Y_ (s_Y). Questo significa che la regola è più informativa rispetto a semplicemente sapere che _Y_ è presente nel dataset.

La tabella di contingenza fornisce un quadro completo per analizzare le relazioni tra _X_ e _Y_ e per valutare l'utilità della regola _X → Y_

## Indipendenza Statistica

La condizione per una regola associativa _X → Y_ di essere interessante è che la probabilità condizionata di _Y_ dato _X_ sia maggiore della probabilità di _Y_. Questo può essere espresso come:

$P (Y |X) > P (Y ) =\frac {P (X, Y )}{ P (X) > P (Y )} = P (X, Y ) > P (X) × P (Y )$

Quando $P(X, Y) > P(X) \times P(Y)$  è soddisfatta si dice che le due variabili sono **correlate positivamente**, negativamente altrimenti. 
Quando $P(X, Y) = P(X) \times P(Y)$  abbiamo **indipendenza**. 
### Misure di Correlazione
Diverse misure vengono utilizzate per quantificare la correlazione tra _X_ e _Y_:

* **Lift**: Misura il rapporto tra la probabilità condizionata di _Y_ dato _X_ e la probabilità di _Y_:
	* $\text{Lift}=\frac{P(Y|X)}{P(Y)}$
	* Utilizzato per le regole associative.
* **Interest**: Misura la differenza tra la probabilità congiunta di _X_ e _Y_ e il prodotto delle probabilità marginali di _X_ e _Y_:
	* $\text{Interest}=\frac{P(X,Y)}{P(X)\times P(Y)}$
	* Utilizzato per gli itemset.
* **ϕ-coefficient:** Misura la correlazione tra due variabili binarie.
$$\phi_\text{coefficient}=\frac{P(X,Y)-P(X)\times P(Y)}{\sqrt{ P(X) \times [1 - P(X)] \times P(Y) \times [1 - P(Y)] }}$$

### Pruning delle Regole
Le misure di correlazione possono essere utilizzate per il **pruning** delle regole associative. Le regole che non superano una certa soglia di correlazione possono essere eliminate.:

![[7) Regole associative-20241004204151553.png|353]]
In questo esempio, la regola con correlazione più alta (0.9) ha un valore di _Lift_ inferiore rispetto alla regola con correlazione più bassa. Questo dimostra che il _Lift_ non è sempre un indicatore affidabile della forza di una regola.

## Proprietà delle Metriche di Valutazione delle Regole Associative

Le metriche utilizzate per valutare le regole associative presentano diverse proprietà che influenzano la loro interpretazione e applicabilità. Ecco alcune proprietà chiave:

**1. Simmetria:**

* Alcune metriche sono **simmetriche**, il che significa che il risultato è lo stesso indipendentemente dall'ordine degli elementi nella regola (es. *A → B* è equivalente a *B → A*). Esempi di metriche simmetriche includono Jaccard, coseno e Lift.
* Altre metriche sono **asimmetriche**, il che significa che il risultato dipende dall'ordine degli elementi nella regola. Esempi di metriche asimmetriche includono confidenza e Laplace.
* Le metriche simmetriche sono più adatte per misurare la validità dell'itemset, mentre le metriche asimmetriche sono più adatte per misurare la validità della regola.

**2. Variazione di Scala:**

* Alcune metriche sono **insensibili** alla variazione di scala del dataset. Questo significa che il risultato rimane lo stesso anche se il numero di elementi nel dataset cambia.
* Altre metriche sono **sensibili** alla variazione di scala. Questo significa che il risultato cambia se il numero di elementi nel dataset cambia.
* La scelta della metrica dipende dal contesto e dalla sensibilità desiderata alla variazione di scala.

**3. Correlazione:**

* Il coefficiente *ϕ* è una metrica che misura la correlazione tra due variabili binarie.
* Il coefficiente *ϕ* è **insensibile** all'inversione dei valori delle variabili. Questo significa che il risultato rimane lo stesso anche se si scambiano i valori di *X* e *Y*.
* La sensibilità all'inversione può essere un vantaggio o uno svantaggio a seconda del contesto.

**4. Addizione di Casi Nulli:**

* Alcune metriche sono **sensibili** all'aggiunta di casi nulli (transazioni che non contengono né *X* né *Y*). Questo significa che il risultato cambia se si aggiungono casi nulli al dataset.
* Altre metriche sono **insensibili** all'aggiunta di casi nulli. Questo significa che il risultato rimane lo stesso anche se si aggiungono casi nulli al dataset.
* La scelta della metrica dipende dalla sensibilità desiderata all'aggiunta di casi nulli.

Di seguito una tabella che riassume le proprietà di alcune misure:

![[7) Regole associative-20241004212304481.png|437]]
## Paradosso di Simpson: Un'Illusione di Correlazione

Il paradosso di Simpson dimostra come l'analisi di dati aggregati può portare a conclusioni errate quando non si tiene conto di variabili nascoste.

**Esempio:**
![[7) Regole associative-20241004204237497.png|317]]
Consideriamo l'analisi di due variabili:

* **Buy HDTV:** persone che comprano un televisore HD
* **Buy Exercise Machine:** persone che comprano macchine per esercizi

L'analisi iniziale suggerisce che le persone che comprano un televisore HD hanno una maggiore probabilità di comprare anche una macchina per esercizi. Questo si può vedere calcolando la confidenza della regola *Buy HDTV → Buy Exercise Machine*:

$c((\{HDTV= Yes\} \to \{text{Exercise Machine} = Yes\}) = 99 /180=55\%$

$c((\{HDTV= No\} \to \{text{Exercise Machine} = Yes\}) = 54 /120=45\%$

Queste due confidenze ci dicono che sono di più le persone che, oltre alla macchina per esercizi, hanno comprato anche il televisore.

Tuttavia, quando si introduce una variabile nascosta, come lo stato di studente o lavoratore, la relazione si inverte.

**Analisi Stratificata:**
![[7) Regole associative-20241004204457945.png|454]]
* **Studenti:** La confidenza è più alta per le persone che comprano la macchina per esercizi ma non il televisore HD.
	$c((\{HDTV= Yes\} \to \{text{Exercise Machine} = Yes\}) = 1 /10=10\%$
	$c((\{HDTV= No\} \to \{text{Exercise Machine} = Yes\}) = 4 /34=11.8\%$
* **Lavoratori:** La confidenza è più alta per le persone che comprano la macchina per esercizi ma non il televisore HD.
	$c((\{HDTV= Yes\} \to \{text{Exercise Machine} = Yes\}) = 98 /170=57.7\%$
	$c((\{HDTV= No\} \to \{text{Exercise Machine} = Yes\}) = 50 /86=58.1\%$

**Spiegazione:**
Il paradosso di Simpson si verifica perché la relazione osservata tra le due variabili (Buy HDTV e Buy Exercise Machine) è influenzata dalla variabile nascosta (stato di studente o lavoratore).

**Conclusione:**
È importante considerare le variabili nascoste quando si analizzano i dati per evitare di trarre conclusioni errate. La stratificazione dei dati, ovvero l'analisi separata di sottogruppi, può aiutare a identificare le relazioni reali tra le variabili e a evitare il paradosso di Simpson.

## Effetti della Distribuzione del Supporto
La distribuzione del supporto negli itemset può influenzare significativamente l'analisi delle regole associative.

**Problema:**
La distribuzione del supporto è spesso distorta, con pochi itemset ad alto supporto e molti itemset a basso supporto.

**Esempio:**

![[7) Regole associative-20241004204636451.png|368]]

**Conseguenze:**
* **Supporto minimo elevato:**
    * Perdita di itemset rari che potrebbero essere interessanti per l'analisi.
* **Supporto minimo basso:**
    * Calcolo costoso con un numero eccessivo di itemset.

## Dataset con Supporto non Omogeneo
L'analisi di regole associative in dataset con supporto non omogeneo presenta sfide specifiche.

**Problema:**
La presenza di itemset con supporto molto diverso tra loro può portare a conclusioni errate sull'interesse delle regole associative.

**Esempio:**
![[7) Regole associative-20241009102544967.png]]

* La regola *q → p* ha una confidenza alta (4/5), ma il supporto di *q* è basso (5/30).
* La regola *p → q* ha una confidenza bassa (4/25), ma il supporto di *p* è alto (25/30).

**Conseguenze:**
* Regole con alta confidenza ma basso supporto possono essere poco significative.
* Regole con bassa confidenza ma alto supporto possono essere interessanti, ma non vengono rilevate.

**Cross-Support:**
* Itemset con supporto elevato ma basso supporto congiunto (es. *{p, q, r}, {p, r}, {p, q}*) possono essere considerati **cross-support**.
* Le regole associative relative a questi itemset sono di scarso interesse, nonostante l'elevata confidenza.

**Conclusione:**
In dataset con supporto non omogeneo, la confidenza da sola non è sufficiente per valutare l'interesse delle regole associative. È necessario considerare anche il supporto degli itemset coinvolti.


## Cross-Support
Il cross-support è un problema che si presenta nell'analisi delle regole associative quando gli itemset contengono elementi con supporti molto diversi tra loro.

**Problema:**
* Regole con alta confidenza ma basso supporto possono essere poco significative.
* Esempi tipici: "caviale → latte" (caviale ha basso supporto, latte ha alto supporto).

**Esempio:**

![[7) Regole associative-20241004204653413.png|330]]

**Soluzione:**
* Utilizzare una misura di cross-support per identificare itemset con supporti disomogenei.
* Eliminare gli itemset con cross-support inferiore a una soglia predefinita.

**Misura di Cross-Support:**
Dato un itemset $X = \{x_1, x_2, ..., x_n\}$, la misura di cross-support, r, è definita come:

$$r = \frac{min\{s(x_1), ..., s(x_n)\}}{max\{s(x_1), ..., s(x_n)\}}$$
dove s(xi) è il supporto dell'i-esimo elemento.

**Interpretazione:**
* Se r < γ (γ è una soglia predefinita), allora X contiene elementi con supporto diverso.
* Gli itemset con cross-support inferiore a γ vengono eliminati.

## H-confidence
L'H-confidence è una misura che aiuta a identificare regole associative significative in presenza di itemset con supporti disomogenei.

**Problema:**
* La confidenza di una regola può essere alta anche se gli itemset coinvolti hanno supporti molto diversi.
* Questo può portare a regole poco significative, come "caviale → latte".

**Soluzione:**
* L'H-confidence considera la confidenza minima di tutte le possibili regole ottenibili da un itemset.
* Questo aiuta a identificare itemset con una forte relazione tra tutti gli elementi, indipendentemente dal loro supporto individuale.

**Definizione:**

Dato un itemset $X = \{x_1, x_2, ..., x_n\}$, l'H-confidence è definita come:

$$H = min\{c(X_1 \to X_2) | X_1 \subset X \wedge X_2 = X - X_1\}$$

dove $c(X_1 \to X_2)$ è la confidenza della regola $X_1 \to X_2$, definita come:

$$c(X_1, X_2) = \frac{s(X_1 \cup X_2)}{s(X_1)}$$

**Calcolo:**
* Si considera la confidenza minima di tutte le regole ottenibili da $X$ effettuando un partizionamento degli elementi.
* Si trova la variabile $X_1$ con supporto massimo.
* L'H-confidence è data da:

$hconf = min\left\{ \frac{s(X)}{s(x_1)}, ..., \frac{s(X)}{s(x_n)} \right\} = \frac{s(X)}{max\{s(x_1), ..., s(x_n)\}}$

**Vantaggi:**
1. Un'elevata H-confidence implica una stretta relazione tra tutti gli elementi del pattern.
2. Vengono eliminati i cross-support pattern.
3. Pattern con basso supporto e elevata H-confidence possono essere individuati efficientemente.

## Cross Support e H-confidence

Il supporto di un itemset diminuisce all'aumentare del numero di elementi che lo compongono. Questo perché il supporto di un itemset è limitato dal supporto dei suoi elementi individuali. 

La **h-confidence** è una misura che quantifica la relazione tra il supporto di un itemset e il supporto dei suoi elementi individuali. È definita come il rapporto tra il supporto dell'itemset e il supporto massimo dei suoi elementi:

$$hconf = \frac{s(X)}{max\{s(x_1), ..., s(x_n)\}}$$

dove:

* $s(X)$ è il supporto dell'itemset $X$
* $s(x_i)$ è il supporto dell'elemento $x_i$

La h-confidence è sempre minore o uguale al rapporto tra il supporto minimo e il supporto massimo degli elementi dell'itemset:

$$hconf \le \frac{min\{s(x_1), ..., s(x_n)\}}{max\{s(x_1), ..., s(x_n)\}} = r$$

Un itemset con h-confidence bassa indica che il supporto dell'itemset è significativamente inferiore al supporto dei suoi elementi individuali. Questi itemset sono chiamati **hyperclique**.

Gli hyperclique sono interessanti perché possono rivelare pattern con basso supporto che potrebbero essere trascurati da un'analisi basata solo sul supporto. 

È possibile definire **hyperclique chiusi** e **massimali** in base alla h-confidence:

* Un hyperclique $X$ è **chiuso** se nessuno dei suoi immediati superset ha la stessa h-confidence di $X$.
* Un hyperclique $X$ è **massimale** se nessuno dei suoi immediati superset è un hyperclique.

La h-confidence può essere utilizzata in sostituzione o congiunzione con il supporto per trovare itemset con supporti comparati. 

## Attributi Categorici e Continui

Fino ad ora abbiamo considerato solamente regole associative create a partire da attributi binari. Vogliamo estendere l'utilizzo delle regole associative anche all'uso degli attributi categorici e continui per poter creare regole del tipo:

![[7) Regole associative-20241004204708126.png|516]]
$\{Gender = Male, Age ∈ [21, 30]\} → \{\text{No of hours Online\} ≥ 10}$ 

### Gestione di attributi categorici
Se abbiamo un dataset di questo tipo

![[7) Regole associative-20241004204718638.png|539]]

 Potremmo generare regole di questo tipo:

${\text{Level of education} = Graduate, \text{Online Banking} = Yes} → {\text{Privacy Concern} = Yes}$ 


Per applicare le regole viste fino a questo momento basta trasformare gli attributi categorici in attributi di tipo binario asimmetrici.

 Per ogni attributo categorico andiamo a generare tanti attributi quanti sono i valori del dominio. Nel caso dell'attributo *Gender* verranno creati due nuovi attributi *Male* e *Female*. Stesso procedimento per tutti gli altri attributi.

![[7) Regole associative-20241004204726076.png|502]]

In seguito per creare le regole applichiamo le tecniche già viste. In una trasformazione di questo tipo un problema che possiamo avere è la dimensione della tabella. Tuttavia, poiché sono valori booleani, la dimensionalità non risulta essere un problema in quanto per memorizzare questi valori è sufficiente un bit o un byte. Dal punto di vista dello spazio non abbiamo crescita di complessità. Tuttavia, avendo tante colonne possono essere generate tante regole. Inoltre, alcuni valori degli attributi categorici che andiamo a codificare in questo modo potrebbero avere un basso supporto. La soluzione sta nell'aggregare i valori che hanno un basso supporto.

Un altro aspetto che va considerato è che se un attributo ha dei valori con distribuzione non omogenea allora potremmo creare dei pattern ridondanti. Per esempio:
La regola
${\text{number of pages} \in[5,10)\cap(\text{Browser=Firefox})} \to (Buy=no)$
è sussunta dalla più generale regola
$\text{number of pages} \in[5,10) \to (Buy=no)$


In questo caso è preferibile eliminare l'attributo binario corrispondente al valore con elevato supporto poiché esso non fornisce informazioni utili. Una soluzione alternativa è quella di utilizzare la tecnica già vista per gestire attributi con supporto disomogeneo (hconf).

Possiamo quindi riassumere quanto detto con i seguenti punti:

* Rendere binari i dati aumenta il numero di elementi.
* La larghezza delle transazioni rimane la stessa.
* Vengono prodotti più itemset frequenti la cui dimensione massima è limitata però al numero degli attributi originali.

Un approccio per ridurre questo overhead è evitare che attributi, adesso binari, derivanti dallo stesso attributo categorico siano presenti più di una volta all'interno di una regola.

## Gestione di Attributi Continui

Le regole associative che includono attributi a valori continui sono dette regole associative quantitative:
$$\begin{gather}
\{ Age \in[21,35) \cap Salary \in[70k,120k) \} \to \{ Buy = Yes \} \\
\{ Age \in[21,35),\text{No of hours onlinee} \in[10,20)\} \to \{\text{Chat Online} -Yes\}\\
\{ Age \in[21,35),\text{Chat Online} = Yes\} \to \text{No of hours online}: \mu=14,\sigma=4
\end{gather}
$$


Non abbiamo più attributi di tipo categorico ma abbiamo valori che descrivono una distribuzione. Di seguito riportati i tre approcci per gestire questi dati.

## Discretizzazione

Abbiamo dei dati tabellari con degli attributi continui. In questo caso, per sostituire tali attributi con attributi binari asimmetrici, quello che possiamo fare è di discretizzare il dominio, cioè dividere il dominio in intervalli come mostrato in figura:

![[7) Regole associative-20241004204733700.png|557]]
Sorgono ovviamente dei problemi, uno di questi è come stabilire il partizionamento. Prevalentemente si utilizzano tre tecniche per stabilire la discretizzazione:

* **Equal-width**: divido il dominio in intervalli di uguale dimensione.
* **Equal-depth**: divido in intervalli con un uguale numero di elementi.
* **Clustering**

(Nella slide divide tra approcci supervisionati e non supervisionati ma né lui né il libro ne parlano). Abbiamo due casistiche:

* **Intervalli ampi**: con intervalli ampi si riduce la confidenza dei pattern poiché il raggruppamento non cattura più un fenomeno omogeneo. Pattern diversi potrebbero essere infatti fusi o addirittura potremmo perdere dei pattern interessanti.
* **Intervalli stretti**: con intervalli troppo stretti si riduce il supporto dei pattern poiché al raggruppamento corrispondono un minor numero di transazioni. Un unico pattern potrebbe essere diviso in più pattern distinti e alcune finestre (creando itemset) potrebbero non superare il supporto minimo.

Una soluzione a questi problemi potrebbe essere andare a provare con tutti i possibili intervalli. Si parte da un valore iniziale *k* che rappresenta la larghezza dell'intervallo per poi andare a fondere gli intervalli vicini nel momento in cui *k* cresce. Questa soluzione, come si può immaginare, è computazionalmente costosa e può portare a pattern ridondanti.

## Regole Ridondanti

Se un intervallo \[*a, b*) è frequente allora tutti gli intervalli che lo contengono sono frequenti, per esempio:
$$\begin{gather}
se \ \{ Age \in[21,35),\text{Chat Online} = Yes\} \text{è  frequente} \\
\to\{ Age \in[10,50),\text{Chat Online} = Yes\} \text{è anche frequente}
\end{gather}
$$

Per evitare intervalli troppo ampi si potrebbe pensare di utilizzare come ottimizzazione un valore soglia (supporto massimo) per evitare intervalli troppo ampi. Nel caso in cui avessimo due regole ridondanti che hanno la stessa confidenza come queste:
$$\begin{gather}
R_{1}: \{ Age \in[18,20)\} \to   \{\text{Chat Online} = Yes\} \\
R_{2}:\{ Age \in[18,23)\} \to   \{\text{Chat Online} = Yes\} 
\end{gather}
$$

 andiamo ad eliminare la regola più specifica (intervallo di valori minore).

## Approccio Statistico

Supponiamo di avere questo dataset:

![[7) Regole associative-20241004205040960.png|635]]

Dove abbiamo attributi numerici e categorici. Supponiamo inoltre di aver generato degli item-set frequenti (come quelli riportati in basso). A partire da questi itemset vogliamo generare delle regole come quelle in basso a destra. Il conseguente delle regole consiste in una variabile continua caratterizzata da valori statistici. La regola che stiamo generando, inoltre, non dipende completamente dagli itemset frequenti, solo il corpo (antecedente) della regola viene generato dagli itemset frequenti. Sugli itemset frequenti andiamo poi a calcolare delle statistiche sugli attributi continui contenuti nel conseguente.

## Estrazione di regole di associazione con attributo target

Nell'algoritmo, in questo caso, l'attributo target non viene considerato. Calcoliamo gli item-set frequenti senza considerare l'attributo target. Una volta generato l'itemset frequente andiamo a generare una regola dove nel corpo abbiamo tutto l'itemset frequente e nel conseguente ci sta l'attributo target del quale consideriamo alcune misure statistiche. I passi sono riportati di seguito:

1. **Elimina l'attributo target dai dati.**
2. **Estrai gli itemset frequenti dal resto degli attributi.**
3. **"Binarizza" gli attributi continui (eccetto l'attributo target).**
4. **Per ogni itemset frequente, calcola le corrispondenti statistiche descrittive dell'attributo target.**
5. **Gli itemset frequenti diventano regole introducendo la variabile target come conseguente.**
6. **Applica il test statistico per determinare quanto è interessante la regola (validazione delle regole).**

**Nota:** La binarizzazione degli attributi continui è un passaggio importante per poter applicare l'algoritmo di estrazione di regole di associazione. Questo passaggio consiste nel trasformare gli attributi continui in attributi binari, ovvero attributi che possono assumere solo due valori (ad esempio, 0 o 1). 

### Validità delle Regole

Ci chiediamo ora quando una regola è significativa. In questo caso non è possibile utilizzare la confidenza in quanto nel conseguente abbiamo dei valori reali. Una regola associativa quantitativa è interessante solo se le statistiche calcolate sulle transazioni coperte dalla regola sono significativamente diverse da quelle calcolate sulle transazioni non coperte dalla regola.
$A \to B: \mu \text{   verso } \bar{A}\to B: \mu$


(Verso sarebbe vs). A è l’insieme di tutti gli oggetti che non stanno nell’itemset frequente. L’obiettivo quindi è verificare quando la differenza tra μ e μ′ è maggiore di una soglia ∆. Se i due valori μ e μ′ sono molto vicini, la regola non è significativa.

Un modo per calcolare la distanza tra i due valori è effettuare lo Z-test, ovvero, date le regole di cui sopra, calcolare il seguente valore:


$$Z = \frac{μ' - μ - ∆}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$


Se \(μ > μ'\), vanno invertiti nella formula. Al denominatore abbiamo la varianza del gruppo catturato dalla prima regola (\(s_1^2\)) rispetto al numero di oggetti catturati dalla prima regola (\(n_1\)), sommata alla varianza della seconda regola diviso il numero di oggetti calcolati dalla seconda regola.

Se questo valore \(Z\) è superiore ad una soglia fissata (\(Z_α\), che è un valore calcolato in base ad una determinata confidenza, come nei test statistici), allora la regola è significativa.



## 16.2.3 Senza Discretizzazione (minApriori)

Queste tecniche per generare regole associative vengono utilizzate su molti tipi di dati, per esempio dati a grafo, dati sequenziali oppure su documenti. Diamo un cenno di quello che succede quando abbiamo dei documenti. In questo caso vogliamo generare delle regole che ci dicono che se un documento contiene con una certa rilevanza una parola o un gruppo di parole allora quel documento contiene anche con una certa frequenza un termine di interesse. Supponendo di avere questi documenti:

![[7) Regole associative-20241004205158405.png|287]]

Sulle righe abbiamo i documenti, sulle colonne le parole, nelle celle abbiamo le occorrenze delle parole nei documenti. Abbiamo quindi dei valori numerici. Non possiamo in questo caso trasformare i dati in valori binari, o meglio, potremmo trasformarli in attributi binari sostituendo i valori diversi da 0 con 1. Questa sostituzione però comporta la perdita di informazione riguardante le occorrenze delle parole. Quello che possiamo fare è normalizzare il contenuto della tabella facendo in modo che il supporto per ogni parola sia pari a 1, oppure, facendo in modo che la somma dei pesi delle parole all'interno dei diversi documenti (lungo le colonne quindi) sia 1.

![[7) Regole associative-20241004205207044.png|536]]

 Vogliamo ora calcolare dei gruppi di parole (itemset) per andarne a calcolare il supporto che è definito come segue: $sup(C) = \sum_{i \in T} \min_{j \in C} D(i, j)$ 


dove:

* *C* è l'insieme di parole.
* *T* rappresenta le transazioni, in questo caso documenti, quindi per ogni documento effettuiamo quella somma, ovvero sommiamo il valore più piccolo che riguarda il gruppo di parole.
* *D*(*i, j*) dà il valore presente nella cella della tabella.

Esempio:

![[7) Regole associative-20241004205212296.png|524]]

Una volta calcolato il supporto è possibile calcolare le regole. Anche in questo caso il supporto

è antimonotono ovvero aggiungendo elementi all'insieme questo tende a diminuire:
Sup(W1) = 0.4 +0 + 0.4 +0 + 0.2 = 1
Sup(W1, W2) = 0.33 + O + 0.4 + O + 0.17 = 0.9
Sup(W1, W2, W3) =0+0+0+0 + 0.17 = 0.17


## Regole Associative Multi-Livello

Un aspetto che va considerato è il fatto che quando abbiamo dei termini questi sono organizzati in gruppi basati sulla semantica dei propri elementi. Per esempio il cibo può essere pane o latte, a sua volta il latte può essere scremato o parzialmente scremato e così via.

![[7) Regole associative-20241004205303179.png|486]]

Quando generiamo delle regole queste possono essere create a diversi livelli di (tassonomia? non si capisce) "astrazione". Le gerarchie di concetti vengono incorporate perché le regole ai livelli più bassi di una gerarchia potrebbero non avere supporto sufficiente per apparire in itemset frequenti. Le regole a livello troppo basso potrebbero essere troppo specifiche. Per esempio:

Latte scremato-> pane bianco
Latte parzialmente scremato-> pane bianco,
Latte scremato -> pane integrale

Sono tutte indicative di un'associazione più generale tra latte e pane. Le ultime te slide non le ha spiegate ma le riporto come immagine per completezza
**Supporto e confidenza variano da livello a livello:**
```
If X è genitore di X1 e X2, then 
	σ(X) ≤ σ(X1) + σ(X2)
    
If σ(X1 ∪ Y1) ≥ minsup, 
and X è genitore di X1, 
Y è genitore di Y1 
then σ(X ∪ Y1) ≥ minsup, 
σ(X1 ∪ Y) ≥ minsup 
σ(X ∪ Y) ≥ minsup
    
If conf(X1 ⇒ Y1) ≥ minconf, 
then conf(X1 ⇒ Y) ≥ minconf
```


## Conseguenze

* **Item ai livelli "alti" della gerarchia avranno supporti molto elevati.**
* **Possibilità di pattern cross-support.** Se la soglia per il supporto è bassa saranno identificati molti pattern che coinvolgono item ai livelli alti della gerarchia.
* **Il numero di associazioni ridondanti aumenta.**
    *  Latte -> Pane
    *  Latte scremato -> Pane
* **La dimensionalità dei dati aumenta e conseguentemente aumenta il tempo di elaborazione.**

**Una soluzione alternativa è quella di generare i pattern frequenti separatamente per i diversi livelli della gerarchia.**

* **Generare inizialmente i pattern al livello più alto della gerarchia.**
* **Procedere iterativamente verso i livelli successivi.**

**Conseguenze:**

* **Il costo di I/O crescerà notevolmente poiché saranno necessarie più scansioni dei dati.**
* **Saranno persi eventuali associazioni cross-livello interessanti.** 

