
##### Stochastic Topic Modeling

* **Processo Generativo:** Non prevede testo parola per parola, ma caratterizza un documento come composizione di distribuzioni di probabilità.
* Ogni distribuzione rappresenta una distribuzione di probabilità sullo spazio delle parole.
* Doppio campionamento:
* Campionamento del topic da una distribuzione di probabilità sui topic.
* Campionamento delle parole da una distribuzione di probabilità sulle parole, specifica per il topic.
* Campionamento dei topic per il documento da una distribuzione di probabilità sui topic.

##### Elaborazione del Linguaggio Naturale (NLP)

* **Obiettivo:** Sviluppare sistemi AI capaci di comprendere e generare linguaggio naturale, simile all'apprendimento infantile.
* **Problema della Rappresentazione:** Rappresentare il linguaggio in modo che possa essere elaborato e generato robustamente.
* **Rappresentazione delle Parole:** Associare unità lessicali a concetti.
* **Semantica Denotativa:** Collegare un significante (parola) a un significato (idea/cosa). Esempio: "albero" → 🌲, 🌳, 🪴.

##### Limiti di WordNet

* **Mancanza di sfumature e contesto:** Non cattura tutte le sfumature di significato e il contesto d'uso delle parole. Esempio: "proficient" ≈ "good" solo in alcuni contesti.
* **Informazione quantitativa:** Non fornisce informazioni quantitative sull'appropriatezza di una parola in un contesto.
* **Mantenimento:** Richiede aggiornamenti continui e costosi a causa dell'evoluzione del linguaggio.
* **Soggettività:** Soggetto a bias culturali (es. prospettiva britannica).
* **Similarità:** Non permette il calcolo accurato della similarità tra parole.

##### Rappresentazione Semantica delle Parole

* **Embedding:** Rappresentazione delle parole in uno spazio multidimensionale per calcolare la similarità semantica.
* **Principio chiave:** Parole sinonime sono rappresentate da punti vicini nello spazio.
* **Metodo comune:** Analisi delle co-occorrenze di parole in un corpus. Maggiore frequenza di co-occorrenza implica maggiore similarità di significato.

### Rappresentazione delle Parole in NLP

##### Rappresentazioni Tradizionali:

* **Simboli Discreti:**
	* Tipo di parola: entry nel vocabolario.
	* Token di parola: occorrenza nel contesto.
* **Vettori One-hot:**
	* Rappresentazione localista: una parola = un vettore.
	* Solo una posizione a "1", le altre a "0".
	* Esempio: *motel* = $[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]$
	* Dimensione = |vocabolario|.
* **Svantaggi:**
	* Mancanza di similarità semantica.
	* Forte dipendenza dal vocabolario.

##### Word Embedding:

* **Evoluzione da LSA:** Trasformazione lineare che cattura relazioni semantiche e riduce la dimensionalità.
* **Proprietà di Prossimità:** Parole simili = vettori vicini nello spazio vettoriale.

##### Semantica Distribuzionale:

* **Principio Fondamentale:** Il significato di una parola è determinato dalle parole che la circondano.
* **Pattern di Co-occorrenza:** Analisi delle parole che appaiono frequentemente vicine.
* **Contesto:** Insieme delle parole vicine a *w* entro una finestra di dimensione fissa.

##### Matrice di Co-occorrenza a Livello di Documento:

* **Costruzione:**
	* Definire un vocabolario *V*.
	* Creare una matrice |V| × |V| (inizialmente a zero).
	* Contare le co-occorrenze di parole all'interno di ogni documento.
	* Normalizzare le righe (dividere per la somma dei valori della riga).
	* **Risultato:** Rappresentazione sparsa, ma ancora costosa computazionalmente.
	* **Problemi:** (immagine allegata nel testo originale, non riproducibile qui).

##### Co-occorrenza a livello di documento: Problematiche e Soluzioni

* **Dimensioni della finestra di co-occorrenza:**
	* **Finestre grandi:** Cattura informazioni semantiche/di argomento, ma perde dettagli sintattici.
	* **Finestre piccole:** Cattura informazioni sintattiche, ma perde dettagli semantici/di argomento.
	* **Conteggi grezzi:** Sovrastimano l'importanza delle parole frequenti.
* **Miglioramento dei conteggi:**
	* **Logaritmo della frequenza:** Mitiga l'effetto delle parole comuni.
	* **GloVe:** Apprende rappresentazioni di parole basate su co-occorrenze in un corpus (approccio più avanzato).

##### Vettori di Parole (Word Embeddings)

* **Obiettivo:** Creare vettori densi per ogni parola, simili per parole in contesti simili.
* **Similarità:** Misurata tramite prodotto scalare dei vettori.
* **Rappresentazione:** Distribuita (spesso neurale).

##### Word2vec: Apprendimento di Vettori di Parole

* **Idea:** Apprendere vettori di parole da un grande corpus di testo, massimizzando la probabilità di co-occorrenza tra parole centrali e di contesto.
* **Processo:**
	* Ogni parola ha un vettore.
	* Si itera su ogni posizione nel testo (parola centrale e parole di contesto).
	* Si usa la similarità dei vettori per calcolare la probabilità di una parola di contesto data la parola centrale (o viceversa).
	* Si aggiornano i vettori per massimizzare questa probabilità.
	* **Funzione obiettivo:** Minimizzare la log-verosimiglianza:
$$ \text{Likelihood}=L_{0}=\prod_{t=1}^T \prod_{-m\leq j\leq m}P(w_{t+j|w_{t};\theta}) $$
dove $P(w_{t+j|w_{t};\theta})$ è la probabilità della parola di contesto shiftata di $j$ rispetto a $t$, data $w_{t}$.
* **Minimizzazione:** Minimizzare la seguente funzione obiettivo:
$$J(θ) = -\frac{1}{T} \sum_{t=1}^T\sum_{j≠0}\log P(W_{t+j} | W_{t}; θ)$$
Questo implica massimizzare l'accuratezza di predizione delle parole di contesto.

##### Calcolo della Probabilità Condizionata

* **Utilizzo di due vettori per parola:**
	* `Vw`: vettore per parola centrale (target word).
	* `Uw`: vettore per parola di contesto (context word).
* **Formula per la probabilità condizionata:**
	* $P(o|c) = \frac{\exp(u_o^T v_c)}{Σ_{w∈V} \exp(u_w^T v_c)}$ (softmax)
	* **Distinzione tra vettori:** Necessaria per gestire parole che appaiono in entrambi i ruoli (target e contesto).

##### Word2vec: Apprendimento delle Rappresentazioni delle Parole

* **Metodo di addestramento:**
	* Selezione iterativa di una parola centrale e della sua finestra di contesto (dimensione *m*) da un documento di lunghezza *T*.
	* Creazione di esempi di addestramento.
* **Rete neurale:**
	* Un singolo strato nascosto di dimensione *N* (*N* << *V*, dove *V* è la dimensione del vocabolario).
	* Strati di input/output: vettori one-hot di dimensione *V*.
	* **Compito di apprendimento:** Predire la probabilità di una parola di contesto data una parola centrale.
	* **Ottimizzazione:** Discesa del gradiente stocastica (SGD).
* **Funzione di costo:**
	* $j(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m}\log P(w_{t+j}|w_{t};\theta)$
	* dove $P(w_{t+j}|w_{t};\theta) = P(o|c)$ (probabilità calcolata tramite softmax).
* **Calcolo del gradiente (esempio per Vc):**
	* $\frac{\delta J}{\delta V_{c}} = u_{o}-\sum_{x\in V}p(x|c)v_{x}$

##### Derivazione delle Regole di Aggiornamento per la Regressione Logistica

* **Funzione di perdita:** Cross-entropy
* $L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$
* *y*: valore reale (0 o 1)
* $\hat{y}$: probabilità prevista (tra 0 e 1)
* **Calcolo del gradiente:**
* **Rispetto a *z*:**
	* $\frac{\partial L}{\partial z} = \hat{y} - y$
* **Rispetto a *w*:**
	* $\frac{\partial L}{\partial w} = (\hat{y}-y)x$ (dove *x* è il valore dell'input)

##### Calcolo del Gradiente e Aggiornamento dei Parametri

* **Gradiente rispetto a *b***: $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = \hat{y} - y$
* **Regole di Aggiornamento (Discesa del Gradiente):**
	* $w^{(k+1)} = w^{(k)} - \lambda \frac{\partial L}{\partial w^{(k)}}$
	* $b^{(k+1)} = b^{(k)} - \lambda \frac{\partial L}{\partial b^{(k)}}$
	* Sostituzione dei gradienti:
	* $w^{(k+1)} = w^{(k)} - \lambda (\hat{y} - y)x$
	* $b^{(k+1)} = b^{(k)} - \lambda (\hat{y} - y)$
	* **Addestramento del Modello:** Ottimizzazione dei parametri ($O$) per minimizzare la funzione di perdita. Nel caso di Word2Vec, $O$ include i vettori $w_i$ (parole) e $b_i$ (contesto) per ogni parola. (Figura omessa, ma presente nel testo originale).

##### Word2Vec: Algoritmi e Implementazione

* **Famiglia di Algoritmi:** Skip-gram e Continuous Bag-of-Words (CBOW). Due vettori per parola ($w_i$, $b_i$) per semplificare l'ottimizzazione (ma possibile con uno solo).
* **Skip-gram (SG):**
	* Predizione delle parole di contesto (window size = n) data una parola target.
	* Associazione uno-a-molti (una parola centrale, M parole di contesto).
	* Spesso si usa la media dei vettori di contesto.
* **Continuous Bag-of-Words (CBOW):**
	* Predizione della parola target date le parole di contesto.
	* Associazione uno-a-uno (M parole di contesto, una parola centrale).
* **Implementazione (generale):**
	* **Skip-gram:** Input one-hot (parola target), hidden layer (dimensione D), matrice di embedding (input-hidden), matrice di contesto (hidden-output). Calcolo probabilità per ogni parola di contesto.
	* **CBOW:** Input one-hot (ogni parola di contesto), hidden layer, output layer. Media dei vettori di contesto, dimensione hidden layer = N o D. Codifica delle parole di contesto in uno spazio N o D dimensionale.

##### CBOW (Continuous Bag-of-Words)

**Processo di apprendimento:** Determina i pesi tra input (parole di contesto) e output (parola centrale) tramite uno strato nascosto.
- One-hot encoding per le parole.
- Media dei vettori delle parole di contesto.
- Decodifica del vettore medio con la matrice dei pesi tra hidden layer e output layer (collegato al softmax).
##### Rappresentazione:

- Dimensione della rappresentazione: dimensione del vocabolario.
- Matrice dei pesi **w** (input-hidden layer): dimensione $d \cdot n$ (d = dimensione vocabolario, n = dimensionalità hidden layer).
- Codifica delle parole di contesto: moltiplicazione del vettore one-hot per **w**.
##### Codifica del contesto:

- Media delle codifiche delle parole di contesto ($\hat{v}$).
- Decodifica di $\hat{v}$ con la matrice dei pesi (hidden-output layer) e input al softmax.
##### Estrazione di Embeddings:

- Da matrice **W**: rappresentazione delle parole di contesto come parole target in Skip-gram.
- Da matrice **W'**: rappresentazione delle parole di contesto come parole di contesto in CBOW.

##### Skip-gram

**Processo di apprendimento:** Predizione delle parole di contesto data una parola centrale.
##### Estrazione di Embeddings:

- Input: singola parola.
- Embedding estratto da matrice **W**: rappresentazione della parola target.

##### Estrazione di Embeddings (generale)

Rappresentazione di una parola come funzione del suo ruolo nella predizione (estraendo embeddings dalla matrice di destra).

##### Softmax e Funzione di Costo in Word2Vec

Probabilità di una parola data un'altra tramite softmax.
Calcolo computazionalmente costoso con vocabolari grandi.

##### Skip-gram con Negative Sampling

**Problema:** Normalizzazione computazionalmente costosa nella softmax.
**Soluzione:** Negative Sampling.
##### Funzione di costo modificata:

$$J_{t}(\theta)=\log\sigma(u_{o}^Tv_{c})+\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$$
- $\log\sigma(u_{o}^Tv_{c})$: logaritmo della sigmoide del prodotto scalare tra vettore parola centrale ($v_c$) e vettore parola output/contesto osservata ($u_o$).
- $\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$: approssimazione del denominatore della softmax.

### Word Embedding: Tecniche di Addestramento

##### Negative Sampling

* **Obiettivo:** Apprendimento discriminativo per avvicinare le parole di contesto alla parola centrale e allontanare le parole negative (rumore).
* Massimizzare la similarità tra parola centrale e parole di contesto.
* Massimizzare la distanza tra parola centrale e parole negative.
* **Campionamento delle parole negative:** Da una distribuzione unigramma elevata a potenza α (≈ 0.75). Questo riduce l'enfasi sulle parole frequenti.
* Distribuzione unigramma: frequenza di occorrenza normalizzata.
* α < 1: riduce il peso delle parole molto frequenti.

##### Softmax Gerarchica

* **Obiettivo:** Migliorare l'efficienza dell'addestramento rispetto alla softmax standard.
* **Meccanismo:** Utilizza un albero di Huffman per organizzare le parole.
* Foglie: parole del vocabolario.
* Parole frequenti più vicine alla radice.
* Calcolo probabilità: percorso dalla radice alla foglia.
* Aggiornamento pesi: durante l'addestramento per migliorare la precisione.
* **Vantaggi:**
	* Maggiore efficienza computazionale.
	* Struttura gerarchica per una migliore comprensione delle relazioni tra le parole.

##### Word2vec: Skip-gram vs. CBOW

* **Skip-gram:**
	* Obiettivo: predire le parole di contesto data la parola centrale.
	* Vantaggi: migliore per parole rare e finestre di contesto ampie.
	* Svantaggi: più lento in addestramento, meno efficiente per task *document-oriented*.
* **CBOW:**
	* Obiettivo: predire la parola centrale date le parole di contesto.
	* Vantaggi: migliore per parole frequenti, più veloce in addestramento, adatto a task *document-oriented*.
	* Svantaggi: meno preciso per parole rare.

| Caratteristica | Skip-gram | CBOW |
|-----------------------|------------|-----------|
| **Obiettivo** | Predire contesto | Predire centro |
| **Parole frequenti** | Meno preciso | Più preciso |
| **Parole poco frequenti** | Più preciso | Meno preciso |
| **Finestra di contesto** | Ampia | Piccola |
| **Velocità** | Lenta | Veloce |
| **Task** | Similarità, analogia | Classificazione, task document-oriented |

* **Skip-gram preferibile per:** vocabolari ampi, corpus specialistici, parole rare, finestre di contesto ampie e task *word-oriented*.

##### CBOW (Continuous Bag-of-Words)

Vantaggi:
- Addestramento più veloce.
- Adatto a vocabolari moderati e corpus generici.
- Ideale per task *document-oriented*.

Tecniche di Addestramento:
- Hierarchical Softmax: Ottimale per parole rare.
- Negative Sampling: Ottimale per parole frequenti e vettori a bassa dimensionalità.

Ottimizzazione:
- Sottocampionamento: Migliora accuratezza e velocità con dataset grandi (1e-3 a 1e-5).
- Dimensionalità Vettori: Generalmente, maggiore è meglio (ma non sempre).
- Dimensione Contesto: ~5

Aggiornamento Gradiente con Negative Sampling:
- Aggiornamento iterativo per ogni finestra (`window`).
- Sparsità del gradiente: al massimo $2m + 1$ parole (più $2km$ parole per il negative sampling). $\nabla_{\theta} J_t(\theta) = \begin{bmatrix} 0 \\ \vdots \\ \nabla_{\theta_{target\_word}} \\ \vdots \\ \nabla_{\theta_{context\_word}} \\ \vdots \\ 0 \end{bmatrix} \in \mathbb{R}^{2dV}$
- Aggiornamento selettivo dei vettori (solo quelli nella finestra).
- Soluzioni per ottimizzare l'aggiornamento: operazioni sparse, hash.
- Evitare aggiornamenti ingombranti (importante per calcolo distribuito).

Matrice di Co-occorrenza:
- Finestra (window): Cattura informazioni sintattiche e semantiche (*spazio delle parole*).
- Documento completo: Permette di ottenere temi generali (*spazio dei documenti*).

##### Limitazioni delle rappresentazioni basate su co-occorrenze

Rappresentazione sparsa: Difficoltà nell'utilizzo di tecniche come LSA.
Relazioni lineari: LSA cattura principalmente sinonimia, non polisemia. Inefficiente per task *word-oriented* complessi.

##### GloVe (Global Vectors)

Approccio ibrido (neurale e statistico).
Considera il contesto delle parole.
Funzionamento: Analisi delle co-occorrenze per costruire una matrice di co-occorrenze.
Vantaggi:
- Gestione della polisemia.
- Migliore rappresentazione di parole poco frequenti.

### GloVe: Word Embeddings

##### Efficienza Computazionale e Natura del Modello:

* GloVe è un modello neurale efficiente per la creazione di word embeddings.
* Utilizza le co-occorrenze di parole per apprendere rappresentazioni dense.
* Non genera un singolo embedding globale per parola; l'embedding dipende dal corpus di addestramento.
* Ogni parola è rappresentata da un unico embedding all'interno di un dato corpus.

##### Rappresentazioni Globali e Context-Free:

* **Rappresentazioni Globali:** Ogni parola ha un solo vettore, indipendentemente dal contesto. Questa rappresentazione è statica.
* **Rappresentazioni Context-Free:** La rappresentazione di una parola ignora il contesto d'uso.
* **Limiti:**
	* **Polisemia:** Non cattura i diversi significati di una parola a seconda del contesto.
	* **Parole poco frequenti:** Rappresentazioni meno accurate per mancanza di dati di addestramento.

##### Funzione Obiettivo di GloVe:

* Minimizza la differenza tra il prodotto scalare degli embedding di due parole e il logaritmo della loro probabilità di co-occorrenza.
* Funzione obiettivo:
$$ f(x)= \begin{cases} \left( \frac{x}{x_{max}} \right)^\alpha, & if \ x<x_{max} \\ 1, & otherwise \end{cases} $$
$$ \text{Loss } J=\sum_{i,j=1}^Vf(X_{ij})(w_{i}T \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^2 $$
Dove:
* $w_i$, $w_j$: embedding delle parole $i$ e $j$.
* $b_i$, $b_j$: bias per le parole $i$ e $j$.
* $X_{ij}$: numero di co-occorrenze delle parole $i$ e $j$ in una finestra di contesto.
* $f(x)$ include una distribuzione unigram con esponente $\frac{3}{4}$ per smorzare l'effetto delle parole frequenti.
* $\log X_{ij}$ approssima $P(i|j)= \frac{X_{ij}}{X_{i}}$.

* L'errore è la differenza tra co-occorrenza reale e attesa.
* L'obiettivo è catturare le proprietà dei rapporti di co-occorrenza.

##### Interpretazione dei Rapporti di Probabilità:

* Considerando tre parole (i, j, k), dove k è una parola di confronto:
* **Rapporto $P(k|i)/P(k|j) > 1$**: k è più correlata a i che a j.
* **Rapporto $P(k|i)/P(k|j) < 1$**: k è più correlata a j che a i.
* Esempio: Tabella con probabilità e rapporti per parole come "ice", "steam" e diverse parole k ("solid", "gas", "water", "fashion").

### Relazioni Geometriche tra Parole

##### Rappresentazione Vettoriale:

* Obiettivo: rappresentare le parole in uno spazio vettoriale per visualizzare le relazioni geometricamente.
* Esempio: "man" : "woman" :: "king" : "queen" (analogia rappresentata geometricamente).
* Embeddings: permettono di definire queste relazioni come combinazioni lineari.

**Funzione di Confronto *F***:
* Scopo: catturare le relazioni tra parole $w_i$, $w_j$, $w_k$.
* Proprietà:
* **Condizione 1 (Combinazione Lineare):** $F((w_{i}-w_{j})^Tw_{k})=\frac{P(k|i)}{P(k|j)}$
* **Condizione 2 (Simmetria):** $F((w_{i}-w_{j})^Tw_{k})=\frac{F(w_{i}^Tw_{k})}{F(w_{j}^Tw_{k})}$
* **Definizione di *F*:** $F(w_{i}^Tw_{k})=e^{w_{i}^Tw_{k}}=P(k|i) = \frac{x_{ik}}{x_{i}}$ (dove $x_{ik}$ è la co-occorrenza di $w_i$ e $w_k$, e $x_i$ è il conteggio di $w_i$)

##### Derivazione e Interpretazione:

* **Equazione derivata:** $w_{i}^Tw_{k}=\log x_{ik}-\log x_{i}$
* **Semplificazione con bias:** $w_{i}^Tw_{k}+b_{i}+b_{j}=\log X_{jk}$ ( $b_i$ e $b_j$ sono bias, inizializzati a 0; $X_{jk}$ rappresenta le co-occorrenze)
* **Interpretazione:** L'equazione finale confronta le co-occorrenze attese delle parole con le co-occorrenze effettive, considerando la simmetria tramite $b_j$.

