## Librerie di riferimento pre-LLM

* **NLTK (Natural Language Toolkit):** Una libreria Python completa per l'elaborazione del linguaggio naturale, che offre una vasta gamma di strumenti per l'analisi del testo, tra cui tokenizzazione, stemming, lemmatizzazione, analisi morfologica, analisi sintattica e classificazione del testo.

* **SpaCy:** Una libreria Python per l'elaborazione del linguaggio naturale, progettata per la velocità e l'efficienza. Offre funzionalità avanzate per l'analisi del testo, come il riconoscimento di entità denominate (NER), l'estrazione di entità, l'analisi del sentimento e la classificazione del testo. È utilizzata anche a livello commerciale per queste attività.

* **Gensim:** Una libreria Python per l'analisi di argomenti e la modellazione di argomenti. Fornisce implementazioni di modelli di argomenti stocastici, come LDA (*Latent Dirichlet Allocation*), che possono essere utilizzati per scoprire argomenti latenti in grandi set di dati di testo.

## Latent Dirichlet Allocation 

Latent Dirichlet Allocation (LDA) è un modello di topic modeling di riferimento. Esistono diverse tecniche matematicamente eleganti che si basano su LDA, per coprire oltre ai tre aspetti di base anche aspetti riguardo il ruolo di autori o categorie speciali per i documenti.

LDA può essere utilizzato per modellare, oltre ai topic, anche un profilo di autore in maniera supervisionata o semi supervisionata (ad esempio, identificare autori fake o meno, esistenti o meno). Queste estensioni complicano il modello da un punto di vista computazionale, ma hanno avuto un discreto successo (in un'era precedente ai Large Language Models).

Si sfrutta una proprietà importante: **coniugate prior tra le distribuzioni** (tra Dirichlet e polinomiale).

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/5)-20241031095017801.png]]
$\alpha$ e $\eta$ sono parametri di dispersione che influenzano la distribuzione di probabilità sui topic e sulle parole. $\alpha$ controlla la dispersione della distribuzione di probabilità sui topic, mentre $\eta$ controlla la dispersione della distribuzione di probabilità sulle parole.

Ciò che importa è mantenere la differenza in ordine di grandezza, con l'obiettivo di essere più inclusivi nel modellare la distribuzione di probabilità per topic (grana fine nel modellare ogni topic), e meno dispersivi per modellare la distribuzione di probabilità sui topic per ogni documento (assumiamo che ogni documento contenga dati caratteristici di pochi topic). 

### Pseudo

$$\text{For each topic, generate a Dirichlet distribution over terms:}$$
$$\beta_k \sim \text{Dir}_M(\eta), k \in \{1, \ldots, K\}$$

$$\text{For each document } d_i, i \in \{1, \ldots, N\}$$
- $\text{Generate a Dirichlet distribution over topics: } \theta_i \sim \text{Dir}_K(\alpha)$
- $\text{For each word position } j \text{ in document } d_i:$
 - $\text{Choose a topic } z_{ij} \text{ from the distribution in step a., i.e., } z_{ij} \sim \text{Multi}(\theta_i)$
 - $\text{Choose word } w_{ij} \text{ from topic } z_{ij} \text{, i.e., } w_{ij} \sim \text{Multi}(\beta_{z_{ij}})$

# Modellazione di documenti segmentati per argomento

## Modello Generativo Basato su Segmenti [Ponti, Tagarelli, Karypis, 2011]

Questo modello rappresenta un approccio più granulare alla modellazione argomento-documento, basato sulla segmentazione del testo in unità più piccole. 

##### Traduzione in un modello probabilistico congiunto per dati triadici:

Il modello si basa su un'idea chiave: la traduzione della relazione tra argomenti, documenti e segmenti in un modello probabilistico congiunto. Questo modello considera la probabilità di un documento, dato un argomento e un segmento, e viceversa.

##### Motivazione:

Per documenti lunghi, assegnare un solo argomento a un intero documento può essere riduttivo. La struttura semantica complessa di un documento lungo potrebbe richiedere una rappresentazione più fine, che tenga conto della presenza di diversi segnali informativi in diverse parti del documento.

##### Approcci alternativi:

* **Soft clustering:** Invece di assegnare un documento a un solo cluster (hard clustering), il soft clustering permette a un documento di appartenere a più cluster con una certa probabilità.
	* Tuttavia, se i documenti hanno una struttura logica, esplicita o implicita, con una semantica complessa, ci saranno segnali informativi caratterizzanti nell'introduzione e nella conclusione. Questo può portare al **topic drift**, ovvero alla deriva tematica.
* **Segmentazione:** La segmentazione del documento in unità più piccole (segmenti) permette di catturare la struttura semantica fine del documento e di associare diversi argomenti a diversi segmenti.

##### Overclustering:

Per catturare la struttura dei micro-topic, si può utilizzare l'overclustering, ovvero stimare un numero di cluster (k) molto più alto del numero effettivo di argomenti. Questo approccio è stato utilizzato in precedenza con modelli vector space e bag of words, dove la segmentazione veniva applicata prima del clustering dei segmenti e poi dei documenti.

##### Integrazione della segmentazione nei topic model:

L'obiettivo di questo lavoro è di integrare la segmentazione del testo nei topic model, creando un modello a grana più fine che tenga conto della variabile dei segmenti. 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/5)-20241031100023606.png|394]]

Questa immagine rappresenta una rete probabilistica, spesso usata nei modelli di generazione del linguaggio, in cui:
- **D** rappresenta i documenti,
- **Z** rappresenta i topic,
- **S** rappresenta i segmenti del documento (o potenzialmente frasi/sottosezioni),
- **V** rappresenta il vocabolario (le parole).

Il modello assume che, dato un documento, si scelga un topic e, condizionatamente a questo e al segmento, si campiona una parola. Questo processo di campionamento riflette la relazione tra i topic, i segmenti e le parole all'interno del testo.

Il joint model diventa triadico, con le variabili latenti che rimangono i topic. È più un'estensione di PLSA, evitando la complessità di RDA. Non rappresenta in maniera esplicita la relazione tra argomenti, documenti e segmenti, ma migliora la modellazione dei topic con una grana più fine. 

I processi rappresentati dalle frecce indicano che:

1. **Dato un documento \(D\), scegliamo un topic \(Z\) secondo la probabilità condizionata $\text{Pr}(z|d)$.**
2. **Dato il topic \(Z\), selezioniamo un segmento \(S\) con probabilità $\text{Pr}(s|z)$ .**
3. **Dato il topic \(Z\), possiamo anche scegliere una parola \(W\) con probabilità $\text{Pr}(w|z)$.**
4. **Dato un segmento \(S\), scegliamo una parola \(W\) con probabilità $\text{Pr}(w|s)$ .**

Il campionamento in questo contesto viene effettuato iterativamente o ripetutamente per generare contenuti a partire dai documenti, utilizzando la struttura probabilistica del modello. 

## Segmentazione del Testo

La segmentazione del testo è un processo fondamentale per la comprensione e l'analisi di documenti. L'obiettivo è suddividere il testo in unità significative, chiamate segmenti, che rappresentano concetti o temi distinti.

Esistono diversi approcci alla segmentazione del testo:

* **Soluzione Naive:** Segmentare il testo in base ai paragrafi. Questo metodo è semplice ma spesso non è accurato, poiché un paragrafo può contenere più temi.
* **Soluzione Intelligente:** In assenza di markup, è necessario utilizzare tecniche di segmentazione non supervisionata. Un esempio è il "TextTiling", che utilizza un modello vector space con tf-idf per calcolare la similarità tra segmenti consecutivi.

### Text Tiling

Il Text Tiling è un metodo di segmentazione non supervisionato che utilizza un modello vettoriale per rappresentare il testo. I passaggi principali sono:

1. **Rappresentazione vettoriale:** Il testo viene rappresentato come un vettore nello spazio vettoriale.
2. **Calcolo della similarità:** La similarità tra segmenti consecutivi viene calcolata utilizzando la similarità coseno.
3. **Curva delle similarità:** Viene tracciata una curva che rappresenta la similarità tra i segmenti.
4. **Identificazione dei punti di discontinuità:** I minimi locali della curva indicano un cambiamento di topic.
5. **Segmentazione:** Il testo viene segmentato in corrispondenza dei minimi locali.

La curva delle similarità mostra massimi locali che rappresentano una continuità di topic. I minimi locali, invece, indicano un cambiamento di topic. L'approccio del Text Tiling prevede di spezzare il testo in corrispondenza dei minimi locali, ottenendo così segmenti che rappresentano temi distinti.

### Modello di argomenti latenti (LDA) per il Text Tiling

Il modello LDA può essere utilizzato per il Text Tiling, considerando la probabilità di un documento, un segmento e una parola come segue:

$$ \operatorname{Pr}(d , s, w) = \operatorname{Pr}(d) \sum_{z \in Z} \operatorname{Pr}(z \mid d) \operatorname{Pr}(s \mid z) \operatorname{Pr}(w \mid z, s) $$

dove:

* $d$ è il documento.
* $s$ è il segmento.
* $w$ è la parola.
* $z$ è l'argomento.
* $Z$ è l'insieme di tutti gli argomenti.

Il processo di generazione di un documento, un segmento e una parola può essere descritto come segue:

$$
\begin{align}
&\text{Select a document } d \text{ from } \mathcal{D} \Rightarrow \operatorname{Pr}(d) \\
&\text{For each segment } s \in S_d: \\
&\text{1. Choose a topic } z \text{ for the document } d \Rightarrow \operatorname{Pr}(z \mid d) \\
&\text{2. Associate topic-to-segment probability to the segment } s \text{ for the selected topic } z \Rightarrow \operatorname{Pr}(s \mid z) \\
&\text{3. For each word } w \text{ in the segment } s: \\
&\text{Choose a word } w \text{ from the current topic and segment } \Rightarrow \operatorname{Pr}(w \mid z, s)
\end{align}
$$

### Algoritmo di inferenza EM per LDA

L'algoritmo di inferenza EM (Expectation-Maximization) per LDA prevede due passaggi:

* **E-step:** Calcolo della probabilità a posteriori dell'argomento dato il documento, il segmento e la parola:

$$ \begin{aligned} \text{E-step} \quad \Pr(z|d, s, w) &= \frac{\Pr(z, d, s, w)}{\Pr(d, s, w)} = \frac{\Pr(z|d)\Pr(s|z)\Pr(w|z, s)}{\sum_{z \in Z} \Pr(z|d)\Pr(s|z)\Pr(w|z, s)} \\ 
\end{aligned} $$

* **M-step:** Aggiornamento delle probabilità di argomento, segmento e parola:

$$
\begin{aligned}
\text{M-step} \quad \mathbf{E}[l] &= \sum_{d \in D} \sum_{s \in S} \sum_{w \in V} n(d, s, w) \times \sum_{z \in Z} \Pr(z|d, s, w) \log(\Pr(d, s, w)) \ 
\\
\text{Update formulas} \quad \Pr(z|d) &\propto \sum_{s \in S} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w) \ 
\\
\Pr(s|z) &\propto \sum_{d \in D} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w) \ 
\\
\Pr(w|z, s) &\propto \sum_{d \in D} n(d, s, w) \Pr(z|d, s, w) \end{aligned} 
$$

dove:

* $n(d, s, w)$ è il numero di volte in cui la parola $w$ appare nel segmento $s$ del documento $d$.
* $V$ è il vocabolario.

L'algoritmo EM viene ripetuto fino a convergenza, ovvero fino a quando le probabilità non cambiano significativamente tra le iterazioni.

### Applicazione del Text Tiling con LDA

Il Text Tiling con LDA può essere utilizzato per segmentare un documento in base ai suoi argomenti. L'algoritmo EM viene utilizzato per inferire le probabilità di argomento, segmento e parola. I segmenti vengono quindi identificati in base ai cambiamenti di argomento, come evidenziato dai minimi locali nella curva delle similarità.

# Valutazione dei modelli di argomenti

I modelli di topic, come Latent Dirichlet Allocation (LDA) e Probabilistic Latent Semantic Analysis (PLSA), cercano di scoprire temi latenti all'interno di un corpus di documenti. Questi modelli si basano sull'idea che ogni documento sia una miscela di topic e che ogni topic sia caratterizzato da una distribuzione di parole.

A differenza dei modelli linguistici neurali, la valutazione dei modelli di topic tradizionalmente non si concentra su aspetti come verbosità, fluency, affettività, confabulazione o bias. Storicamente, si misurano **coerenza** e **perplessità**.

## Coerenza

Gli argomenti sono rappresentati dalle prime N parole con la probabilità più alta di appartenere a quell'argomento specifico. Questo N è relativamente grande rispetto al numero di topic che caratterizzano un documento.

La coerenza misura la co-occorrenza tra parole e argomenti: maggiore è il punteggio, migliore è la coerenza. Gli incrementi diminuiscono all'aumentare del numero di argomenti.

#### Punteggio di coerenza UMass

Per ogni argomento, il punteggio medio a coppie sulle prime N parole che descrivono quell'argomento è calcolato come segue:

$$C_{\text{UMass}}(w_{i},w_{j})=\log \frac{D(w_{i},w_{j})+1}{D(w_{i})}$$

Dove:

* $D(w_{i}, w_{j})$: numero di volte in cui le parole $w_{i}$ e $w_{j}$ compaiono insieme nei documenti.
* $D(w_{i})$: numero di volte in cui la parola $w_{i}$ compare da sola.

#### Punteggio di coerenza UCI

Il punteggio di coerenza UCI si basa su finestre scorrevoli e sul PMI (Probabilità di Informazione Mutua) di tutte le coppie di parole utilizzando le prime N parole per occorrenza.

* La co-occorrenza delle parole è vincolata a una *finestra scorrevole* e non all'intero documento.
$$C_{\text{UCI}}(w_{i},w_{j})=\log \frac{P(w_{i},w_{j})+1}{P(w_{i})\cdot P(w_{j})}$$

Dove:

* $P(w_{i}, w_{j})$: probabilità di vedere $w_{i}$ e $w_{j}$ insieme in una finestra scorrevole.
* $P(w_{i})$: probabilità di vedere $w_{i}$ in una finestra scorrevole.
 * Entrambe le probabilità sono stimate dall'intero corpus di oltre due milioni di articoli di Wikipedia in inglese utilizzando una finestra scorrevole di 10 parole.
- È un approccio robusto ma parametrico: i parametri sono 2: stride e overlap

#### Coerenza basata sulla similarità intra/inter-argomento

Per ogni coppia di argomenti, la coerenza è calcolata come la media della similarità intra-argomento divisa per la similarità inter-argomento.

* **Similarità intra-argomento:** media della similarità tra ogni possibile coppia di parole principali in quell'argomento.
* **Similarità inter-argomento:** media della similarità tra le parole principali di due argomenti diversi.
* Si presume che le embedding delle parole siano disponibili.
* Il coseno è utilizzato come misura di similarità. 

### Coerenza e Separazione dei Topic

Se due topic hanno distribuzioni di probabilità simili, significa che attivano più o meno gli stessi termini con la stessa probabilità. Questa osservazione suggerisce che la valutazione dei topic non dovrebbe limitarsi alla coerenza interna (compattezza) dei cluster, ma dovrebbe anche considerare la loro **separazione**. 

### Entropia e Cross-Entropia

Un **topic**, o **classe**, è un concetto che introduciamo per raggruppare attributi. Per valutare il grado di incertezza associato a un topic, utilizziamo l'**entropia** della sua distribuzione di probabilità. In alcuni casi, potremmo anche considerare la **cross-entropia**, che misura l'incertezza di una distribuzione rispetto a un'altra.

## Distanza tra distribuzioni

### Divergenza di Kullback-Leibler

La divergenza di Kullback-Leibler (da Q a P), nota anche come entropia relativa di P rispetto a Q, è definita come:

$$D_{KL}(P\|Q) = -\sum_{x\in X}P(x)\log\left( \frac{Q(x)}{P(x)} \right)$$

La divergenza di Kullback-Leibler presenta le seguenti proprietà:

* **Non negatività:** $D_{KL}(P\|Q) \ge 0$, con uguaglianza se e solo se P = Q.
* **Asimmetria:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$.
* **Non metrica:** Non soddisfa la disuguaglianza triangolare.

### Divergenza di Jensen-Shannon

La divergenza di Jensen-Shannon è definita come:

$$D_{JS} = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$$

dove M è la media di P e Q, ovvero:

$M = \frac{1}{2}(P + Q)$.

### Cross-entropia

Quando si confrontano due distribuzioni di probabilità, si utilizza spesso la cross-entropia, che può essere relativizzata o normalizzata. La cross-entropia misura la differenza tra la distribuzione di probabilità prevista e la distribuzione di probabilità effettiva. 

## Perplessità

La perplessità è una misura del grado di incertezza di un modello linguistico (LM) quando genera un nuovo token, mediato su sequenze molto lunghe. 
In termini di processi generativi, indica l'incertezza del modello nello scegliere la prossima parola da campionare durante la generazione del documento. **Minore è la perplexity, migliore è il modello.**

**Nota:** la perplessità dipende dalla specifica tokenizzazione utilizzata dal modello, quindi il confronto tra due LM ha senso solo se entrambi utilizzano la stessa tokenizzazione.

La perplessità è la prima misura da utilizzare per valutare le prestazioni di un modello linguistico.

### Comprendere la Perplexity

La formula della perplexity è una funzione esponenziale di una media normalizzata di logaritmi di probabilità. In sostanza, è una funzione esponenziale dell'entropia. 

Per comprendere meglio il concetto, è utile richiamare alcuni principi fondamentali:

* **Variabile 'x':** Rappresenta la sorgente informativa (es. una registrazione universitaria). I valori di 'x', indicati con xᵢ, sono le parole, i token.
* **Entropia di 'x':** Misura il tasso di informazione trasmesso dalla sorgente. Il limite superiore per la codifica dei token è rappresentato dall'entropia. Questo limite è massimo quando la distribuzione degli eventi è uniforme (tutti i possibili risultati hanno la stessa probabilità).
* **Token frequenti:** Gli eventi più probabili, che minimizzano la lunghezza della descrizione.
* **Componente informativa:** La componente informativa della variabile che modella la frequenza dei token, P(x), è -log P(x).
* **Perplexity:** Una variabile aleatoria uguale a 2 elevato all'entropia di x, H(x).

### Definizione

Sia X una fonte di informazioni testuali (variabile aleatoria sorgente), e i suoi valori x siano token.

Ricordiamo: l'entropia di X è una misura del tasso di informazioni prodotte da X. L'entropia è massima quando è possibile osservare tutti gli esiti della variabile aleatoria. È indicata con $H[X]$.

* **Teorema di Codifica Senza Rumore di Shannon:** il limite inferiore per la lunghezza attesa del codice di codifica dei token.
* per ogni token, la sua lunghezza di codifica è $-log(p(x))$.
	* ovvero, i token frequenti dovrebbero essere assegnati a codici più brevi: minimizzano la lunghezza della descrizione.

##### Perplessità di una singola variabile casuale X

$$PP[X]:=2^{H[X]}$$
### Perplessità di un processo stocastico

Siamo interessati a un processo stocastico χ di sequenze **non i.i.d.** (indipendenti e identicamente distribuite) di variabili casuali (X₁, X₂, …).  Le occorrenze di parole all'interno di un testo non sono indipendenti.

#### Prima ipotesi: Stazionarietà

La probabilità di osservare una parola non cambia a seconda della finestra di testo considerata.  Questa ipotesi non è vera per un documento di testo, poiché le parole sono distribuite in modo diverso all'inizio e alla fine di un testo. Tuttavia, ciò implica che il limite dell'entropia media per token coincide con il limite dell'entropia media dell'ultimo token:

* $\lim_{ n \to \infty } \frac{1}{n}H[X_{1},\dots,X_{n}]$
* $\lim_{ n \to \infty } \frac{1}{n}H[X_{n}|X_{1},\dots,X_{n-1}]$


#### Seconda ipotesi: Ergodicità

Se il numero di osservazioni di una variabile è molto grande, il valore atteso coincide con la media delle misurazioni.  Inoltre, per sequenze molto lunghe, la media dei logaritmi negativi delle probabilità sui vari step di generazione approssima l'entropia. L'ergodicità garantisce che l'aspettativa $𝔼[X₁]$ di qualsiasi singola variabile casuale X₁ sulla distribuzione P del processo χ può essere sostituita con la media temporale di una singola sequenza molto lunga (x₁, x₂, …) estratta da χ (Teorema Ergodico di Birkhoff):

* $\frac{1}{n}\sum_{i=1}^n X_{i}\to_{n \to \infty} E_{p}[X_{1}]\text{ con probabilità 1}$

Questo risultato, insieme al Teorema di Shannon-McMillan-Breiman, implica:

* $-\frac{1}{n}\log p(X_{1},\dots,X_{n})\to_{n\to \infty }H[X]\text{ con probabilità 1}$


#### Applicazione all'entropia

Il risultato precedente sarebbe perfetto se conoscessimo le distribuzioni di probabilità p(x₁, x₂, …).  Tuttavia, non le conosciamo, quindi ricorriamo a un modello linguistico q(x₁, x₂, …) come approssimazione. Un limite superiore al tasso di entropia per p è la cross-entropia del modello Q (il modello linguistico) rispetto alla sorgente P (il linguaggio naturale):

* $CE[P,Q]:=\lim_{ n \to \infty }-E_{p}\log q(X_{n}|X_{<n})=\lim_{ n \to \infty } -\frac{1}{n}E_{p}\log q(X_{1},\dots,X_{n})$


### Perplessità di un modello Q per un linguaggio considerato come una sorgente P sconosciuta:

La *perplexity* del modello linguistico Q rispetto al linguaggio naturale è 2 elevato alla cross-entropia. In pratica, calcoliamo 2 elevato alla media, per ogni parola in ogni documento, della distribuzione di probabilità delle parole in esso contenute, indicata con w,d. N,d è il numero di parole nel documento d, quindi normalizziamo i contributi di ciascuna parola. Sommiamo su tutti i documenti e poi eleviamo a 2. Questo è il valore della *perplexity*. 
Formalmente:

$$PP[P,Q]:=2^{CE[P,Q]}$$
dove
$$-\frac{1}{n}\log q(X_{1},\dots,X_{n})\to_{n\to \infty}CE[P,Q]$$

### Perplessità di un campione di holdout:

$$
-\frac{\sum_{d=1}^M\log p(w_{d})}{\sum_{d=1}^MN_{d}}
$$
* M è il numero di documenti nel campione di test.
* $w_{d}$ rappresenta le parole nel documento d.
* $N_{d}$ il numero di parole nel documento d.

### In LDA: 

$$\log p(w|\alpha,\beta)=E[\log p(\theta,z,w|\alpha,\beta)]-E[\log q(\theta,z)]$$
Nel caso di LDA, valutiamo alpha e beta, i parametri di dispersione. Questa forma deriva dall'utilizzo della variational inference, una tecnica più accurata rispetto al Gibbs sampling.

### In SGM:

$$Pr(d,S_{d},V)=Pr(d)\prod_{S\in S_{d}}\sum_{z\in Z}Pr(z|d)Pr(s|z)\prod_{w\in V}Pr(w|z,s)$$
Anche in SGM, il modello LDA adattato a stream di dati, la *perplexity* si basa sul calcolo di probabilità simili, valutate per ogni documento, utilizzando i valori calcolati durante la fase di test.
