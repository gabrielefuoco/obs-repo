
# Language Modeling

**Definizione:** Task autoregressivo per la generazione di testo.

* **Input:** Sequenza di parole osservate $x_1, ..., x_t$.
* **Output:** Predizione della parola successiva $x_{t+1}$.
* **Vocabolario:** Noto a priori.
* **Distribuzione di probabilità:** $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$
* **Probabilità di generare un testo T:** $P(x^{(1)}, \ldots, x^{(T)}) = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)})$

**Applicazioni:**

* Machine Translation (encoding/decoding)
* Speech Recognition
* Spelling/Grammar Correction
* Summarization (estrattiva e astrattiva - quest'ultima come caso particolare di language modeling)


## N-gram Language Models

**Definizione:** Stima la probabilità della parola successiva contando le occorrenze di n-gram in un corpus.

* **n-gram:** Sequenza di *n* token consecutivi.
* **Esempi:** Unigrammi, Bigrammi, Trigrammi, Four-grammi.
* **Assunzione di Markov:** $x^{(t+1)}$ dipende solo dalle *n-1* parole precedenti.
* **Probabilità condizionata:** $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)}) \approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})}$

**Esempio (modello 4-gram):**

* Testo: "gli studenti aprirono i"
* $P(w \mid \text{gli studenti aprirono}) = \frac{\text{count(gli studenti aprirono } w)}{\text{count(gli studenti aprirono)}}$
* Esempio numerico: $P(\text{libri} \mid \text{gli studenti aprirono i}) = 0.4$, $P(\text{compiti} \mid \text{gli studenti aprirono i}) = 0.1$

**Gestione della sparsità:**

* **Problema 1 (Numeratore = 0):** Smoothing (aggiunta di δ al conteggio)
* **Problema 2 (Denominatore = 0):** Backoff (condizionamento su n-gram più corti)

**Limiti:**

* Aumentare *n* peggiora la sparsità.
* *n* tipicamente non supera 5.
* Alta sparsità con *n* elevato, causando sottostima delle probabilità.


## I. Modello Linguistico Neurale a Finestra Fissa

**A. Input e Output:**

* **Input:** Sequenza di parole $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$
* **Output:** Distribuzione di probabilità della parola successiva $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$

**B. Rappresentazione delle Parole:**

* **Embeddings:** Ogni parola rappresentata da un embedding.
* **Concatenazione:** Gli embeddings vengono concatenati in un vettore $\mathbf{e} = [e^{(1)}, e^{(2)}, e^{(3)}, e^{(4)}]$.

**C. Architettura del Modello:**

* **Layer Nascosto:** $\mathbf{h} = f(\mathbf{W} \mathbf{e} + \mathbf{b}_1)$ (f = funzione di attivazione, W = matrice di pesi, b<sub>1</sub> = bias)
* **Distribuzione di Output:** $\mathbf{y} = \text{softmax}(\mathbf{U} \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{V}$ (U = matrice di trasformazione, b<sub>2</sub> = bias, V = dimensione del vocabolario)

**D. Vantaggi rispetto ai modelli n-gram:**

* Nessun problema di sparsità.
* Non richiede la memorizzazione di tutti gli n-gram osservati.

**E. Limiti:**

* Finestra di contesto fissa troppo piccola.
* Aumento esponenziale della dimensione di **W** con l'aumento della finestra.
* Impossibilità di catturare tutte le dipendenze a lungo termine.
* Asimmetria nell'elaborazione dell'input (pesi diversi per parole diverse nella finestra).
* Granularità: diminuisce all'aumentare di *n*, riducendo la capacità di catturare dipendenze a lungo termine, portando a probabilità poco informative e testo incoerente.


## II. Reti Neurali Ricorrenti (RNN)

**A. Obiettivo:** Condividere i pesi (**W**) per ogni parola nella sequenza di input, permettendo l'elaborazione di sequenze di lunghezza variabile con una rete di dimensione fissa.

**B. Output:** Generato ad ogni timestep o solo all'ultimo, a seconda del task.


## III. Confronto tra Modelli

* **Modelli a finestra fissa:** limitati dalla dimensione fissa della finestra, che impedisce la cattura di dipendenze a lungo termine e introduce asimmetrie nell'elaborazione.
* **RNN:** superano i limiti dei modelli a finestra fissa grazie alla condivisione dei pesi, permettendo l'elaborazione di sequenze di lunghezza variabile.


## Reti Neurali Ricorrenti (RNN) per Modelli Linguistici

**I. Architettura RNN**

**A. Input:** Sequenza di parole, rappresentate da vettori one-hot $\mathbf{x}^{(t)} \in \mathbb{R}^{|\mathcal{V}|}$.

**B. Word Embeddings:** $\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$ trasforma i vettori one-hot in rappresentazioni vettoriali dense.

**C. Stati Nascosti:** $h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$ aggiorna lo stato nascosto in base allo stato precedente e all'embedding corrente.

---

# Recurrent Neural Networks (RNNs) e Backpropagation Through Time (BPTT)

## I. Architettura RNN e Addestramento

Lo stato iniziale è rappresentato da $h^{(0)}$.

**A. Distribuzione di Output:** La distribuzione di probabilità sulle parole del vocabolario è calcolata come:  $y^{(t)} = \text{softmax}(h^{(t)} + b_o) \in \mathbb{R}^{|\mathcal{V}|}$, dove $|\mathcal{V}|$ è la dimensione del vocabolario.

**B. Addestramento:**

* **Funzione di Costo:**  Cross-entropy ad ogni timestep: $J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w) = -\log \hat{y}^{(t)}_{x_{t+1}}$.  $y^{(t)}$ rappresenta la distribuzione di probabilità vera, mentre $\hat{y}^{(t)}$ è la distribuzione di probabilità predetta.

* **Loss Totale:** La loss totale è la media delle loss su tutti i timestep dell'intero training set. Il calcolo su tutto il corpus contemporaneamente è computazionalmente costoso.

**C. Pro e Contro:**

* **Pro:**
    * Simmetria dei pesi: gli stessi pesi sono applicati ad ogni timestep.
    * Dimensione del modello costante: indipendente dalla lunghezza della sequenza.

* **Contro:**
    * Lunghezza della sequenza limitata: il problema del vanishing gradient si manifesta per sequenze lunghe.
    * Tempo di addestramento lungo.


## II. Backpropagation Through Time (BPTT)

**A. Scopo:** Addestramento di Recurrent Neural Networks (RNNs).

**B. Metodo:** Variante dell'algoritmo di backpropagation per calcolare i gradienti dei pesi rispetto alla funzione di costo.

**C. Calcolo del gradiente per pesi ripetuti (es. $\boldsymbol{W}_{h}$):**

* $\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} }\mid_{(i)}$: somma dei gradienti calcolati ad ogni timestep in cui il peso contribuisce.
* Si applica la regola della catena multivariabile.

**D. Regola della catena multivariabile:**

* Per $f(x,y)$, $x(t)$, $y(t)$: $\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$.
* Applicata iterativamente per calcolare i gradienti attraverso i timestep.
* Propagazione all'indietro dei gradienti attraverso il tempo.


## III. Valutazione del Modello

* **Metrica:** Perplexity.
* **Formula:** $\text{Perplexity} = \left( \prod_{t=1}^{T} \frac{1}{P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})} \right)^{1/T}$.
* **Interpretazione:** Inverso della probabilità geometrica media di predire correttamente le parole.
* **Relazione con la cross-entropy:** $\text{Perplexity} = \exp(J(\theta))$, dove $J(\theta)$ è la cross-entropy loss media.
* **Miglior modello:** Perplexity inferiore.


## IV. Vanishing Gradient Problem

**A. Causa:** Probabilità molto piccole durante il calcolo ricorsivo delle derivate in BPTT.

**B. Effetto:** Gradienti che diminuiscono esponenzialmente procedendo indietro nel tempo, rendendo difficile l'addestramento su sequenze lunghe.

**C. Esemplificazione con σ(x) = x:**

* $\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W_{h}$

**D. Analisi del gradiente:**

* $\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell}$, dove $\ell = i - j$.

**E. Condizione per il vanishing gradient:** Autovalori di $W_{h}$ con modulo minore di 1. Il termine $W_{h}^{\ell}$ diventa esponenzialmente piccolo all'aumentare di $\ell$.  Con funzioni di attivazione non lineari, la condizione diventa $|\lambda_i| < \gamma$ (γ dipende da dimensionalità e funzione di attivazione).


## V. Exploding Gradient Problem

**A. Causa:** Autovalori di $W_h$ maggiori di 1 in modulo.

**B. Effetto:** Aggiornamenti SGD troppo grandi ($\theta^{nuovo} = \theta^{vecchio} - \alpha \nabla_{\theta} J(\theta)$), portando ad aggiornamenti errati, *loss* elevata e potenziali valori Inf o NaN nella rete.

**C. Soluzione: Gradient Clipping**

* **Algoritmo:** Se $\|\mathbf{g}\| >$ soglia, allora $\mathbf{g} \leftarrow \frac{\text{soglia}}{\|\mathbf{g}\|} \mathbf{g}$.
* **Intuizione:** Ridimensionamento del gradiente senza cambiarne la direzione.


## VI.  Definizione di h<sup>(t)</sup> e Vanishing Gradient (Riepilogo)

**A. h<sup>(t)</sup>:** Applicazione di una funzione di attivazione non lineare (es. tanh, ReLU) alla combinazione lineare di: embedding dell'input al timestep `t`, bias e trasformazione dello stato nascosto al timestep precedente.

**B. Vanishing Gradient:** La derivata della funzione di costo J rispetto allo stato nascosto a passi precedenti contiene $W_h^\ell$ (dove ℓ è la distanza temporale). Se gli autovalori di $W_h$ sono minori di 1 in modulo ($|\lambda_i| < 1$), $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(j)}} = \sum_{k=1}^n c_k \lambda_{k}^\ell \mathbf{q}_k$ tende a 0 per grandi ℓ, causando il *vanishing gradient*.



---

# Limiti delle RNN Standard e Soluzioni

## I. Limiti delle RNN Standard

Il problema principale delle Recurrent Neural Network (RNN) standard risiede nella difficoltà di preservare informazioni su molti *timestep*.  Questo è causato dalla costante riscrittura dello stato nascosto ad ogni passo temporale:

$$h^{(t)} = \sigma (W_{hh} h^{(t-1)} + W_{xz} x^{(t)} + b)$$

Di conseguenza, l'aggiornamento dei pesi si basa principalmente su effetti a breve termine, trascurando quelli a lungo termine, portando al fenomeno del *vanishing gradient*.


## II. Soluzioni al Problema del *Vanishing Gradient*

Esistono due approcci principali per mitigare il problema del *vanishing gradient*:

### Approccio 1: Memoria Separata

Questo approccio utilizza una memoria aggiuntiva per gestire le informazioni a lungo termine.  Un esempio chiave è l'architettura LSTM (Long Short-Term Memory).

### Approccio 2: Connessioni Dirette e Lineari

Questo approccio crea connessioni più dirette e lineari nel modello. Esempi includono i meccanismi di *attention* e le connessioni residuali.  Si tratta di un intervento architetturale che permette di aggiornare lo stato rispetto a un contesto più breve, mantenendo un buffer separato per il contesto precedente.


## III. Long Short-Term Memory (LSTM)

L'obiettivo delle LSTM è risolvere il problema del *vanishing gradient* nelle RNN aggiungendo una "memoria" interna. La componente chiave è la cella di memoria (`c`), che gestisce le informazioni a lungo termine.  Il controllo della lettura, scrittura e cancellazione di informazioni nella cella di memoria è gestito da tre *gate*: Forget, Input e Output. Lo stato di questi *gate* è dinamico e varia in base all'input e al contesto.


## IV. Calcolo degli Stati in LSTM

L'input è una sequenza di input $x^{(t)}$, mentre l'output è una sequenza di stati nascosti $h^{(t)}$ e stati delle celle $c^{(t)}$.  Ad ogni *timestep* `t` vengono eseguite le seguenti operazioni:

* **Forget Gate:** $f^{(t)} = \sigma \left( W_f h^{(t-1)} + U_f x^{(t)} + b_f \right)$ (determina quali informazioni dallo stato della cella precedente mantenere)
* **Input Gate:** $i^{(t)} = \sigma \left( W_i h^{(t-1)} + U_i x^{(t)} + b_i \right)$ (determina quali parti del nuovo contenuto scrivere nella cella)
* **Output Gate:** $o^{(t)} = \sigma \left( W_o h^{(t-1)} + U_o x^{(t)} + b_o \right)$ (determina quali parti della cella inviare allo stato nascosto)
* **Nuovo Contenuto della Cella:** $\tilde{c}^{(t)} = \tanh \left( W_c h^{(t-1)} + U_c x^{(t)} + b_c \right)$ (nuovo contenuto potenziale da scrivere nella cella)


## LSTM: Schema Riassuntivo

### I. Meccanismo di funzionamento delle LSTM:

**A. Stato della Cella ($c^{(t)}$):** Aggiorna lo stato memorizzando nuove informazioni e dimenticando quelle obsolete.

* Formula: $c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)}$
    * $c^{(t)}$: stato della cella al timestep t
    * $f^{(t)}$: gate di dimenticanza
    * $c^{(t-1)}$: stato della cella al timestep precedente
    * $i^{(t)}$: gate di input
    * $\tilde{c}^{(t)}$: nuovo contenuto della cella
    * $\circ$: prodotto elemento-wise (Hadamard)

**B. Stato Nascosto ($h^{(t)}$):** Genera lo stato nascosto leggendo informazioni dallo stato della cella.

* Formula: $h^{(t)} = o^{(t)} \circ \tanh(c^{(t)})$
    * $h^{(t)}$: stato nascosto al timestep t
    * $o^{(t)}$: gate di output
    * $\tanh$: funzione tangente iperbolica

**C. Gate:** Ogni gate ($f^{(t)}$, $i^{(t)}$, $o^{(t)}$) è ottenuto tramite una trasformazione non lineare (sigmoide) di una combinazione lineare dell'input $x^{(t)}$ e dello stato nascosto $h^{(t-1)}$. Ogni gate ha parametri distinti. Tutti i vettori hanno lunghezza n.


### II. Risoluzione del *Vanishing Gradient*:

**A. Preservazione delle informazioni:** L'architettura LSTM facilita la preservazione delle informazioni su molti *timestep* grazie ai *gate*.  Ad esempio, $f^{(t)}=1$ e $i^{(t)}=0$ preservano indefinitamente l'informazione.

**B. Confronto con RNN standard:** Le LSTM raggiungono circa 100 *timestep*, contro i circa 7 delle RNN standard.

**C. Skip Connections/Connessioni Residuali:** Alternative per preservare dipendenze a lungo raggio, permettendo al gradiente di passare direttamente tra *timestep*, evitando il *vanishing gradient*.

* **Connessioni dense (DenseNet):** ogni layer connesso a ogni altro layer successivo.
* **Highway connections (HighwayNet):** meccanismo di gating per determinare la parte di informazione da passare direttamente.


### III. Estensione Bidirezionale:

**A. Problema:** L'ambiguità in frasi richiede un contesto più ampio (es. "terribly").

**B. Soluzione:** Due layer RNN paralleli (forward e backward) elaborano l'input in direzioni opposte.

**C. ![[immagine_bidirezionale]]**

---

# Appunti su Reti Neurali Ricorrenti e Traduzione Automatica

## Combinazione Bidirezionale e Reti Neurali Ricorrenti

Lo stato nascosto  $h^{(t)}$ in una rete neurale ricorrente bidirezionale è la combinazione di $h^{(t)}_{forward}$ (stato derivato dalla sequenza letta da sinistra a destra) e $h^{(t)}_{backward}$ (stato derivato dalla sequenza letta da destra a sinistra).

**Applicabilità:** La bidirezionalità è applicabile a task non autoregressivi, migliorando la rappresentazione contestuale.  Non è direttamente applicabile a task autoregressivi come il language modeling, dove il contesto futuro non è disponibile durante la predizione.


### I. Reti Neurali Ricorrenti (RNN)

#### A. Bidirezionalità

* Applicabile solo con accesso all'intera sequenza di input.
* Non adatta al Language Modeling (solo contesto sinistro disponibile).
* Molto potente per encoding (es. BERT).

#### B. Multi-Layer RNN

* Aggiunta di più layer per rappresentazioni più complesse.
* RNN inferiori: feature di basso livello.
* RNN superiori: feature di alto livello.
* Chiamate anche RNN stacked.
* Cattura relazioni di diverso ordine (locale e globale).
* Migliora la capacità di catturare proprietà grammaticali e sintattiche (layer inferiori) e relazioni semantiche (layer superiori).
* Tipicamente 1-3 layer; fino a 8 con skip connection, 12 in BERT, 24 in Transformer specifici per encoding.


## Traduzione Automatica (Machine Translation)

### I. Sfida

Task complesso fino al 2015, caratterizzato da input e output con ruoli diversi e dalla necessità di modellare la relazione tra due linguaggi (simile alla summarization).

### II. Apprendimento Probabilistico

Trovare la migliore frase inglese `y` data una frase francese `x`: $\arg\max_y P(y|x)$

Utilizzo della regola di Bayes: $\arg\max_y P(x|y)P(y)$

### III. Componenti del Modello

* **1. Modello di Traduzione:** Apprende la traduzione di parole e frasi (fedeltà). Addestrato su dati paralleli (coppie di frasi tradotte).
* **2. Modello Linguistico:** Modella la correttezza grammaticale in inglese (fluidità). Addestrato su dati monolinguali (corpus di testo in inglese).

I due modelli non sono addestrati separatamente; il modello linguistico è condizionato all'input.


## Modelli Seq2Seq per la Traduzione Automatica

### I. Modello Seq2Seq

Utilizza due RNN (o LSTM) multi-layer:

* **Encoder:** Codifica la frase di input in una rappresentazione vettoriale. Lo stato nascosto finale ($h^{(0)}$) inizializza il decoder.
* **Decoder:** Genera la frase di output parola per parola, condizionata dall'encoder. Inizia con un token "START" e usa un approccio autoregressivo.

Predizione tramite cross-entropy, massimizzando $P(y|x)$. L'encoder può essere bidirezionale per un contesto migliore.

### II. Architettura Encoder-Decoder

* **Encoder:** Crea una rappresentazione vettoriale compatta dell'input.
* **Decoder:** Genera l'output dalla rappresentazione dell'encoder.
* **Applicazioni:** Traduzione automatica, summarization, dialogo, parsing, generazione di codice.

**Modello Linguistico Condizionale:**

* Modello linguistico: predice la prossima parola (`y`).
* Condizionale: le predizioni dipendono dalla frase sorgente (`x`).

**Traduzione Automatica (NMT):** Calcola direttamente $P(y|x) = P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \ldots P(y_T|y_1, \ldots, y_{T-1}, x)$.

* **Addestramento:** Backpropagation sui pesi dell'encoder e del decoder (convergenza lenta).
* **Collo di Bottiglia:** Il decoder è condizionato solo dall'output globale dell'encoder.


### III. Strategie di Decoding

* **Decoding Greedy:** Seleziona la parola con la probabilità più alta ad ogni passo. Non ottimale a lungo termine.
* **Ricerca Esaustiva:** Massimizza $P(y|x) = \prod_{t=1}^T P(y_{t}|y_{1},\dots,y_{t-1},x)$, ma ha complessità computazionale $O(V^T)$ (impraticabile).
* **Beam Search:** Soluzione per la complessità computazionale della ricerca esaustiva. (Dettagli non specificati nel testo).


## Beam Search

### I. Scopo

Trovare le traduzioni più probabili in modo efficiente. Mantiene le `k` ipotesi (traduzioni parziali) più probabili ad ogni *timestep*. `k` è la dimensione del beam (tipicamente 5-10).

### II. Score delle ipotesi

$\text{score}(y_1, \ldots, y_t) = \log P_{LM}(y_1, \ldots, y_t | x) = \sum_{i=1}^{t} \log P_{LM}(y_i | y_1, \ldots, y_{i-1}, x)$

Score più alto indica maggiore probabilità (score negativi).

### III. Efficienza

Non garantisce la soluzione ottimale, ma è molto più efficiente della ricerca esaustiva.


## BLEU (Bilingual Evaluation Understudy)

### I. Scopo

Valutare la qualità di una traduzione automatica. Confronta l'output del traduttore con traduzioni umane di riferimento ("ground truth"). Si basa su precisioni n-gram (da 1-gram a 4-gram). Include una penalità di brevità.

---

# Valutazione di Traduzioni Automatiche e Meccanismo di Attention

## I. BLEU Score

**Precisione "clipped":** Conta le corrispondenze parola per parola, limitando il conteggio al numero massimo di occorrenze nella frase di riferimento.

**Geometric Average (Clipped) Precision Scores:**  $\prod_{n=1}^{N} p_n^{w_n}$

**Brevity Penalty:** $\begin{cases} 1, & \text{if } c > r \\ e^{(1-r/c)}, & \text{if } c \le r \end{cases}$ (dove *c* = lunghezza della traduzione, *r* = lunghezza della frase di riferimento)

**Punteggio BLEU:** Prodotto della precisione geometrica media e della penalità di brevità.

**Vantaggi:** Calcolo rapido, intuitivo, indipendente dalla lingua, gestibile con più frasi di riferimento.

**Svantaggi:** Ignora significato, importanza e ordine delle parole (es. "La guardia arrivò tardi a causa della pioggia" e "La pioggia arrivò tardi a causa della guardia" potrebbero avere lo stesso punteggio unigramma).


## II. Meccanismo di Attention in Modelli Seq2Seq

**A. Obiettivi:** Creare connessioni dirette tra decoder ed encoder, permettendo al decoder di focalizzarsi su parti specifiche dell'input ad ogni timestep. Il decoder "attende" all'encoder, generando parole a partire da "START", basandosi su un riassunto ponderato delle informazioni dell'encoder.

**B. Calcolo dell'Attention:**

1. **Attention Scores:** $\boldsymbol{e}^t = [\boldsymbol{s}_t^T \boldsymbol{h}_1, \ldots, \boldsymbol{s}_t^T \boldsymbol{h}_N] \in \mathbb{R}^N$ (dove $\boldsymbol{h}_i$ sono gli stati nascosti dell'encoder e $\boldsymbol{s}_t$ lo stato nascosto del decoder al timestep *t*).

2. **Distribuzione di Attenzione:** $\boldsymbol{\alpha}^t = \operatorname{softmax}(\boldsymbol{e}^t) \in \mathbb{R}^N$ (distribuzione di probabilità che somma a 1).

3. **Output di Attenzione (Vettore di Contesto):** $\boldsymbol{a}_t = \sum_{i=1}^N \alpha_{i}^t \boldsymbol{h}_i \in \mathbb{R}^d$ (somma pesata degli stati nascosti dell'encoder).

4. **Concatenazione:** $[\boldsymbol{a}_t; \boldsymbol{s}_t] \in \mathbb{R}^{2d}$ (concatenazione dell'output di attenzione con lo stato nascosto del decoder).


**C. Vantaggi dell'Attention in NMT:**

* Migliora le prestazioni.
* Focalizzazione su parti specifiche della frase sorgente.
* Modello più "umano".
* Risoluzione del *bottleneck*.
* Mitigazione del *vanishing gradient*.
* Maggiore interpretabilità.


**D. Tipi di Attention:**

* **Attention a prodotto scalare:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{h}_i \in \mathbb{R}$
* **Attention moltiplicativa:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{W} \boldsymbol{h}_i \in \mathbb{R}$
* **Attention moltiplicativa a rango ridotto:** $\boldsymbol{e}_i = \boldsymbol{s}^T (\boldsymbol{U}^T \boldsymbol{V}) \boldsymbol{h}_i = (\boldsymbol{U} \boldsymbol{s})^T (\boldsymbol{V} \boldsymbol{h}_i)$
* **Attention additiva:** $\boldsymbol{e}_i = \boldsymbol{v}^T \tanh(\boldsymbol{W}_1 \boldsymbol{h}_i + \boldsymbol{W}_2 \boldsymbol{s}) \in \mathbb{R}$


**E. Fasi Generali dell'Attention:**

1. Calcolo degli *attention scores* ($\boldsymbol{e}$).
2. Applicazione della softmax per ottenere $\boldsymbol{\alpha}$.
3. Calcolo della somma pesata dei valori usando $\boldsymbol{\alpha}$, ottenendo l'output di attenzione ($\boldsymbol{a}$).


## III. Attention Mechanism: Scopo e Applicazioni

**Scopo:** Risolvere il *conditioning bottleneck* dei modelli encoder-decoder. Invece di basarsi solo sull'output finale dell'encoder, permette ad ogni *timestep* del decoder di accedere a una combinazione pesata delle rappresentazioni di ogni parola dell'input.  Questo evita la perdita di informazioni dovuta al condizionamento solo sull'output globale dell'encoder.

**Attention come Tecnica Generale di Deep Learning:**

**A. Definizione:** Calcolo di una somma ponderata di vettori (valori) in base a un vettore *query*. La somma ponderata rappresenta un riassunto selettivo delle informazioni nei valori.  È applicabile a diverse architetture e task.

**Meccanismo di Attention:**

* **Funzione principale:** Calcolo di una somma ponderata di valori, condizionata da una query.
* **Selezione dei valori:** La query determina implicitamente quali valori considerare tramite pesi.
* **Rappresentazione di dimensione fissa:** Produce una rappresentazione di dimensione fissa da un insieme arbitrario di rappresentazioni (valori).
* **Condizionamento:** La rappresentazione finale è condizionata alla query.
* **Metafora:** La query "si concentra" sui valori, selezionando le informazioni più rilevanti.

**Applicazioni:**

* **Modelli seq2seq con cross-attention:** Ogni stato nascosto del decoder (query) si concentra su tutti gli stati nascosti dell'encoder (valori).
* **Ampia applicabilità:** Non limitata alle architetture seq2seq; utilizzabile in diversi modelli di deep learning per la manipolazione di puntatori e memoria.

---

# Risultato: Riassunto Selettivo

La somma ponderata fornisce un riassunto selettivo delle informazioni contenute in un insieme di valori.  Questo riassunto si concentra sulle parti più rilevanti, determinando l'importanza di ciascun valore in base a una query o a un criterio di ponderazione specifico.  In altre parole, la somma ponderata non considera tutti i valori in modo uniforme, ma assegna a ciascun valore un peso che riflette la sua importanza relativa nel contesto della query.

---

Per favore, forniscimi il testo da formattare.  Non ho ricevuto alcun testo da elaborare nell'input precedente.  Inserisci il testo che desideri formattare e io lo elaborerò seguendo le istruzioni fornite.

---
