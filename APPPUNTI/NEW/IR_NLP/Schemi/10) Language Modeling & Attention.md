
##### Language Modeling

* **Definizione:** Task autoregressivo per la generazione di testo.
* Input: Sequenza di parole osservate $x_1, ..., x_t$.
* Output: Predizione della parola successiva $x_{t+1}$.
* Vocabolario: Noto a priori.
* Distribuzione di probabilità: $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$
* Probabilità di generare un testo T: $P(x^{(1)}, \ldots, x^{(T)}) = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)})$

* **Applicazioni:**
	* Machine Translation (encoding/decoding)
	* Speech Recognition
	* Spelling/Grammar Correction
	* Summarization (estrattiva e astrattiva - quest'ultima come caso particolare di language modeling)

##### N-gram Language Models

* **Definizione:** Stima la probabilità della parola successiva contando le occorrenze di n-gram in un corpus.
* n-gram: Sequenza di *n* token consecutivi.
* Esempi: Unigrammi, Bigrammi, Trigrammi, Four-grammi.
* Assunzione di Markov: $x^{(t+1)}$ dipende solo dalle *n-1* parole precedenti.
* Probabilità condizionata: $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)}) \approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})}$

* **Esempio (modello 4-gram):**
	* Testo: "gli studenti aprirono i"
	* $P(w \mid \text{gli studenti aprirono}) = \frac{\text{count(gli studenti aprirono } w)}{\text{count(gli studenti aprirono)}}$
	* Esempio numerico: $P(\text{libri} \mid \text{gli studenti aprirono i}) = 0.4$, $P(\text{compiti} \mid \text{gli studenti aprirono i}) = 0.1$

* **Gestione della sparsità:**
	* **Problema 1 (Numeratore = 0):** Smoothing (aggiunta di δ al conteggio)
	* **Problema 2 (Denominatore = 0):** Backoff (condizionamento su n-gram più corti)

* **Limiti:**
	* Aumentare *n* peggiora la sparsità.
	* *n* tipicamente non supera 5.
	* Alta sparsità con *n* elevato, causando sottostima delle probabilità.

##### Modello Linguistico Neurale a Finestra Fissa

* **Input e Output:**
	* Input: Sequenza di parole $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$
	* Output: Distribuzione di probabilità della parola successiva $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$
* **Rappresentazione delle Parole:**
	* Embeddings: Ogni parola rappresentata da un embedding.
	* Concatenazione: Gli embeddings vengono concatenati in un vettore $\mathbf{e} = [e^{(1)}, e^{(2)}, e^{(3)}, e^{(4)}]$.
* **Architettura del Modello:**
	* Layer Nascosto: $\mathbf{h} = f(\mathbf{W} \mathbf{e} + \mathbf{b}_1)$ (f = funzione di attivazione, W = matrice di pesi, b<sub>1</sub> = bias)
	* Distribuzione di Output: $\mathbf{y} = \text{softmax}(\mathbf{U} \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{V}$ (U = matrice di trasformazione, b<sub>2</sub> = bias, V = dimensione del vocabolario)
* **Vantaggi rispetto ai modelli n-gram:**
	* Nessun problema di sparsità.
	* Non richiede la memorizzazione di tutti gli n-gram osservati.
* **Limiti:**
	* Finestra di contesto fissa troppo piccola.
	* Aumento esponenziale della dimensione di **W** con l'aumento della finestra.
	* Impossibilità di catturare tutte le dipendenze a lungo termine.
	* Asimmetria nell'elaborazione dell'input (pesi diversi per parole diverse nella finestra).
	* Granularità: diminuisce all'aumentare di *n*, riducendo la capacità di catturare dipendenze a lungo termine, portando a probabilità poco informative e testo incoerente.

##### Reti Neurali Ricorrenti (RNN)

* **Obiettivo:** Condividere i pesi (**W**) per ogni parola nella sequenza di input, permettendo l'elaborazione di sequenze di lunghezza variabile con una rete di dimensione fissa.
* **Output:** Generato ad ogni timestep o solo all'ultimo, a seconda del task.

##### Confronto tra Modelli

* Modelli a finestra fissa: limitati dalla dimensione fissa della finestra, che impedisce la cattura di dipendenze a lungo termine e introduce asimmetrie nell'elaborazione.
* RNN: superano i limiti dei modelli a finestra fissa grazie alla condivisione dei pesi, permettendo l'elaborazione di sequenze di lunghezza variabile.

### Reti Neurali Ricorrenti (RNN) per Modelli Linguistici

##### Architettura RNN

**Input:** Sequenza di parole, rappresentate da vettori one-hot $\mathbf{x}^{(t)} \in \mathbb{R}^{|\mathcal{V}|}$.
**Word Embeddings:** $\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$ trasforma i vettori one-hot in rappresentazioni vettoriali dense.
**Stati Nascosti:** $h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$ aggiorna lo stato nascosto in base allo stato precedente e all'embedding corrente. $h^{(0)}$ è lo stato iniziale.
**Distribuzione di Output:** $y^{(t)} = \text{softmax}(h^{(t)} + b_o) \in \mathbb{R}^{|\mathcal{V}|}$ produce una distribuzione di probabilità sulle parole del vocabolario.

##### Addestramento

**Funzione di Costo:** Cross-entropy ad ogni timestep: $J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w) = -\log \hat{y}^{(t)}_{x_{t+1}}$.
**Loss Totale:** Media delle loss su tutti i timestep dell'intero training set. Calcolo computazionalmente costoso se effettuato su tutto il corpus contemporaneamente.

##### Pro:

* Simmetria dei pesi: stessi pesi applicati ad ogni timestep.
* Dimensione del modello costante: indipendente dalla lunghezza della sequenza.
##### Contro:

* Lunghezza della sequenza limitata: Problema del vanishing gradient per sequenze lunghe.
* Tempo di addestramento lungo.

##### Backpropagation Through Time (BPTT)

* **Scopo:** Addestramento di Recurrent Neural Networks (RNNs).
* **Metodo:** Variante dell'algoritmo di backpropagation per calcolare i gradienti dei pesi rispetto alla funzione di costo.
* **Calcolo del gradiente per pesi ripetuti (es. $\boldsymbol{W}_{h}$):**
	* $\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} }\mid_{(i)}$
	* Somma dei gradienti calcolati ad ogni *timestep* in cui il peso contribuisce.
	* Applicazione della regola della catena multivariabile.
* **Regola della catena multivariabile:**
	* Per $f(x,y)$, $x(t)$, $y(t)$: $\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$
	* Applicata iterativamente per calcolare i gradienti attraverso i *timestep*.
	* Propagazione all'indietro dei gradienti attraverso il tempo.

##### Valutazione del Modello

* **Metrica:** Perplexity
* **Formula:** $\text{Perplexity} = \left( \prod_{t=1}^{T} \frac{1}{P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})} \right)^{1/T}$
* **Interpretazione:** Inverso della probabilità geometrica media di predire correttamente le parole.
* **Relazione con la cross-entropy:** $\text{Perplexity} = \exp(J(\theta))$, dove $J(\theta)$ è la cross-entropy loss media.
* **Miglior modello:** Perplexity inferiore.

##### Vanishing Gradient Problem

* **Causa:** Probabilità molto piccole durante il calcolo ricorsivo delle derivate in BPTT.
* **Effetto:** Gradienti che diminuiscono esponenzialmente procedendo indietro nel tempo, rendendo difficile l'addestramento su sequenze lunghe.
* **Esemplificazione con σ(x) = x:**
	* $\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W_{h}$
* **Analisi del gradiente:**
	* $\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell}$, dove $\ell = i - j$.
	* **Condizione per il vanishing gradient:** Autovalori di $W_{h}$ con modulo minore di 1. Il termine $W_{h}^{\ell}$ diventa esponenzialmente piccolo all'aumentare di $\ell$.

##### Definizione di h<sup>(t)</sup> e Vanishing Gradient

* **h<sup>(t)</sup>:** Applicazione di una funzione di attivazione non lineare (es. tanh, ReLU) alla combinazione lineare di:
* Embedding dell'input al timestep `t`
* Bias
* Trasformazione dello stato nascosto al timestep precedente
* **Vanishing Gradient:** Derivata della funzione di costo J rispetto allo stato nascosto a passi precedenti contiene $W_h^\ell$ (dove ℓ è la distanza temporale). Se gli autovalori di $W_h$ sono minori di 1 in modulo ($|\lambda_i| < 1$), $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(j)}} = \sum_{k=1}^n c_k \lambda_{k}^\ell \mathbf{q}_k$ tende a 0 per grandi ℓ, causando il *vanishing gradient*.
* Con funzioni di attivazione non lineari, la condizione diventa $|\lambda_i| < \gamma$ (γ dipende da dimensionalità e funzione di attivazione).

##### Exploding Gradient

* **Causa:** Autovalori di $W_h$ maggiori di 1 in modulo.
* **Effetto:** Aggiornamenti SGD troppo grandi ($\theta^{nuovo} = \theta^{vecchio} - \alpha \nabla_{\theta} J(\theta)$), portando a:
* Aggiornamenti errati
* *Loss* elevata
* Potenziali valori Inf o NaN nella rete.
* **Soluzione:** *Gradient Clipping*
* Algoritmo: Se $\|\mathbf{g}\| >$ soglia, allora $\mathbf{g} \leftarrow \frac{\text{soglia}}{\|\mathbf{g}\|} \mathbf{g}$
* Intuizione: Ridimensionamento del gradiente senza cambiarne la direzione.

##### Risolvere il Vanishing Gradient

* **Problema:** Il segnale del gradiente da timestep lontani è molto più piccolo di quello da timestep vicini. (Nota: la soluzione al vanishing gradient non è esplicitamente dettagliata nel testo oltre alla discussione sulla natura del problema).

##### Limiti delle RNN Standard

* **Problema principale:** Difficoltà nel preservare informazioni su molti *timestep*.
* **Causa:** Riscrittura costante dello stato nascosto: $$h^{(t)} = \sigma (W_{hh} h^{(t-1)} + W_{xz} x^{(t)} + b)$$
* **Conseguenza:** Aggiornamento dei pesi basato principalmente su effetti a breve termine, trascurando quelli a lungo termine.

##### Soluzioni al Problema del *Vanishing Gradient*

* **Approccio 1: Memoria Separata**
	* Utilizzo di una memoria aggiuntiva per gestire informazioni a lungo termine.
	* Esempio: LSTM (Long Short-Term Memory).
* **Approccio 2: Connessioni Dirette e Lineari**
	* Creazione di connessioni più dirette e lineari nel modello.
	* Esempi: Meccanismi di *attention*, connessioni residuali.
	* Necessità di un intervento architetturale per aggiornare lo stato rispetto a un contesto più breve, mantenendo un buffer separato per il contesto precedente.

##### Long Short-Term Memory (LSTM)

* **Obiettivo:** Risolvere il problema del *vanishing gradient* nelle RNN aggiungendo una "memoria" interna.
* **Componente chiave:** Cella di memoria (`c`) per gestire informazioni a lungo termine.
* **Meccanismi di controllo:** *Gate* (Forget, Input, Output) che regolano la lettura, scrittura e cancellazione di informazioni nella cella di memoria. Lo stato dei *gate* è dinamico e varia in base all'input e al contesto.

##### Calcolo degli Stati in LSTM

* **Input:** Sequenza di input $x^{(t)}$.
* **Output:** Sequenza di stati nascosti $h^{(t)}$ e stati delle celle $c^{(t)}$.
* **Operazioni al timestep *t*:**
	* **Forget Gate:** $f^{(t)} = \sigma \left( W_f h^{(t-1)} + U_f x^{(t)} + b_f \right)$ (determina quali informazioni dallo stato della cella precedente mantenere)
	* **Input Gate:** $i^{(t)} = \sigma \left( W_i h^{(t-1)} + U_i x^{(t)} + b_i \right)$ (determina quali parti del nuovo contenuto scrivere nella cella)
	* **Output Gate:** $o^{(t)} = \sigma \left( W_o h^{(t-1)} + U_o x^{(t)} + b_o \right)$ (determina quali parti della cella inviare allo stato nascosto)
	* **Nuovo Contenuto della Cella:** $\tilde{c}^{(t)} = \tanh \left( W_c h^{(t-1)} + U_c x^{(t)} + b_c \right)$ (nuovo contenuto potenziale da scrivere nella cella)

##### Meccanismo di funzionamento delle LSTM:

**Stato della Cella ($c^{(t)}$):** Aggiorna lo stato memorizzando nuove informazioni e dimenticando quelle obsolete.
* Formula: $c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)}$
* $c^{(t)}$: stato della cella al timestep t
* $f^{(t)}$: gate di dimenticanza
* $c^{(t-1)}$: stato della cella al timestep precedente
* $i^{(t)}$: gate di input
* $\tilde{c}^{(t)}$: nuovo contenuto della cella
* $\circ$: prodotto elemento-wise (Hadamard)

**Stato Nascosto ($h^{(t)}$):** Genera lo stato nascosto leggendo informazioni dallo stato della cella.
* Formula: $h^{(t)} = o^{(t)} \circ \tanh(c^{(t)})$
* $h^{(t)}$: stato nascosto al timestep t
* $o^{(t)}$: gate di output
* $\tanh$: funzione tangente iperbolica

**Gate:** Ogni gate ($f^{(t)}$, $i^{(t)}$, $o^{(t)}$) è ottenuto tramite una trasformazione non lineare (sigmoide) di una combinazione lineare dell'input $x^{(t)}$ e dello stato nascosto $h^{(t-1)}$. Ogni gate ha parametri distinti. Tutti i vettori hanno lunghezza n.

##### Risoluzione del *Vanishing Gradient*:

**Preservazione delle informazioni:** L'architettura LSTM facilita la preservazione delle informazioni su molti timestep grazie ai gate. Es: $f^{(t)}=1$ e $i^{(t)}=0$ preservano indefinitamente l'informazione.
**Confronto con RNN standard:** Le LSTM raggiungono circa 100 timestep contro i circa 7 delle RNN standard.
**Skip Connections/Connessioni Residuali:** Alternative per preservare dipendenze a lungo raggio, permettendo al gradiente di passare direttamente tra timestep, evitando il *vanishing gradient*.
* **Connessioni dense (DenseNet):** ogni layer connesso a ogni altro layer successivo.
* **Highway connections (HighwayNet):** meccanismo di gating per determinare la parte di informazione da passare direttamente.

##### Estensione Bidirezionale:

**Problema:** L'ambiguità in frasi richiede un contesto più ampio (es. "terribly").
**Soluzione:** Due layer RNN paralleli (forward e backward) elaborano l'input in direzioni opposte.
**Combinazione:** Lo stato nascosto $h^{(t)}$ è la combinazione di $h^{(t)}_{forward}$ e $h^{(t)}_{backward}$.
**Applicabilità:** La bidirezionalità è applicabile a task non autoregressivi, migliorando la rappresentazione contestuale. Non direttamente applicabile a task autoregressivi (es. language modeling).

##### Reti Neurali Ricorrenti (RNN)

* **Bidirezionalità:**
	* Applicabile solo con accesso all'intera sequenza di input.
	* Non adatta al Language Modeling (solo contesto sinistro disponibile).
	* Molto potente per encoding (es. BERT).

* **Multi-Layer RNN:**
	* Aggiunta di più layer per rappresentazioni più complesse.
	* RNN inferiori: feature di basso livello.
	* RNN superiori: feature di alto livello.
	* Chiamate anche RNN stacked.
	* Cattura relazioni di diverso ordine (locale e globale).
	* Migliora la capacità di catturare proprietà grammaticali e sintattiche (layer inferiori) e relazioni semantiche (layer superiori).
	* Tipicamente 1-3 layer; fino a 8 con skip connection, 12 in BERT, 24 in Transformer specifici per encoding.

##### Traduzione Automatica (Machine Translation)

* **Sfida:** Task complesso fino al 2015, input e output con ruoli diversi, modellazione di relazione tra due linguaggi (simile alla summarization).
* **Apprendimento Probabilistico:**
	* Trovare la migliore frase inglese `y` data una frase francese `x`: $\arg\max_y P(y|x)$
	* Utilizzo della regola di Bayes: $\arg\max_y P(x|y)P(y)$
* **Componenti del Modello:**
	* **Modello di Traduzione:** Apprende la traduzione di parole e frasi (fedeltà). Addestrato su dati paralleli (coppie di frasi tradotte).
	* **Modello Linguistico:** Modella la correttezza grammaticale in inglese (fluidità). Addestrato su dati monolinguali (corpus di testo in inglese).
	* I due modelli non sono addestrati separatamente; il modello linguistico è condizionato all'input.

##### Modello Seq2Seq:

* Utilizza due RNN (o LSTM) multi-layer:
* **Encoder:** Codifica la frase di input in una rappresentazione vettoriale. Lo stato nascosto finale ($h^{(0)}$) inizializza il decoder.
* **Decoder:** Genera la frase di output parola per parola, condizionata dall'encoder. Inizia con un token "START" e usa un approccio autoregressivo.
* Predizione tramite cross-entropy, massimizzando $P(y|x)$.
* L'encoder può essere bidirezionale per un contesto migliore.

##### Architettura Encoder-Decoder:

* **Encoder:** Crea una rappresentazione vettoriale compatta dell'input.
* **Decoder:** Genera l'output dalla rappresentazione dell'encoder.
* **Applicazioni:** Traduzione automatica, summarization, dialogo, parsing, generazione di codice.
* **Modello Linguistico Condizionale:**
	* Modello linguistico: predice la prossima parola (`y`).
	* Condizionale: le predizioni dipendono dalla frase sorgente (`x`).
	* **Traduzione Automatica (NMT):** Calcola direttamente $P(y|x) = P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \ldots P(y_T|y_1, \ldots, y_{T-1}, x)$.
	* **Addestramento:** Backpropagation sui pesi dell'encoder e del decoder (convergenza lenta).
	* **Collo di Bottiglia:** Il decoder è condizionato solo dall'output globale dell'encoder.

##### Strategie di Decoding:

* **Decoding Greedy:** Seleziona la parola con la probabilità più alta ad ogni passo. Non ottimale a lungo termine.
* **Ricerca Esaustiva:** Massimizza $P(y|x) = \prod_{t=1}^T P(y_{t}|y_{1},\dots,y_{t-1},x)$, ma ha complessità computazionale $O(V^T)$ (impraticabile).
* **Beam Search:** Soluzione per la complessità computazionale della ricerca esaustiva. (Dettagli non specificati nel testo).

##### Beam Search

* **Scopo:** Trovare le traduzioni più probabili in modo efficiente.
* Mantiene le `k` ipotesi (traduzioni parziali) più probabili ad ogni *timestep*.
* `k` è la dimensione del beam (tipicamente 5-10).
* **Score delle ipotesi:** $\text{score}(y_1, \ldots, y_t) = \log P_{LM}(y_1, \ldots, y_t | x) = \sum_{i=1}^{t} \log P_{LM}(y_i | y_1, \ldots, y_{i-1}, x)$
* Score più alto indica maggiore probabilità (score negativi).
* **Efficienza:** Non garantisce la soluzione ottimale, ma è molto più efficiente della ricerca esaustiva.

##### BLEU (Bilingual Evaluation Understudy)

* **Scopo:** Valutare la qualità di una traduzione automatica.
* Confronta l'output del traduttore con traduzioni umane di riferimento ("ground truth").
* Si basa su precisioni n-gram (da 1-gram a 4-gram).
* Include una penalità di brevità.
* **Precisione "clipped":**
	* Conta le corrispondenze parola per parola, limitando il conteggio al numero massimo di occorrenze nella frase di riferimento.
	* **Geometric Average (Clipped) Precision Scores:** $\prod_{n=1}^{N} p_n^{w_n}$
	* **Brevity Penalty:** $\begin{cases} 1, & \text{if } c > r \\ e^{(1-r/c)}, & \text{if } c <= r \end{cases}$ (c = lunghezza della traduzione, r = lunghezza della frase di riferimento)
	* **Punteggio BLEU:** Prodotto della precisione geometrica media e della penalità di brevità.
	* **Vantaggi:** Calcolo rapido, intuitivo, indipendente dalla lingua, gestibile con più frasi di riferimento.
	* **Svantaggi:** Ignora significato, importanza e ordine delle parole (es. "La guardia arrivò tardi a causa della pioggia" e "La pioggia arrivò tardi a causa della guardia" potrebbero avere lo stesso punteggio unigramma).

##### Attention Mechanism

* **Scopo:** Risolvere il *conditioning bottleneck* dei modelli encoder-decoder.
* Invece di basarsi solo sull'output finale dell'encoder, permette ad ogni *timestep* del decoder di accedere a una combinazione pesata delle rappresentazioni di ogni parola dell'input.
* **Vantaggio:** Evita la perdita di informazioni dovuta al condizionamento solo sull'output globale dell'encoder.

##### Meccanismo di Attention in Modelli Seq2Seq

* **Obiettivi:** Creare connessioni dirette tra decoder ed encoder, permettendo al decoder di focalizzarsi su parti specifiche dell'input ad ogni timestep.
* Il decoder "attende" all'encoder.
* Partendo da "START", genera parole basandosi su un riassunto ponderato delle informazioni dell'encoder.

* **Calcolo dell'Attention:**
	* **Attention Scores:** $\boldsymbol{e}^t = [\boldsymbol{s}_t^T \boldsymbol{h}_1, \ldots, \boldsymbol{s}_t^T \boldsymbol{h}_N] \in \mathbb{R}^N$ (dove $\boldsymbol{h}_i$ sono gli stati nascosti dell'encoder e $\boldsymbol{s}_t$ lo stato nascosto del decoder al timestep *t*).
	* **Distribuzione di Attenzione:** $\boldsymbol{\alpha}^t = \operatorname{softmax}(\boldsymbol{e}^t) \in \mathbb{R}^N$ (distribuzione di probabilità che somma a 1).
	* **Output di Attenzione (Vettore di Contesto):** $\boldsymbol{a}_t = \sum_{i=1}^N \alpha_{i}^t \boldsymbol{h}_i \in \mathbb{R}^d$ (somma pesata degli stati nascosti dell'encoder).
	* **Concatenazione:** $[\boldsymbol{a}_t; \boldsymbol{s}_t] \in \mathbb{R}^{2d}$ (concatenazione dell'output di attenzione con lo stato nascosto del decoder).

* **Vantaggi dell'Attention in NMT:**
	* Migliora le prestazioni.
	* Focalizzazione su parti specifiche della frase sorgente.
	* Modello più "umano".
	* Risoluzione del *bottleneck*.
	* Mitigazione del *vanishing gradient*.
	* Maggiore interpretabilità.

* **Tipi di Attention:**
	* Attention a prodotto scalare: $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{h}_i \in \mathbb{R}$
	* Attention moltiplicativa: $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{W} \boldsymbol{h}_i \in \mathbb{R}$
	* Attention moltiplicativa a rango ridotto: $\boldsymbol{e}_i = \boldsymbol{s}^T (\boldsymbol{U}^T \boldsymbol{V}) \boldsymbol{h}_i = (\boldsymbol{U} \boldsymbol{s})^T (\boldsymbol{V} \boldsymbol{h}_i)$
	* Attention additiva: $\boldsymbol{e}_i = \boldsymbol{v}^T \tanh(\boldsymbol{W}_1 \boldsymbol{h}_i + \boldsymbol{W}_2 \boldsymbol{s}) \in \mathbb{R}$

* **Fasi Generali dell'Attention:**
	* Calcolo degli *attention scores* ($\boldsymbol{e}$).
	* Applicazione della softmax per ottenere $\boldsymbol{\alpha}$.
	* Calcolo della somma pesata dei valori usando $\boldsymbol{\alpha}$, ottenendo l'output di attenzione ($\boldsymbol{a}$).

##### Attention come Tecnica Generale di Deep Learning

* **Definizione:** Calcolo di una somma ponderata di vettori (valori) in base a un vettore *query*.
* La somma ponderata rappresenta un riassunto selettivo delle informazioni nei valori.
* Applicabile a diverse architetture e task.

##### Meccanismo di Attention:

* **Funzione principale:** Calcolo di una somma ponderata di valori, condizionata da una query.

* **Selezione dei valori:** La query determina implicitamente quali valori considerare tramite pesi.
* **Rappresentazione di dimensione fissa:** Produce una rappresentazione di dimensione fissa da un insieme arbitrario di rappresentazioni (valori).
* **Condizionamento:** La rappresentazione finale è condizionata alla query.
* **Metafora:** La query "si concentra" sui valori, selezionando le informazioni più rilevanti.

* **Applicazioni:**

* **Modelli seq2seq con cross-attention:** Ogni stato nascosto del decoder (query) si concentra su tutti gli stati nascosti dell'encoder (valori).
* **Ampia applicabilità:** Non limitata alle architetture seq2seq; utilizzabile in diversi modelli di deep learning per la manipolazione di puntatori e memoria.
* **Riassunto selettivo:** La somma ponderata rappresenta un riassunto delle informazioni contenute nei valori, focalizzato sulle parti più rilevanti in base alla query.

