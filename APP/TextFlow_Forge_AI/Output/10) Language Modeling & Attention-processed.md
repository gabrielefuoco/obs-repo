
**I. Language Modeling**

   *   **Definizione:** Task autoregressivo per la generazione di testo.
   *   **Input:** Sequenza di parole osservate $x_1, ..., x_t$.
   *   **Obiettivo:** Predire la parola successiva $x_{t+1}$ dal vocabolario $V = \{w_1, \ldots, w_{|V|}\}$.
   *   **Formalizzazione:**
        *   Calcolo della probabilità condizionata: $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$
        *   Probabilità di una sequenza di testo T:
            $$ P(x^{(1)}, \ldots, x^{(T)}) = P(x^{(1)}) \times P(x^{(2)} \mid x^{(1)}) \times \cdots \times P(x^{(T)} \mid x^{(T-1)}, \ldots, x^{(1)}) $$
            $$ = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)}) $$
   *   **Applicazioni:**
        *   Machine Translation: Encoding nel linguaggio sorgente e decoding nel linguaggio target.
        *   Speech Recognition: Predizione di parole successive per la trascrizione.
        *   Spelling/Grammar Correction: Identificazione e correzione di errori.
        *   Summarization:
            *   Estrattiva: Evidenzia le frasi più importanti.
            *   Astrattiva: Rimodula il testo originale. (Considerabile come language modeling)

**II. N-gram Language Models**

   *   **Definizione:** Modelli che stimano la probabilità della parola successiva basandosi su sequenze di *n* token consecutivi.
   *   **Esempi:**
        *   Unigrammi: "il", "gatto", "sedeva", "sul"
        *   Bigrammi: "il gatto", "gatto sedeva", "sedeva sul"
        *   Trigrammi: "il gatto sedeva", "gatto sedeva sul"
        *   Four-grammi: "il gatto sedeva sul"
   *   **Assunzione di Markov:** $x^{(t+1)}$ dipende solo dalle *n-1* parole precedenti.
   *   **Calcolo della probabilità condizionata:**
        $$ P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)}) \approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})} $$
   *   **Esempio (modello 4-gram):**
        *   Testo: "gli studenti aprirono i"
        *   $P(w \mid \text{gli studenti aprirono}) = \frac{\text{count(gli studenti aprirono } w)}{\text{count(gli studenti aprirono)}}$
        *   Esempio numerico:
            *   "gli studenti aprirono i" (1000 volte)
            *   "gli studenti aprirono i libri" (400 volte)  => $P(\text{libri} \mid \text{gli studenti aprirono i}) = 0.4$
            *   "gli studenti aprirono i compiti" (100 volte) => $P(\text{compiti} \mid \text{gli studenti aprirono i}) = 0.1$
   *   **Gestione della sparsità:**
        *   **Problema 1: Numeratore = 0:** n-gram non presente nel corpus.
            *   **Soluzione (Smoothing):** Aggiungere un piccolo valore δ al conteggio di ogni parola.
        *   **Problema 2: Denominatore = 0:** (n-1)-gram non presente nel corpus.
            *   **Soluzione (Backoff):** Condizionare su un (n-1)-gram, (n-2)-gram, ecc.
   *   **Limiti:**
        *   Aumentare *n* peggiora la sparsità.
        *   Tipicamente, *n* non supera 5.
        *   Sparsità: Aumenta all'aumentare di *n*, causando una sottostima delle probabilità.

---

**Schema Riassuntivo: Language Modeling Neurale**

**1. Language Model Neurale (Window-Based)**

   *   **Obiettivo:** Predire la parola successiva data una sequenza di parole.
        *   Input: Sequenza di parole $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$
        *   Output: Distribuzione di probabilità $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$
   *   **Architettura:**
        *   Ogni parola è rappresentata da un embedding.
        *   Embeddings concatenati: $\mathbf{e} = [e^{(1)}, e^{(2)}, e^{(3)}, e^{(4)}]$
        *   Parole/vettori one-hot: $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \mathbf{x}^{(3)}, \mathbf{x}^{(4)}$
        *   Layer nascosto: $\mathbf{h} = f(\mathbf{W} \mathbf{e} + \mathbf{b}_1)$
            *   $\mathbf{h}$: vettore nascosto
            *   $f$: funzione di attivazione (es. tanh, ReLU)
            *   $\mathbf{W}$: matrice dei pesi del layer nascosto
            *   $\mathbf{e}$: vettore degli embeddings concatenati
            *   $\mathbf{b}_1$: bias del layer nascosto
        *   Distribuzione di output: $\mathbf{y} = \text{softmax}(\mathbf{U} \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{V}$
            *   $\mathbf{y}$: vettore di probabilità per ogni parola nel vocabolario
            *   $\mathbf{U}$: matrice di trasformazione
            *   $\mathbf{h}$: vettore nascosto
            *   $\mathbf{b}_2$: bias del layer di output
            *   $V$: dimensione del vocabolario

**2. Fixed-Window Neural Language Model: Miglioramenti e Problemi**

   *   **Miglioramenti rispetto ai modelli n-gram:**
        *   Nessun problema di sparsità.
        *   Non è necessario memorizzare tutti gli n-gram osservati.
   *   **Problemi:**
        *   Finestra di contesto fissa troppo piccola.
        *   Aumentare la dimensione della finestra aumenta esponenzialmente la dimensione della matrice dei pesi **W**.
        *   Nessuna finestra di dimensione fissa può essere sufficientemente grande per catturare tutte le dipendenze rilevanti nel linguaggio.
        *   Le parole $x^{(1)}$ e $x^{(2)}$ sono moltiplicate per pesi completamente diversi nella matrice **W**. Non c'è simmetria nel modo in cui gli input vengono elaborati.
        *   **Dimensione della matrice W:** Una finestra di contesto ampia implica una matrice di pesi **W** di grandi dimensioni, rendendo l'addestramento computazionalmente costoso e soggetto a overfitting.
        *   **Asimmetria nell'elaborazione dell'input:** Le parole nella finestra di contesto sono moltiplicate per pesi diversi nella matrice **W**.

**3. Reti Neurali Ricorrenti (RNN)**

   *   **Obiettivo:** Elaborare sequenze di lunghezza variabile.
   *   **Soluzione:** Condividere i pesi (**W**) per ogni parola nella sequenza di input.
   *   **Output:** Generato ad ogni *timestep* o solo all'ultimo, a seconda del *task*.

---

## Schema Riassuntivo: Reti Neurali Ricorrenti (RNN) e Modelli Linguistici

**1. Architettura RNN:**

*   **1.1. Principio Fondamentale:**
    *   La codifica del passo precedente influenza ogni *timestep* successivo.
    *   **W** contribuisce ad ogni passo.
*   **1.2. Input:**
    *   Sequenza di parole.
*   **1.3. Output:**
    *   Distribuzione di probabilità sulle parole del vocabolario.
        *   Formula:  $y^{(t)} = \text{softmax}(h^{(t)} + b_o) \in \mathbb{R}^{|\mathcal{V}|}$
            *   $y^{(t)}$: Distribuzione di probabilità sulle parole del vocabolario $\mathcal{V}$ al *timestep* $t$.
            *   $h^{(t)}$: Stato nascosto al *timestep* $t$.
            *   $b_o$: Bias dell'output.
*   **1.4. Stati Nascosti:**
    *   Aggiornamento iterativo basato sull'input corrente e lo stato precedente.
        *   Formula: $h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$
        *   Stato nascosto iniziale: $h^{(0)}$
            *   $h^{(t)}$: Stato nascosto al *timestep* $t$.
            *   $\mathbf{W}_{hh}$: Matrice dei pesi che connette lo stato nascosto precedente allo stato nascosto corrente.
            *   $\mathbf{W}_{xo}$: Matrice dei pesi che connette l'embedding della parola allo stato nascosto corrente.
            *   $\mathbf{e}^{(t)}$: Embedding della parola al *timestep* $t$.
            *   $\mathbf{b}_h$: Bias dello stato nascosto.
            *   $\sigma$: Funzione di attivazione (es. tanh o ReLU).
*   **1.5. Word Embeddings:**
    *   Rappresentazione vettoriale delle parole.
        *   Formula: $\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$
            *   $\mathbf{e}^{(t)}$: Embedding della parola al *timestep* $t$.
            *   $\mathbf{E}$: Matrice di embedding.
            *   $\mathbf{x}^{(t)}$: Vettore one-hot della parola al *timestep* $t$.
*   **1.6. Parole / Vettori One-hot:**
    *   Codifica delle parole come vettori binari.
        *   Formula: $\mathbf{x}^{(t)} \in \mathbb{R}^{|\mathcal{V}|}$
            *   $\mathbf{x}^{(t)}$: Vettore one-hot di dimensione $|\mathcal{V}|$ (dimensione del vocabolario).

**2. Vantaggi e Svantaggi:**

*   **2.1. Pro:**
    *   Simmetria dei pesi: I pesi vengono applicati ad ogni *timestep*.
    *   Dimensione del modello costante: Indipendente dalla lunghezza della sequenza.
*   **2.2. Contro:**
    *   Lunghezza della sequenza limitata: Difficoltà con sequenze lunghe (vanishing gradient problem).
    *   Tempo di addestramento: Richiede tempi lunghi.

**3. Addestramento di un Modello Linguistico RNN:**

*   **3.1. Dati di Addestramento:**
    *   Ampio corpus di testo con sequenze di lunghezza variabile.
*   **3.2. Predizione:**
    *   Ad ogni *timestep*, il modello predice la distribuzione di probabilità per la parola successiva.
*   **3.3. Funzione di Costo (Loss):**
    *   Cross-entropy ad ogni *timestep*.
        *   Formula: $J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w) = -\log \hat{y}^{(t)}_{x_{t+1}}$
            *   $y^{(t)}$: Vettore one-hot rappresentante la parola effettiva al passo (t+1).
            *   $\hat{y}^{(t)}$: Distribuzione di probabilità predetta dal modello al passo *t*.
            *   V: Vocabolario.
*   **3.4. Loss Totale:**
    *   Media delle loss calcolate ad ogni *timestep*.
*   **3.5. Considerazioni Computazionali:**
    *   Calcolare la loss e i gradienti sull'intero corpus contemporaneamente è computazionalmente costoso.

---

# Schema Riassuntivo: RNNs, BPTT, Perplexity e Vanishing Gradients

## 1. Backpropagation Through Time (BPTT)
    *   **Descrizione:** Variante della backpropagation usata per addestrare RNNs.
    *   **Obiettivo:** Calcolare i gradienti dei pesi rispetto alla funzione di costo.
    *   **Gradiente rispetto a pesi ripetuti:**
        *   Formula: $\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} }\mid_{(i)}$
        *   **Spiegazione:** Somma dei gradienti calcolati ad ogni timestep in cui il peso contribuisce.
    *   **Regola della Catena Multivariabile:**
        *   Formula: $\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$
        *   **Applicazione:** Usata per calcolare la derivata di funzioni composte ad ogni timestep.
    *   **Processo:** Propagazione all'indietro del gradiente cumulativo attraverso il tempo.

## 2. Valutazione del Modello: Perplexity
    *   **Definizione:** Misura di quanto bene un language model predice un campione di testo.
    *   **Formula:** $\text{Perplexity} = \left( \prod_{t=1}^{T} \frac{1}{P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})} \right)^{1/T}$
        *   $T$: Lunghezza della sequenza.
        *   $P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})$: Probabilità della parola successiva dato il contesto.
    *   **Interpretazione:** Inverso della probabilità geometrica media di predire correttamente le parole.
    *   **Valore:** Una perplexity inferiore indica un modello migliore.
    *   **Relazione con la Cross-Entropy:**
        *   Formula: $\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)}) \right) = \exp(J(\theta))$
        *   $J(\theta)$: Cross-entropy loss media.

## 3. Vanishing Gradient Problem
    *   **Descrizione:** I gradienti diventano esponenzialmente piccoli durante la BPTT, rendendo difficile l'addestramento su sequenze lunghe.
    *   **Causa:** Moltiplicazione ricorsiva di derivate che possono essere molto piccole.
    *   **Analisi con funzione identità:**
        *   Se $\sigma(x) = x$, allora $\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W_{h}$
    *   **Gradiente della loss rispetto allo stato nascosto precedente:**
        *   Formula: $\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{t=j+1}^{i} \frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell}$ dove $\ell = i - j$
    *   **Conseguenza:** Se gli autovalori di $W_{h}$ hanno modulo < 1, il termine $W_{h}^{\ell}$ diventa esponenzialmente piccolo all'aumentare di $\ell$.

---

Ecco uno schema riassuntivo del testo fornito, organizzato gerarchicamente:

**1. Definizione di  $h^{(t)}$ e Diagonalizzazione di $W_h$**

   *   Applicazione di una funzione di attivazione (es. tangente iperbolica, ReLU) alla combinazione lineare:
        *   Embedding dell'input al *timestep* `t`
        *   Bias
        *   Trasformazione dello stato nascosto al *timestep* precedente.
   *   La diagonalizzazione della derivata della funzione di attivazione per $W_h$ semplifica l'analisi del *vanishing gradient*.

**2. Il Problema del Vanishing Gradient**

   *   La derivata della funzione di costo J rispetto allo stato nascosto contiene il termine $W_h^\ell$, dove $\ell$ è la distanza temporale.
   *   **Causa:** Autovalori di $W_h$ minori di 1 in modulo: $\lambda_1, \lambda_2, \ldots, \lambda_n < 1$
        *   Autovettori: $\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n$
        *   Gradiente riscritto usando gli autovettori: $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(j)}} = \sum_{k=1}^n c_k \lambda_{k}^\ell \mathbf{q}_k$
        *   Per grandi $\ell$, $\lambda_k^\ell \rightarrow 0$, quindi il gradiente tende a 0.
   *   **Funzioni di attivazione non lineari:**
        *   Il problema persiste, ma richiede $|\lambda_i| < \gamma$ (dove $\gamma$ dipende dalla dimensionalità e dalla funzione di attivazione σ).
   *   Il *vanishing gradient* è causato dalla potenza $\ell$ di valori molto piccoli quando gli autovalori di $W_h$ sono minori di 1 in modulo.

**3. Il Problema dell'Exploding Gradient**

   *   Autovalori di $W_h$ maggiori di 1 in modulo.
   *   Aggiornamento SGD troppo grande: $\theta^{nuovo} = \theta^{vecchio} - \alpha \nabla_{\theta} J(\theta)$
   *   **Conseguenze:**
        *   Aggiornamenti errati e *loss* elevata.
        *   Valori **Inf** o **NaN** nella rete.

**4. Gradient Clipping**

   *   Se la norma del gradiente supera una soglia, viene ridimensionata.
   *   **Algoritmo 1 (Pseudo-codice):**
        *   **se** $\|\mathbf{g}\| >$ soglia **allora**
        *   $\mathbf{g} \leftarrow \frac{\text{soglia}}{\|\mathbf{g}\|} \mathbf{g}$
        *   **fine se**
   *   **Intuizione:** Passo nella stessa direzione, ma più piccolo.
   *   Gli *exploding gradient* sono più facili da risolvere rispetto al *vanishing gradient*.
   *   Si risolve con operazioni di normalizzazione (scaling, *clipping*).

**5. Risolvere il Vanishing Gradient Problem**

   *   Il problema nasce perché il segnale del gradiente proveniente da *timestep* lontani è molto più piccolo del segnale proveniente da *timestep* vicini.

---

**Schema Riassuntivo: RNN, LSTM e Gestione della Memoria a Lungo Termine**

1.  **Problemi delle RNN Standard**
    *   Aggiornamento dei pesi basato principalmente su effetti a breve termine.
    *   Difficoltà nel preservare informazioni su molti *timestep*.
    *   Stato nascosto costantemente riscritto:
        *   Formula:  $h^{(t)} = \sigma (W_{hh} h^{(t-1)} + W_{xz} x^{(t)} + b)$

2.  **Soluzioni per la Gestione della Memoria a Lungo Termine**
    *   Utilizzo di una memoria separata che viene aggiunta (approccio LSTM).
    *   Creazione di connessioni dirette e più lineari nel modello (es. Attention, Connessioni Residuali).
        *   Intervento architetturale: aggiornamento dello stato rispetto a un contesto più breve, mantenendo separatamente un buffer.

3.  **Long Short-Term Memory (LSTM)**
    *   Tipo di RNN progettato per risolvere il problema del *vanishing gradient*.
    *   Obiettivo: riprogettare le RNN dotandole di una "memoria" interna per migliorare la backpropagation.
    *   Introduzione della cella di memoria `c` per gestire le informazioni a lungo termine.
    *   Operazioni di lettura, scrittura e cancellazione controllate da *gate* specifici.
        *   I *gate* sono vettori che determinano quali informazioni devono essere gestite.
        *   Stato dei *gate* dinamico e variabile in base all'input e al contesto.

4.  **Calcolo degli Stati Nascosti e degli Stati delle Celle nelle LSTM**
    *   Data una sequenza di input $x^{(t)}$, l'obiettivo è calcolare $h^{(t)}$ e $c^{(t)}$.
    *   Operazioni principali al timestep $t$:
        *   **Forget Gate:** Determina quali informazioni dallo stato della cella precedente dimenticare.
            *   Formula: $f^{(t)} = \sigma \left( W_f h^{(t-1)} + U_f x^{(t)} + b_f \right)$
        *   **Gate di Input:** Determina quali parti del nuovo contenuto della cella scrivere nella cella.
            *   Formula: $i^{(t)} = \sigma \left( W_i h^{(t-1)} + U_i x^{(t)} + b_i \right)$
        *   **Gate di Output:** Determina quali parti della cella inviare allo stato nascosto.
            *   Formula: $o^{(t)} = \sigma \left( W_o h^{(t-1)} + U_o x^{(t)} + b_o \right)$
        *   **Nuovo Contenuto della Cella:** Rappresenta il nuovo contenuto potenziale da scrivere nella cella.
            *   Formula: $\tilde{c}^{(t)} = \tanh \left( W_c h^{(t-1)} + U_c x^{(t)} + b_c \right)$

---

**LSTM (Long Short-Term Memory)**

1.  **Aggiornamento dello Stato della Cella:**
    *   "Dimentica" informazioni dallo stato precedente e "memorizza" il nuovo contenuto.
    *   Formula: $c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)}$
        *   $c^{(t)}$: Stato della cella al timestep $t$
        *   $f^{(t)}$: Gate di dimenticanza
        *   $c^{(t-1)}$: Stato della cella precedente
        *   $i^{(t)}$: Gate di input
        *   $\tilde{c}^{(t)}$: Nuovo contenuto della cella
        *   $\circ$: Prodotto elemento-wise (Hadamard)
2.  **Generazione dello Stato Nascosto:**
    *   "Legge" informazioni dalla cella.
    *   Formula: $h^{(t)} = o^{(t)} \circ \tanh c^{(t)}$
        *   $h^{(t)}$: Stato nascosto al timestep $t$
        *   $o^{(t)}$: Gate di output
        *   $c^{(t)}$: Stato della cella
        *   $\circ$: Prodotto elemento-wise (Hadamard)
3.  **Funzionamento Interno:**
    *   Tutti i vettori hanno lunghezza $n$.
    *   I gate sono applicati tramite prodotto elemento-wise (Hadamard).
    *   $h^{(t)}$: Combinazione element-wise tra attivazione dello stato della cella (tanh) e $o^{(t)}$ (output gate).
    *   $c^{(t)}$: Combinazione tra $c^{(t-1)}$ e il nuovo contenuto, controllata da `f` (forget gate) e `i` (input gate).
    *   $\tilde{c}^{(t)}$: Risultato della trasformazione dell'input combinato con lo stato nascosto precedente.
    *   Ogni gate è una trasformazione non lineare (sigmoide) della combinazione lineare dell'input $x^{(t)}$ e dello stato nascosto $h^{(t-1)}$.
    *   Ogni gate ha parametri distinti.
4.  **Risoluzione del Vanishing Gradient:**
    *   Preservazione delle informazioni su molti timestep.
    *   Esempio: Se $f^{(t)} = 1$ e $i^{(t)} = 0$, l'informazione viene preservata indefinitamente.
    *   Miglioramento rispetto alle RNN standard (circa 100 timestep vs. 7).
    *   Connessioni dirette (skip connection/connessioni residuali):
        *   Gradiente passa direttamente senza trasformazioni.
        *   Input subisce trasformazione con funzione d'identità.
        *   Input si combina con l'output del layer.
    *   Connessioni dense (DenseNet): Ogni layer è connesso a ogni altro layer che lo segue.
    *   Highway connections (HighwayNet): Meccanismo di gating per determinare quale parte far passare direttamente.
5.  **Estensione con Bidirezionalità:**
    *   Necessità di contesto più ampio (es. analisi del sentiment).
    *   Due layer RNN paralleli: forward (sinistra a destra) e backward (destra a sinistra).
    *   Combinazione degli output delle due RNN.
    *   Non applicabile a task autoregressivi (es. language modeling).
    *   Migliora la rappresentazione contestuale per task non autoregressivi.
    *   $h^{(t)}$: Combinazione di $h^{(t)}_{forward}$ e $h^{(t)}_{backward}$, ciascuno con i propri parametri.

---

**I. RNN Bidirezionali**

*   Applicabili solo con accesso all'intera sequenza di input.
*   Non adatte al Language Modeling (solo contesto sinistro disponibile).
*   Potenti per l'encoding (uso predefinito).
    *   Esempio: BERT (Bidirectional Encoder Representations from Transformers).

**II. RNN Multi-Layer (Stacked RNN)**

*   Aumentano la profondità della rete aggiungendo layer sequenziali.
*   Permettono rappresentazioni più complesse.
    *   Layer inferiori: feature di basso livello (grammaticali, sintattiche).
    *   Layer superiori: feature di alto livello (semantiche).
*   Architettura: cattura relazioni di diverso ordine tra le parole (prossimità locale e globale).
*   Applicazioni:
    *   Language Modeling: testi grammaticalmente corretti e con maggiore coerenza linguistica.
*   Profondità tipica:
    *   Fino a 3 layer (massimo 4).
    *   Con skip connection: fino a 8 layer.
    *   BERT: 12 layer.
    *   Transformer (encoding): 24 layer.

**III. Traduzione Automatica (Machine Translation)**

*   Task complesso (input e output con ruoli diversi).
*   Richiede la modellazione di una relazione tra due linguaggi.
*   Importanza di dati di training comparabili per entrambe le lingue.
*   **Idea Centrale:** Apprendere un modello probabilistico dai dati.
    *   Obiettivo: trovare la migliore frase inglese `y` data la frase francese `x`: $\arg\max_y P(y|x)$.
    *   Scomposizione con la regola di Bayes: $= \arg\max_y P(x|y)P(y)$.
*   **Componenti:**
    *   **Modello di Traduzione:**
        *   Modella la fedeltà della traduzione.
        *   Appreso da dati paralleli (coppie di frasi tradotte).
    *   **Modello Linguistico:**
        *   Modella la fluidità della lingua target (inglese).
        *   Appreso da dati monolinguali (corpus di testo in inglese).
*   I due modelli non vengono addestrati separatamente.
    *   Il modello linguistico deve essere condizionato all'input.

---

**I. Modelli Seq2Seq per la Traduzione Automatica**

    A.  **Spazio di Rappresentazione Comune:**
        *   L'input viene codificato in uno spazio denso (embedding della frase).
        *   Questo embedding condiziona la generazione parola per parola della frase prodotta.
    B.  **Architettura Seq2Seq:**
        *   Utilizza due RNN (o LSTM multi-layer): encoder e decoder.
        *   **Encoder:**
            *   Codifica la frase di input (es. francese).
            *   Lo stato nascosto finale inizializza lo stato nascosto iniziale $h^{(0)}$ del decoder.
            *   Può essere bidirezionale per una migliore rappresentazione contestuale.
        *   **Decoder:**
            *   Ad ogni *timestep*, predice la prossima parola della frase di output (es. inglese).
            *   Il primo input è un token speciale "START".
            *   Predizione tramite cross-entropy, massimizzando la probabilità della parola corretta.
            *   Autoregressivo: la predizione dipende dalle parole generate precedentemente.
    C.  **Architettura Encoder-Decoder:**
        *   **Encoder:** Rete neurale che produce una rappresentazione vettoriale compatta (rappresentazione neurale) della sequenza di input.
        *   **Decoder:** Rete neurale che genera l'output basandosi sulla rappresentazione vettoriale dell'encoder.
        *   **Seq2Seq:** Modello in cui sia l'input che l'output sono sequenze.
    D.  **Applicazioni dei Modelli Seq2Seq:**
        *   Summarization (testo lungo → testo corto)
        *   Dialogo (turni precedenti → turno successivo)
        *   Parsing (testo in input → albero sintattico come sequenza)
        *   Generazione di codice (linguaggio naturale → codice Python)
    E.  **Modello Linguistico Condizionale:**
        *   **Modello linguistico:** il decoder predice la parola successiva della frase target `y`.
        *   **Condizionale:** le predizioni sono condizionate sulla frase sorgente `x`.
    F.  **Traduzione Automatica (NMT):**
        *   Calcola direttamente $P(y|x)$:
            $$ P(y|x) = P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \ldots P(y_T|y_1, \ldots, y_{T-1}, x) $$
        *   Probabilità della prossima parola target, date le parole target precedenti e la frase sorgente `x`.
        *   Addestramento su un grande corpus parallelo.
        *   Ricerca su "NMT non supervisionato", aumento dei dati, ecc.
    G.  **Addestramento del Modello Linguistico Condizionale:**
        *   Aggiornamento dei pesi di encoder e decoder ad ogni step di backpropagation.
        *   Convergenza lenta e complessa.
    H.  **Collo di Bottiglia del Condizionamento:**
        *   Il decoder è condizionato dall'output globale dell'encoder.
        *   Potrebbe essere meglio avere un riferimento non solo globale, ma anche relativo a ciascun elemento della frase di input ad ogni passo del decoding.

**II. Decoding Greedy**

    A.  **Strategia:**
        *   Seleziona, ad ogni passo del decoder, la parola con la probabilità più alta.
    B.  **Limitazioni:**
        *   Non considera l'impatto delle scelte precedenti sulle predizioni future.
    C.  **Obiettivo:**
        *   Massimizzare la probabilità condizionata della sequenza target `y` dato l'input `x`:
            $$P(y|x)=\prod_{t=1}^T P(y_{t}|y_{1},\dots,y_{t-1},x)$$
    D.  **Ricerca Esaustiva:**
        *   Idealmente, si cerca la traduzione `y` (di lunghezza T) che massimizza:
            $$ P(y|x) = P(y_1|x) \cdot P(y_2|y_1, x) \cdot P(y_3|y_1, y_2, x) \cdots P(y_T|y_1, \dots, y_{t-1}, x) $$
            $$ = \prod_{t=1}^{T} P(y_t|y_1, \dots, y_{t-1}, x) $$
        *   Complessità computazionale: $O(V^T)$, dove V è la dimensione del vocabolario e T è la lunghezza della sequenza.
        *   Impraticabile a causa della complessità.

**III. Beam Search**

    A.  **Motivazione:**
        *   Risolvere il problema della complessità computazionale della ricerca esaustiva.

---

**I. Beam Search**

*   **A. Concetto:** Mantenere le `k` traduzioni parziali più probabili (ipotesi) ad ogni *timestep* del decoder.
    *   `k` = dimensione del beam (tipicamente 5-10).
*   **B. Ipotesi:** Una frase candidata.
    *   `k` ipotesi ad ogni *timestep*.
*   **C. Score:** Logaritmo di probabilità di un'ipotesi.
    *   Formula: $\text{score}(y_1, \ldots, y_t) = \log P_{LM}(y_1, \ldots, y_t | x) = \sum_{i=1}^{t} \log P_{LM}(y_i | y_1, \ldots, y_{i-1}, x)$
    *   Score più alto = migliore.
*   **D. Obiettivo:** Trovare le ipotesi con gli score più alti.
*   **E. Limitazioni:** Non garantisce la soluzione ottimale, ma è più efficiente della ricerca esaustiva.

**II. BLEU (Bilingual Evaluation Understudy)**

*   **A. Scopo:** Valutare la qualità di una traduzione automatica confrontandola con traduzioni umane di riferimento (ground truth).
*   **B. Metodo:** Calcola un punteggio di corrispondenza basato su precisioni n-gram (1-gram a 4-gram) e una penalità per brevità.
*   **C. Precisione "clipped":**
    *   Confronta ogni parola della frase predetta con le frasi di riferimento.
    *   Limita il conteggio delle parole corrette al numero massimo di volte in cui la parola appare nella frase di riferimento.
*   **D. Geometric Average (Clipped) Precision Scores:**
    *   Formula: $\prod_{n=1}^{N} p_n^{w_n}$
*   **E. Brevity Penalty:**
    *   Formula:
        $$\begin{cases} 1, & \text{if } c > r \\ e^{(1-r/c)}, & \text{if } c <= r \end{cases}$$
*   **F. Punteggio BLEU:** Prodotto della precisione geometrica media e della penalità di brevità.
*   **G. Vantaggi:**
    *   Calcolo rapido e facile.
    *   Corrisponde alla valutazione umana.
    *   Indipendente dalla lingua.
    *   Utilizzabile con più frasi di riferimento.
*   **H. Svantaggi:**
    *   Non considera il significato delle parole.
    *   Ignora l'importanza delle parole e l'ordine.
    *   Esempio: "La guardia arrivò tardi a causa della pioggia" e "La pioggia arrivò tardi a causa della guardia" avrebbero lo stesso punteggio BLEU unigramma.
*   **I. Soluzioni:** Modelli multilingue sofisticati che codificano gli embedding delle frasi nel loro complesso.

**III. Attention Mechanism**

*   **A. Problema Risolto:** *Conditioning bottleneck* (decoder condizionato solo dall'output globale dell'encoder).
*   **B. Soluzione:** Permettere ad ogni *timestep* del decoder di essere guidato da una combinazione pesata delle rappresentazioni di ogni parola dell'input.

---

## Schema Riassuntivo: Attention Mechanism

**1. Introduzione all'Attention Mechanism**

*   Permette al decoder di focalizzarsi su parti specifiche dell'input (encoder) ad ogni *timestep*.
*   Crea connessioni dirette tra decoder ed encoder.
*   Il decoder "attende" all'encoder.

**2. Funzionamento dell'Attention Mechanism**

*   **Input:** Stati nascosti dell'encoder ($\boldsymbol{h}_1, \ldots, \boldsymbol{h}_N \in \mathbb{R}^d$) e stato nascosto del decoder al *timestep* `t` ($\boldsymbol{s}_t \in \mathbb{R}^d$).
*   **Calcolo degli Attention Scores:**
    *   $\boldsymbol{e}^t = [\boldsymbol{s}_t^T \boldsymbol{h}_1, \ldots, \boldsymbol{s}_t^T \boldsymbol{h}_N] \in \mathbb{R}^N$
*   **Distribuzione di Attenzione:**
    *   Applica la softmax agli attention scores per ottenere $\boldsymbol{\alpha}^t$:
        *   $\boldsymbol{\alpha}^t = \operatorname{softmax}(\boldsymbol{e}^t) \in \mathbb{R}^N$
*   **Output di Attenzione (Vettore di Contesto):**
    *   Calcola la somma pesata degli stati nascosti dell'encoder usando $\boldsymbol{\alpha}^t$:
        *   $\boldsymbol{a}_t = \sum_{i=1}^N \alpha_{i}^t \boldsymbol{h}_i \in \mathbb{R}^d$
*   **Integrazione nel Decoder:**
    *   Concatena l'output di attenzione $\boldsymbol{a}_t$ con lo stato nascosto del decoder $\boldsymbol{s}_t$:
        *   $[\boldsymbol{a}_t; \boldsymbol{s}_t] \in \mathbb{R}^{2d}$
    *   Utilizza il vettore concatenato per la generazione della parola successiva.

**3. Variazioni nel Calcolo degli Attention Scores ($\boldsymbol{e} \in \mathbb{R}^N$)**

*   **Attention a prodotto scalare:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{h}_i \in \mathbb{R}$ (assume $d_1 = d_2$)
*   **Attention moltiplicativa:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{W} \boldsymbol{h}_i \in \mathbb{R}$ (con matrice di pesi $\boldsymbol{W}$)
*   **Attention moltiplicativa a rango ridotto:** $\boldsymbol{e}_i = \boldsymbol{s}^T (\boldsymbol{U}^T \boldsymbol{V}) \boldsymbol{h}_i = (\boldsymbol{U} \boldsymbol{s})^T (\boldsymbol{V} \boldsymbol{h}_i)$ (con matrici di rango basso $\boldsymbol{U}$ e $\boldsymbol{V}$)
*   **Attention additiva:** $\boldsymbol{e}_i = \boldsymbol{v}^T \tanh(\boldsymbol{W}_1 \boldsymbol{h}_i + \boldsymbol{W}_2 \boldsymbol{s}) \in \mathbb{R}$ (con matrici di pesi $\boldsymbol{W}_1$ e $\boldsymbol{W}_2$ e vettore di pesi $\boldsymbol{v}$)

**4. Componenti Fondamentali dell'Attention Mechanism**

*   Calcolo degli *attention scores* ($\boldsymbol{e} \in \mathbb{R}^{N}$).
*   Applicazione della softmax per ottenere la distribuzione di attenzione ($\boldsymbol{\alpha} = \operatorname{softmax}(\boldsymbol{e}) \in \mathbb{R}^{N}$).
*   Calcolo della somma pesata dei valori usando la distribuzione di attenzione ($\boldsymbol{a} = \sum_{i=1}^{N} \alpha_{i} \boldsymbol{h}_{i} \in \mathbb{R}^{d_{1}}$), ottenendo l'**output di attenzione** $\boldsymbol{a}$.

**5. Benefici dell'Attention Mechanism nella NMT**

*   Migliora significativamente le prestazioni.
*   Permette al decoder di focalizzarsi su parti specifiche della frase sorgente.
*   Fornisce un modello più "umano" del processo di traduzione.
*   Risolve il problema del *bottleneck*.
*   Aiuta a mitigare il problema del *vanishing gradient*.
*   Fornisce interpretabilità.

**6. Generalizzazione dell'Attention**

*   Tecnica generale di deep learning, applicabile a diverse architetture e task.
*   **Definizione:** Data una serie di vettori (*valori*) e un vettore *query*, l'attenzione calcola una somma ponderata dei valori, dipendente dalla query.
*   La somma ponderata rappresenta un riassunto selettivo delle informazioni contenute nei valori.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Attenzione (Attention): Concetto Chiave**

    *   Definizione: Meccanismo per ottenere una rappresentazione di dimensione fissa da un insieme di rappresentazioni (valori), condizionata da una query.
    *   Funzione: Calcola una somma ponderata di valori, condizionata a una query.

**II. Ruolo della Query e dei Valori**

    *   Query: Determina su quali valori "concentrarsi" (pesi).
    *   Valori: Insieme di rappresentazioni da cui estrarre informazioni.

**III. Applicazioni e Flessibilità**

    *   Potenza e Flessibilità: Meccanismo potente e flessibile per la manipolazione di puntatori e memoria nel deep learning.
    *   Esempio: Modello seq2seq con cross-attention (decoder query si concentra su encoder valori).
    *   Generalizzazione: Non limitata alle architetture seq2seq.

**IV. Risultato dell'Attenzione**

    *   Riassunto Selettivo: La somma ponderata rappresenta un riassunto selettivo delle informazioni contenute nei valori.

---
