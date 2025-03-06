
## Schema Riassuntivo: Reti Ricorrenti Stacked e Meccanismo di Attenzione

**1. Limiti delle Reti Ricorrenti (RNN)**

*   **1.1. Codifica Sequenziale:**
    *   RNN codificano l'input da sinistra a destra, in modo lineare.
    *   L'impatto di una parola è determinato dal suo contesto (parole adiacenti).
*   **1.2. Difficoltà con Sequenze Lunghe:**
    *   Richiedono molti step per codificare testi lunghi.
    *   **1.2.1. Problema di Parallelizzazione:**
        *   Passo forward e backward hanno complessità O(Lunghezza della sequenza) e non sono parallelizzabili.
        *   Gli stati nascosti futuri dipendono dagli stati nascosti passati.
        *   Limita l'addestramento su dataset di grandi dimensioni.
        *   Il numero minimo di step per calcolare lo stato al passo *t* dipende dalla lunghezza della sequenza.

**2. Meccanismo di Attenzione**

*   **2.1. Funzionamento:**
    *   Tratta la rappresentazione di ogni parola come una query.
    *   Accede e incorpora informazioni da un insieme di valori.
*   **2.2. Vantaggi:**
    *   Il numero di operazioni non parallelizzabili non aumenta con la lunghezza della sequenza.
    *   Distanza di interazione massima: O(1) (tutte le parole interagiscono ad ogni livello).
*   **2.3. Generalizzazione:**
    *   Può essere generalizzato con più layer.
    *   Richiede uno step di trasformazione perché ogni stato viene confrontato con ogni altro stato.
    *   Richiede un numero quadratico di confronti.

**3. Self-Attention**

*   **3.1. Concetto Chiave-Valore:**
    *   Simile a una ricerca "fuzzy" in un archivio chiave-valore.
    *   La query corrisponde a tutte le chiavi "softly", con un peso tra 0 e 1.
    *   I valori delle chiavi vengono moltiplicati per i pesi e sommati.
*   **3.2. Processo:**
    *   **3.2.1. Input:** Sequenza di parole $\mathbf{w}_{1:n}$ nel vocabolario $V$.
    *   **3.2.2. Embedding:** $\mathbf{x}_{i} = E \mathbf{w}_{i}$, dove $E \in \mathbb{R}^{d \times |V|}$ è la matrice di embedding.
    *   **3.2.3. Trasformazioni:**
        *   $\mathbf{q}_{i} = Q \mathbf{x}_{i}$ (queries)
        *   $\mathbf{k}_{i} = K \mathbf{x}_{i}$ (keys)
        *   $\mathbf{v}_{i} = V \mathbf{x}_{i}$ (values)
        *   Dove $Q, K, V \in \mathbb{R}^{d \times d}$ sono matrici di peso.
    *   **3.2.4. Calcolo Similarità:**
        *   $e_{i j} = \mathbf{q}_{i}^{\top} \mathbf{k}_{j}$
        *   $\alpha_{i j} = \frac{\exp\left(e_{i j}\right)}{\sum_{j^{\prime}} \exp\left(e_{i j^{\prime}}\right)}$
    *   **3.2.5. Output:**
        *   $\mathbf{o}_{i} = \sum_{j} \alpha_{i j} \mathbf{v}_{j}$
*   **3.3. Caratteristica:**
    *   Ogni stato "attende" a tutti gli altri (self-attention).
    *   Le similarità sono calcolate a coppie tra query e chiave.

---

# Schema Riassuntivo della Self-Attention e Tecniche di Addestramento

## 1. Self-Attention: Meccanismo Base
    *   La parola target (query) è confrontata con tutte le altre (chiavi).
    *   Probabilità ottenute con softmax.
    *   Somma pesata delle parole.

## 2. Barriere e Soluzioni della Self-Attention
    *   **2.1 Ordine delle Parole:**
        *   *Barriera:* La self-attention non tiene conto dell'ordine intrinseco delle parole.
        *   *Soluzione:* Aggiungere rappresentazioni posizionali agli input.
            *   Ogni indice di sequenza è un vettore $p_i \in \mathbb{R}^d$, per $i \in \{1,2,...,n\}$.
            *   Incorporazione posizionata: $\tilde{x}_i = x_i + p_i$
    *   **2.2 Non Linearità:**
        *   *Barriera:* La self-attention è solo una media pesata, manca di non linearità.
        *   *Soluzione:* Applicare una rete feedforward a ciascun output della self-attention.
    *   **2.3 Futuro (Mascheramento):**
        *   *Barriera:* In compiti di previsione, bisogna evitare di "guardare al futuro".
        *   *Soluzione:* Mascherare il futuro impostando i pesi dell'attenzione a 0.

## 3. Componenti della Self-Attention come Blocco Costruttivo
    *   **3.1 Rappresentazioni Posizionali:** Specificano l'ordine della sequenza.
    *   **3.2 Non Linearità:** Applicate all'output del blocco di self-attention tramite reti feed-forward.
    *   **3.3 Mascheramento:** Impedisce la "fuga" di informazioni dal futuro al passato.

## 4. Connessioni Residuali
    *   Tecnica per migliorare l'addestramento.
    *   Invece di: $X^{(i)} = \text{Layer}(X^{(i-1)})$
    *   Si usa: $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$
    *   Il gradiente è elevato attraverso la connessione residuale.
    *   Bias verso la funzione identità.

## 5. Normalizzazione per Layer (Layer Normalization)
    *   Tecnica per velocizzare l'addestramento.
    *   **Idea:** Ridurre la variazione normalizzando a media unitaria e deviazione standard.
    *   **Formulazione Matematica:**
        *   Sia $x \in \mathbb{R}^d$ un vettore.
        *   Media: $\mu = \frac{1}{d} \sum_{j=1}^d x_j$
        *   Deviazione Standard: $\sigma = \sqrt{\frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2}$
        *   Output: $\text{output} = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$
            *   $\gamma$ e $\beta$ sono parametri di "guadagno" e "bias" appresi.
            *   $\epsilon$ evita divisioni per zero.
    *   **Processo di Normalizzazione:**
        *   Normalizzazione tramite media e varianza scalari.
        *   Modulazione tramite guadagno e bias elementari appresi.
    *   **Successo:** Potrebbe essere dovuto alla normalizzazione dei gradienti.
    *   **Obiettivo:** Stabilizzare l'apprendimento all'interno di ogni layer.

---

**Schema Riassuntivo: Transformer e Attenzione Multi-Head**

**1. Standardizzazione e Pre-Normalizzazione**
    *   La standardizzazione di un vettore di dimensione *d* è un blocco "add & norm".
    *   La pre-normalizzazione, non la post-normalizzazione, rende efficiente l'addestramento.

**2. Sequence-Stacked Attention**
    *   **2.1 Definizione delle Variabili:**
        *   $X = [x_1; x_2; ...; x_n] \in \mathbb{R}^{n \times d}$: Concatenazione dei vettori di input.
            *   *n*: Numero di elementi nella sequenza.
            *   *d*: Dimensione di ogni vettore.
        *   $X_K \in \mathbb{R}^{n \times d}$, $X_Q \in \mathbb{R}^{n \times d}$, $X_V \in \mathbb{R}^{n \times d}$: Trasformazioni di X tramite matrici K, Q, V.
    *   **2.2 Calcolo dell'Output:**
        *   $\text{output} = \text{softmax}(XQ(XK)^T)XV \in \mathbb{R}^{n \times d}$
    *   **2.3 Passaggi del Calcolo:**
        *   Calcolo dei prodotti scalari query-key: $XQ(XK)^T$
        *   Applicazione della funzione softmax: $\text{softmax}(XQ(XK)^T)$
        *   Calcolo della media pesata: $\text{softmax}(XQ(XK)^T)XV$

**3. Transformer: Evoluzione delle RNN**
    *   Rete multi-layer e multi-head.
    *   Più teste = più matrici query-valore.
    *   Ogni testa lavora in parallelo senza impatto sul tempo di calcolo.
    *   Aggiungere complessità a ogni layer offre più interpretazioni delle relazioni tra le parole.
    *   Numero di teste (*h*) è un iperparametro per ogni layer.

**4. Ottimizzazione Computazionale con Multi-Head**
    *   Costo computazionale non aumenta significativamente con più teste.
    *   Calcolo e Riformattazione:
        *   $XQ \in \mathbb{R}^{n \times d}$ riformattato a $\mathbb{R}^{n \times h \times d / h}$ (simile per $XK$, $XV$).
        *   Trasposizione a $\mathbb{R}^{h \times n \times d / h}$: asse della testa diventa asse batch.

**5. Passaggi Dettagliati del Calcolo Multi-Head**
    *   **5.1 Calcolo dei Prodotti Scalari Query-Key:**
        *   $XQ(XK)^T$: Calcola tutti i set di coppie di punteggi di attenzione.
    *   **5.2 Calcolo della Media Ponderata:**
        *   $\text{softmax}\left(\frac{XQ(XK)^T}{\sqrt{d}}\right) XV$: Media ponderata con fattore di scala $\sqrt{d}$ per stabilizzare il gradiente.
    *   **5.3 Output:**
        *   Matrice di dimensioni $n \times d$.
    *   **5.4 Distribuzione della Multidimensionalità:**
        *   Ogni matrice di trasformazione di una testa ha forma $h \times d \times n$.

**6. Attenzione Multi-Head: Motivazione e Soluzione**
    *   **6.1 Problema:**
        *   Necessità di "guardare" in più punti della frase contemporaneamente.
        *   Concentrarsi su diversi *j* (parole) per motivi diversi.
    *   **6.2 Soluzione:**
        *   Definire più "teste" di attenzione tramite molteplici matrici Q, K, V.
        *   $Q_\ell, K_\ell, V_\ell \in \mathbb{R}^{d \times d/h}$, dove *ℓ* varia da 1 a *h*.

---

**Schema Riassuntivo del Testo sui Transformer**

**1. Multi-Head Attention**

*   Ogni testa di attenzione opera indipendentemente:
    *   $\text{output}_\ell = \text{softmax}(X Q_\ell K_\ell^T X^T) \times X V_\ell$, dove $\text{output}_\ell \in \mathbb{R}^{n \times d/h}$
*   Output delle teste combinati:
    *   $\text{output} = [\text{output}_1; ...;\text{output}_h]Y$, dove $Y \in \mathbb{R}^{d \times d}$
*   Ogni testa "guarda" aspetti diversi e costruisce vettori di valori in modo diverso.

**2. Scaled Dot Product Attention**

*   Scopo: Stabilizzare l'addestramento.
*   Problema: Prodotti scalari grandi con dimensionalità *d* elevata portano a gradienti piccoli.
*   Soluzione: Dividere gli score di attenzione per $\sqrt{d/h}$.
    *   Formula originale: $\text{output}_{e} = \text{softmax}\left(X Q_{e} K_{e}^{\top} X^{\top}\right) \times X V_{e}$
    *   Formula scalata: $\text{output}_{e} = \text{softmax}\left(\frac{X Q_{e} K_{e}^{\top} X^{\top}}{\sqrt{d/h}}\right) \times X V_{e}$

**3. Transformer Decoder**

*   Ottimizzazioni:
    *   Connessioni Residuali: Migliorano il flusso del gradiente.
    *   Normalizzazione per Layer: Stabilizza l'addestramento (media unitaria e deviazione standard unitaria).
    *   Spesso combinate come "Add & Norm".
*   Struttura del Blocco Decoder:
    *   Self-Attention (Masked): Focalizza l'attenzione sull'input, impedendo di "vedere" il futuro.
    *   Add & Norm: Connessione residuale e normalizzazione.
    *   Feed-Forward: Rete neurale feed-forward.
    *   Add & Norm: Connessione residuale e normalizzazione.
*   Funzionamento:
    *   Input nel primo blocco.
    *   Self-attention assegna pesi.
    *   Add & Norm.
    *   Feed-Forward.
    *   Add & Norm.
    *   Ripetizione per ogni blocco.
*   Mascheramento:
    *   Necessario per la predizione di parole.
    *   Non necessario per classificazione o analisi del sentiment.

**4. Transformer Encoder-Decoder**

*   Utilizzo: Traduzione automatica (seq2seq).
*   Encoder: Transformer standard (bidirezionale).
*   Decoder: Transformer modificato con cross-attention sull'output dell'Encoder (unidirezionale).
*   Tipi di Attenzione:
    *   Multi-head self-attention (Encoder).
    *   Masked multi-head self-attention (Decoder).
    *   Cross-attention (Decoder all'output dell'Encoder).
    *   Output della cross-attention passa attraverso un blocco Add & Norm.

**5. Cross-Attention**

*   Differenza dalla Self-Attention:
    *   Self-attention: Chiavi, query e valori dalla stessa sorgente.
    *   Cross-attention: Query dal decoder, chiavi e valori dall'encoder.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Attenzione in Transformers**

   *  **A. Vettori Encoder e Decoder:**
        *   $h_{1},\dots,h_n$: Vettori di output dell'encoder ($h_i \in \mathbb{R}^d$).
        *   $z_1, \ldots, z_n$: Vettori di input del decoder ($z_i \in \mathbb{R}^d$).

   *  **B. Chiavi, Valori e Query:**
        *   Chiavi ($k_i$) e Valori ($v_i$) derivati dall'output dell'encoder ($h_j$):
            *   $k_i = K h_j$
            *   $v_i = V h_j$
        *   Query ($q_i$) derivate dall'input del decoder ($z_i$):
            *   $q_i = Q z_i$

   *  **C. `h`:**
        *   Rappresenta la codifica finale dell'encoder.

**II. Tokenizzazione Subword**

   *  **A. Pre-processing:**
        *   Unica preoccupazione a livello di pre-processing per la preparazione dell'input.

---
