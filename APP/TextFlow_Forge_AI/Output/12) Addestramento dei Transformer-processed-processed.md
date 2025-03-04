
# Preprocessing nei Language Model

## Semplificazione della Pipeline

L'obiettivo del preprocessing nei Language Model è semplificare la pipeline, evitando task di riduzione della dimensionalità spinta sulla sintassi.  Si concentra su fasi cruciali di trattamento del testo grezzo.

## Pre-tokenizzazione

Questa fase prepara il dato grezzo per la tokenizzazione vera e propria.  Include:

* **Distinzione parole/delimitatori:** Identifica le parole e i delimitatori (punteggiatura, spazi).
* **Gestione della case delle lettere:** Decide se mantenere o meno la distinzione tra maiuscole e minuscole.


## Tokenizzazione

La tokenizzazione suddivide il testo in unità elementari (token).

### Apprendimento del Vocabolario

Il tokenizzatore apprende un vocabolario di *sub-word token*.

### Strategia di Sub-wording

La strategia di sub-wording viene appresa ottimizzando una funzione obiettivo specifica.

### Composizione del Vocabolario

Il vocabolario include caratteri, simboli non alfabetici, parole intere e parti di parole.

### Inferenza

Durante l'inferenza, le parole vengono suddivise in sequenze di token noti. Un token può coincidere con una parola intera.

### Gestione degli Spazi

La gestione degli spazi è variabile e la reversibilità non è sempre garantita. I metodi possono essere basati su sequenze di parole, caratteri o byte.


## Principali Approcci

### Byte-Pair Encoding (BPE)

* **Addestramento:** Inizia con parole uniche e caratteri ASCII nel corpus.
* **Merging Iterativo:** Unisce iterativamente le coppie di token più frequenti.
* **Byte-level Encoding (variante):** Tratta i simboli come byte (es. GPT, BERT).
* **Gestione Spazi ed Emoji:** Utilizza token speciali per gli spazi e rappresenta le emoji con caratteri Unicode.

### WordPiece

* **Pattern Matching Iterativo:** Crea token più lunghi a partire da quelli più frequenti.
* **Prefisso "##":** Indica lo split di un termine che appare come singola parola nel corpus.
* **Minimizzazione dei Token:** Tende a minimizzare il numero di token generati.
* **Gestione Spazi:** Non preserva gli spazi.

### Differenze tra BPE e WordPiece

* **Criterio di Selezione:** WordPiece usa uno score (frequenza coppia / frequenza singoli token), dando priorità a coppie con parti individuali meno frequenti. BPE si basa solo sulla frequenza della coppia.
* **Lunghezza Sub-word:** WordPiece tende a produrre sub-word più lunghe; usa un token "unknown" per parole non scomponibili. BPE usa "unknown" solo per caratteri singoli non presenti.


---

# Schema Riassuntivo: Unigram, Model Pretraining e Paradigma Pretrain-Finetune

## I. Unigram Tokenizzazione

* **Approccio:** Algoritmo probabilistico inverso rispetto a BPE e WordPiece. Inizia con un vocabolario ampio e rimuove iterativamente i simboli meno informativi.
* **Fase di training:** Rimuove simboli che minimizzano l'incremento della *likelihood loss*.
* **Fase di inferenza:** Tutti i simboli vengono splittati.
* **Criterio di split:** Massimizza la *negative log likelihood*.
* **Gestione spazi:** Non preserva spazi multipli, ma cattura simboli speciali (emoji).


## II. Model Pretraining

* **Evoluzione da Word2Vec:** Word2Vec genera *word embedding* context-free. I Transformer generalizzano questo concetto, incorporando il contesto.
* **Caratteristiche:**
    * Quasi tutti i parametri NLP sono inizializzati tramite pre-addestramento.
    * I modelli vengono addestrati a ricostruire parti nascoste dell'input.
    * Genera:
        * Rappresentazioni linguistiche robuste.
        * Inizializzazioni per modelli NLP performanti.
        * Distribuzioni di probabilità sul linguaggio.
* **Output:** Ogni token ha diversi embedding intermedi (a seconda dei livelli dell'architettura), dipendenti dal contesto.
* **Applicazioni:** Ricostruzione sintassi, semantica lessicale, sentiment analysis (predizione token successivo o completamento frasi).


## III. Paradigma Pretrain-Finetune

* **Modelli Linguistici:** Sono *word-models* con conoscenza del mondo dipendente da *model size* e dati di addestramento.
* **Utilizzo modelli pre-addestrati:** Crea modelli di *Narrow AI* specializzati tramite *fine-tuning*.
* **Obiettivo:** Raggiungere l'AGI (Intelligenza Artificiale Generale) prima della singolarità tecnologica.
* **Architettura chiave:** Transformer, utilizzati per generare modelli pre-addestrati.
* **Compiti di pre-addestramento:** Includono *Language Modeling* per i decoder.


---

## Pre-addestramento degli Encoder

### Masked Language Modeling (MLM)

* Sostituzione di una frazione di parole con "[MASK]".
* Predizione delle parole mascherate.
* Apprendimento di $p_\theta(x|\tilde{x})$, dove $\tilde{x}$ è la versione mascherata di $x$.
* $h_1, ..., h_T = \text{Encoder}(w_1, ..., w_T)$ (rappresentazione dell'encoder)
* $y_i \sim Aw_i + b$ (output)
* Perdita calcolata solo sulle parole mascherate.
* Analogo a un compito di "filling-in mask".

---

# BERT (Bidirectional Encoder Representations from Transformers)

BERT utilizza il *Masked Language Modeling* (MLM) per creare una rappresentazione bidirezionale profonda del testo.  Il 15% dei token viene mascherato casualmente: l'80% viene sostituito con "[MASK]", il 10% con token casuali e il 10% rimane invariato.  Un ulteriore task di pre-training è il *Next Sentence Prediction* (NSP), che determina se due sequenze sono consecutive nel documento originale.

## Tokenizzazione e Rappresentazione dell'Input (BERT)

* **Tokenizzazione:** Utilizza WordPiece (vocabolario di 30.000 token).
* **Lunghezza degli embeddings:** 768.
* **Rappresentazione dell'input:** Somma di *positional embeddings* e *segment embeddings*.
* **Token speciali:** `[CLS]` (rappresentazione dell'intera sequenza) e `[SEP]` (separatore di frasi).

## Applicazione di BERT a Diversi Task

* **Classificazione del testo:** Aggiunta di una testa di classificazione all'ultimo layer.  L'embedding `[CLS]` viene utilizzato come input per la classificazione, oppure si può utilizzare l' *average pooling* su tutti gli embedding di output.
* **Varianti di BERT:** Esistono varianti con 12 layer (768 dimensioni) e 24 layer (1024 dimensioni).
* **Specializzazione:** Gli embedding dell'ultimo livello vengono utilizzati per il *fine-tuning*.
* **Gerarchia di apprendimento:** I layer più vicini all'input apprendono relazioni a corto raggio, mentre quelli più vicini all'output apprendono relazioni semantiche (a lungo raggio).


# Schema Riassuntivo: BERT e le sue Estensioni

**I. Task a Livello di Sequenza con BERT**

* Aggiunta di un layer specifico per il task sopra il modello BERT, seguito da *fine-tuning*.
* Utilizzo dell'embedding `[CLS]` come input per task di classificazione (es. entailment).
* Utilizzo degli embedding dei token come input per task a livello di token (es. *sequence tagging*, *question answering*).

**II. BERT come Feature Extractor**

* Gli embedding di BERT (soprattutto degli ultimi 4 layer) migliorano le prestazioni di modelli come BiLSTM.

**III. Conoscenza Codificata in BERT**

* **Informazione sull'ordine delle parole:** Layer inferiori.
* **Informazione sintattica:** Layer intermedi.
* **Informazione specifica del task:** Layer superiori.
* **Informazione semantica:** Distribuita su tutti i layer.

**IV. Fine-tuning di BERT e Task**

* *Fine-tuning* di successo su diversi task con risultati allo stato dell'arte.
* Esempi di Task: QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE.

**V. Estensioni di BERT**

* Miglioramenti nel pre-addestramento: RoBERTa e SpanBERT.


# RoBERTa

**Differenze principali da BERT:**

* Solo MLM come task di pre-training (rimozione di NSP).
* Maggiori dati di addestramento (CC-News, Stories, OpenWebText).
* Addestramento molto più lungo (MLM dinamico, sequenze più lunghe).
* Tokenizzazione BPE con 50.000 sub-word.
* Due dimensioni del modello: RoBERTa-base e RoBERTa-large.
* **Risultati:** Prestazioni superiori a BERT su GLUE, SQuAD v2.0 e RACE. Dimostrazione dell'importanza di maggiore potenza di calcolo e dati nel pre-training.


# SpanBERT

**Differenza principale da BERT:** Maschera sequenze contigue di parole durante il pre-addestramento.


# Schema Riassuntivo: SpanBERT e S-BERT

**I. SpanBERT**

* **A. Nuovo Approccio di Pre-addestramento:**
    1. Mascheramento di span adiacenti di lunghezza variabile (distribuzione geometrica sbilanciata).
    2. Mascheramento di parole complete.
    3. Selezione casuale del punto di partenza dello span.
* **B. Obiettivo di Confine dello Span (SBO):**
    1. Predizione di ogni token nello span usando solo i token ai confini.
    2. Utilizzo delle codifiche di output dei confini e dell'embedding di posizione relativa del token target.
    3. Minimizzazione della perdita di cross-entropia (MLM + SBO).

**II. S-BERT: BERT per la Similarità di Frasi**

* **A. Sentence Transformer:**
    1. Migliora la codifica di intere frasi rispetto all'utilizzo di embedding `[CLS]` o di output di BERT.
    2. Utilizza *cross-encoding* (coppie di frasi come input).
    3. Aggiunge una testa di classificazione per calcolare lo score di similarità.
    4.  ![[]]

---

# Appunti su Modelli Linguistici e Apprendimento

## I. Further Pretraining e Apprendimento Contrastivo

**Further Pretraining**, a differenza del fine-tuning, si concentra sull'ulteriore addestramento del modello senza adattarlo a un compito specifico.  Un esempio importante è il **B. Similarity Learning (Apprendimento Contrastivo):**

1.  Impara a identificare similarità tra frasi semanticamente simili.
2.  Impara a distinguere frasi semanticamente differenti.


## II. Estensione di BERT per il Similarity Learning

**A. Approcci Principali:**

1.  **Cross-encoding:** Un singolo modello BERT riceve in input una coppia di frasi.
2.  **Rete Siamese:** Due istanze dello stesso modello BERT (condivisione dei pesi), una per ogni frase.

**B. Processo della Rete Siamese:**

1.  Generazione di embedding (es. average pooling).
2.  Aggiunta di residui (opzionale).
3.  Feed-forward.
4.  Calcolo della similarità (softmax o loss contrastiva).

**C. Obiettivo:** Ottenere *sentence embeddings* che catturino la similarità semantica.


## III. Funzioni Obiettivo per Apprendimento Similarità Semantica

*   **Triplet Loss:** Confronta un'istanza target con una positiva (simile) e una negativa (dissimile) per distinguere similarità e dissimilarità.
*   **Multiple Negative Ranking Loss:** Impone che la similarità tra istanze positive superi quella tra istanze negative di un certo margine. Margini più piccoli aumentano la robustezza del modello.


## IV. Fine-tuning di Modelli Pre-addestrati

*   **Full Fine-tuning:** Fine-tuning di tutti i parametri del modello. Richiede meno dati e converge più velocemente rispetto all'addestramento *from scratch*.
*   **Parameter-Efficient Fine-tuning:** Approcci più efficienti dal punto di vista computazionale e dei parametri.  Esempi includono:
    *   **Lightweight Fine-tuning:** Congela alcuni pesi del modello pre-addestrato e aggiunge nuovi layer.
    *   **Adapted Tuning:** Aggiunge *adapter* (moduli aggiuntivi) sopra il modello pre-addestrato; solo i pesi degli *adapter* vengono aggiornati.
    *   **Prefix-Tuning e Prompt Tuning:** Non modificano l'architettura; agiscono sull'input aggiungendo token speciali o parametri di prefisso. Vantaggio: ogni elemento di un batch può usare un modello finetuned in modo diverso.


## V. Pre-training di Encoder-Decoder

**Approccio:** Generalizza il masked language modeling. Utilizza tecniche di *denoising* e *span corruption*.

**Meccanismo:** Maschera *span* di testo (non solo token singoli). L'obiettivo è la generazione di testo, ripristinando gli *span* mascherati.

**Token Sentinella:** Ogni *span* mascherato è sostituito con un token sentinella unico. Il target è associare ogni token sentinella al suo *bound* (testo corretto).

**Formulazione Matematica:**

$$h_{1},\dots,h_{r}=\text{Encoder}(w_{1},\dots,w_{r})$$
$$h_{r+1},\dots,h_{r}=\text{Decoder}(w_{1},\dots,w_{t}),h_{1},\dots,h_{r}$$
$$y_{i}\approx Ah_{i}+b,i>T$$

L'encoder beneficia di un contesto bidirezionale, il decoder allena l'intero modello.


## VI. Modelli Linguistici di Grandi Dimensioni: T5, GPT, GPT-2, GPT-3

### I. T5 (Text-to-Text Transfer Transformer)

*   Architettura: Encoder-Decoder
*   Addestramento: Instruction Training (task multipli formulati come problemi testo-a-testo)
*   Gestione token sentinella: Trattato come sequenza di testo, non con masked language modelling.

### II. GPT (Generative Pre-trained Transformer)

*   Architettura: Decoder-only Transformer (12 layer, 117 milioni di parametri, stati nascosti a 768 dimensioni, layer feed-forward nascosti a 3072 dimensioni)
*   Tokenizzazione: Byte-Pair Encoding (40.000 merges)
*   Dataset: BooksCorpus (>7000 libri)
*   Valutazione: Benchmark NLI (Natural Language Inference) per il task di entailment (Entailment, Contraddittorio, Neutrale)
*   Input: Sequenza di token con token di inizio frase e token di inizio/fine frase per il decoder.

### III. GPT-2

*   Architettura: Decoder Transformer
*   Dataset: 40 GB di testo
*   Parametri: 1.5 miliardi
*   Capacità emergente: Zero-shot learning

### IV. Zero-shot Learning

*   Definizione: Esecuzione di task senza esempi positivi durante l'addestramento. Nessun aggiornamento del gradiente durante l'inferenza.
*   Vantaggi:
    *   Utile per problemi text-to-text (es. risposta a domande).
    *   Alternativa al fine-tuning in caso di:
        *   Limitazione dei dati.
        *   Difficoltà nella definizione del contesto.
*   Implementazione: Modifica del prompt per guidare la generazione.

### V. GPT-3

*   Parametri: 175 miliardi
*   Dati: >600 GB
*   Capacità emergente: Few-shot learning (In-context learning)

### VI. Few-shot Learning (In-context Learning)

*   Definizione: Fornire esempi nel prompt per guidare il modello.


---

## Efficacia dell'apprendimento

L'efficacia dell'apprendimento mostra un miglioramento delle prestazioni all'aumentare del numero di esempi utilizzati.  Tuttavia, si osserva un rendimento decrescente dopo 3-4 esempi.  Questo suggerisce che, oltre un certo numero di esempi, l'incremento di prestazioni ottenuto è marginale.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
