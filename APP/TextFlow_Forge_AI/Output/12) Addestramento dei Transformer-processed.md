
## Schema Riassuntivo: Preprocessing nei Language Model

**1. Preprocessing Semplificato:**

*   Non richiede riduzione della dimensionalità spinta sulla sintassi.

**2. Pre-tokenizzazione:**

*   Tratta il dato grezzo.
*   Distingue parole e delimitatori naturali (punteggiatura e spazi).
*   Gestisce il case delle lettere.

**3. Tokenizzazione:**

*   Modulo cruciale per la preparazione dei dati.
*   Apprende un vocabolario di *sub-word token* (token).
*   Non identifica a priori le parole, ma le apprende.
*   Apprende una strategia di sub-wording.
*   Ottimizza una funzione obiettivo.
*   Il vocabolario include: caratteri, simboli non alfabetici, parole intere e parti di parole.
*   A tempo di inferenza, ogni parola è splittata in una sequenza di token noti.
*   Gestisce gli spazi (con token speciali) - valutare se reversibile.

**4. Approcci Principali:**

*   Utilizzati da tutti i Language Model.

    *   **4.1 Byte-Pair Encoding (BPE):**

        *   Addestramento:
            *   Trova le parole uniche nel corpus (token candidati).
            *   Aggiunge i caratteri ASCII.
            *   Effettua un merging iterativo a partire dalle unità (simboli).
            *   Apprende regole di merge: unisce la coppia di token più frequente ad ogni step.
        *   Variante: **byte-level encoding** (GPT, BERT) - tratta i simboli come byte.
        *   Preserva gli spazi (token speciali).
        *   Codifica le emoji (caratteri unicode).

    *   **4.2 WordPiece:**

        *   Pattern matching iterativo guidato dalla frequenza dei token più piccoli.
        *   Utilizza il prefisso "##" per indicare lo split di un termine.
        *   Tende a minimizzare il numero di token generati.
        *   Non preserva gli spazi.

        *   **4.2.1 Differenze con BPE:**

            *   **Criterio di selezione:**
                *   BPE: coppia di token più frequente.
                *   WordPiece: calcola lo score: frequenza di occorrenza della coppia / frequenza di occorrenza dei singoli token.
                *   WordPiece dà priorità alle coppie le cui parti individuali sono meno frequenti nel vocabolario.
            *   **Lunghezza dei sub-word:**
                *   WordPiece: tende a identificare sub-word più lunghe e decide lo split su queste.
                *   WordPiece: usa token "unknown" per parole non splittabili.
                *   BPE: classificherebbe come "unknown" solo i singoli caratteri non presenti nel vocabolario.

---

## Schema Riassuntivo: Unigram e Pre-addestramento

### 1. Unigram: Tokenizzazione Probabilistica

*   **1.1. Approccio:**
    *   Inverso a BPE e WordPiece: parte da un ampio vocabolario.
    *   Effettua split ripetuti durante il training, ma splitta sempre in inferenza.
    *   Algoritmo di tokenizzazione probabilistico.
*   **1.2. Minimizzazione della Loss:**
    *   Obiettivo: minimizzare la *likelihood loss* del corpus.
    *   Ad ogni passo, calcola la *likelihood loss* rispetto al vocabolario corrente.
    *   Calcola la variazione della loss rimuovendo ogni simbolo.
    *   Rimuove i simboli che portano al minor incremento della loss (meno informativi).
*   **1.3. Tokenizzazione:**
    *   Considera tutti i possibili split di una parola.
    *   Sceglie lo split con la massima *negative log likelihood*.
*   **1.4. Caratteristiche:**
    *   Non preserva gli spazi multipli.
    *   Cattura simboli particolari (emoji).
*   **1.5. Confronto con BPE e WordPiece:**
    *   Vedi tabella nel testo originale.

### 2. Model Pretraining

*   **2.1. Word2Vec:**
    *   Prima forma di apprendimento di rappresentazioni testuali.
    *   Genera modelli di word embedding pre-addestrati.
    *   Modelli *context-free*: il significato di una parola non dipende dal contesto.
*   **2.2. Transformer:**
    *   Generalizza il concetto di Word2Vec.
    *   Incorpora il contesto della parola per generare testo.
*   **2.3. NLP Moderno:**
    *   Quasi tutti i parametri sono inizializzati tramite pre-addestramento.
    *   Metodi di pre-addestramento nascondono parti dell'input.
    *   Addestra il modello a ricostruire tali parti.
    *   Efficace per:
        *   Rappresentazioni robuste del linguaggio.
        *   Inizializzazioni dei parametri per modelli NLP performanti.
        *   Distribuzioni di probabilità sul linguaggio da cui campionare.
*   **2.4. Pre-addestramento Contestualizzato:**
    *   Rappresentazione del linguaggio.
    *   Inizializzazione di un modello per task specifici (es. classificazione).
    *   Ogni token ha tanti embedding intermedi quanti sono i livelli dell'architettura.
    *   Le versioni di embedding dipendono dal contesto.
*   **2.5. Task Risolvibili con Modelli Pre-addestrati:**
    *   Ricostruzione della sintassi.
    *   Semantica lessicale.
    *   Sentiment analysis.
    *   Predizione del token successivo.
    *   Completamento di frasi.

### 3. Paradigma Pretrain-Finetune

*   **3.1. Word-Models:**
    *   Modelli linguistici con conoscenza del mondo.
    *   La conoscenza dipende dalla dimensione del modello (*model size*) e dalla quantità di dati.
*   **3.2. Narrow AI:**
    *   Modelli pre-addestrati specializzati tramite *fine-tuning*.
*   **3.3. Obiettivo Finale:**
    *   AGI (Intelligenza Artificiale Generale).
    *   Singolarità tecnologica: IA supera le capacità intellettive umane.
*   **3.4. Transformer come Architettura:**
    *   Genera modelli pre-addestrati.
    *   Utilizzati come punto di partenza per modelli specializzati.
*   **3.5. Compiti di Pre-addestramento:**
    *   Language Modeling (per i decoder).

---

## Schema Riassuntivo: Pre-addestramento di Encoder e BERT

### 1. Pre-addestramento degli Encoder
    *   **1.1. Limitazioni del Language Modeling:**
        *   Gli encoder utilizzano contesto bidirezionale, rendendo il language modeling tradizionale inadatto.
    *   **1.2. Masked Language Modeling (MLM):**
        *   **1.2.1. Concetto:** Sostituire una frazione di parole con token `[MASK]` e predire le parole mascherate.
        *   **1.2.2. Processo:**
            *   `h_1, ..., h_T = Encoder(w_1, ..., w_T)`
            *   `y_i ~ Aw_i + b`
        *   **1.2.3. Funzione di Perdita:** Calcolata solo sulle parole mascherate. Apprendimento di `p_θ(x|x̃)`, dove `x̃` è la versione mascherata di `x`.
        *   **1.2.4. Obiettivo:** Rendere il Transformer capace di completare una frase (filling-in mask).
        *   **1.2.5. Valutazione:** Capacità di ricostruire le parole mascherate.

### 2. BERT: Bidirectional Encoder Representations from Transformers
    *   **2.1. Caratteristiche Principali:**
        *   Rappresentazione bidirezionale profonda di testo non etichettato tramite MLM.
        *   Condizionamento congiunto delle parole di contesto a sinistra e a destra in tutti gli strati dell'encoder.
    *   **2.2. Mascheramento Casuale:**
        *   **2.2.1. Percentuale:** Mascheramento del 15% dei token (sub-word).
        *   **2.2.2. Strategie:**
            *   Sostituzione con `[MASK]` nell'80% dei casi.
            *   Sostituzione con token casuale nel 10% dei casi (introduzione di rumore).
            *   Nessuna modifica nel 10% dei casi (riconoscimento dell'assenza di rumore).
        *   **2.2.3. Motivazione:** Prevenire la "compiacenza" e favorire rappresentazioni robuste.
    *   **2.3. Compiti di Pre-addestramento:**
        *   **2.3.1. Masked Language Modeling (MLM):** Predire i token mascherati e il rumore.
        *   **2.3.2. Next Sentence Prediction (NSP):** Determinare se la seconda sequenza segue la prima nel documento originale.

### 3. Tokenizzazione e Rappresentazione dell'Input
    *   **3.1. Tokenizzazione:** Utilizzo di WordPiece (vocabolario di 30.000 token predefiniti in BERT base).
    *   **3.2. Dimensione Embedding:** 768 in BERT base.
    *   **3.3. Rappresentazione dell'Input:** Somma di positional embeddings e segment embeddings.
        *   **3.3.1. Segment Embeddings:** Posizione del token nella sequenza (stesso encoding per token nella stessa frase).
        *   **3.3.2. Position Embeddings:** Posizione assoluta del token nel testo.
    *   **3.4. Token Speciali:**
        *   **3.4.1. `[CLS]`:** Primo token della sequenza, rappresentativo dell'intera sequenza di input.
        *   **3.4.2. `[SEP]`:** Separa due frasi in input.

### 4. Applicazione di BERT a diversi task
    *   **4.1. Classificazione del Testo:**
        *   Aggiunta di una "testa di classificazione" specifica per il task all'ultimo layer.
        *   Utilizzo dell'embedding `[CLS]` come input per la testa di classificazione.
        *   Alternativa: average pooling su tutti gli embedding di output.
    *   **4.2. Varianti di BERT:**
        *   12 layer (dimensionalità 764).
        *   24 layer (dimensionalità 1024).
    *   **4.3. Apprendimento Gerarchico:**
        *   Layer vicini all'input: relazioni a corto raggio.
        *   Layer vicini all'output: relazioni semantiche (a lungo raggio).
    *   **4.4. Specializzazione:** Partire dagli embedding dell'ultimo livello (12° o 24°).

---

## Schema Riassuntivo di BERT e sue Estensioni

### 1. Utilizzo di BERT per Task Specifici

*   **1.1 Task a Livello di Sequenza:**
    *   Aggiunta di un layer specifico per il task e fine-tuning.
    *   Utilizzo dell'embedding \[CLS] per classificazione e entailment.
    *   Utilizzo degli embedding dei token per sequence tagging e risposta alle domande.
*   **1.2 Utilizzo degli Embedding di BERT in Modelli Basati su Features:**
    *   Embedding di BERT come input per altri modelli (es. BiLSTM).
    *   Miglioramento delle prestazioni combinando output di diversi layer di BERT (ultimi quattro).

### 2. Conoscenza Codificata nelle Rappresentazioni Ponderate di BERT

*   **2.1 Informazione sull'Ordine delle Parole:** Presente nei layer inferiori.
*   **2.2 Informazione Sintattica:** Presente nei layer intermedi.
*   **2.3 Informazione Specifica del Task:** Presente negli ultimi layer.
*   **2.4 Informazione Semantica:** Distribuita in tutto il modello.

### 3. Fine-tuning di BERT

*   **3.1 Successo e Versatilità:** Fine-tuning porta a risultati allo stato dell'arte.
*   **3.2 Esempi di Task:**
    *   QQP (Quora Question Pairs): Rileva parafrasi.
    *   QNLI (Question Natural Language Inference): Inferenza linguistica naturale su domande e risposte.
    *   SST-2 (Stanford Sentiment Treebank): Analisi del sentiment.
    *   CoLA (Corpus of Linguistic Acceptability): Rileva correttezza grammaticale.
    *   STS-B (Semantic Textual Similarity Benchmark): Similarità testuale semantica.
    *   MRPC (Microsoft Paraphrase Corpus): Corpus di parafrasi.
    *   RTE (Recognizing Textual Entailment): Inferenza linguistica naturale.

### 4. Estensioni di BERT

*   **4.1 Miglioramenti Comuni:**
    *   RoBERTa: Addestramento più lungo, rimozione di NSP.
    *   SpanBERT: Maschera sequenze contigue di parole.

### 5. RoBERTa: Dettagli

*   **5.1 Differenza Principale:** Utilizzo esclusivo di **masked language modelling (MLM)**.
*   **5.2 Miglioramenti:**
    *   **Maggiori Dati di Addestramento:** BookCorpus, Wikipedia inglese, CC-News (76 GB), Stories (31 GB), OpenWebText (38 GB).
    *   **Addestramento Molto Più Lungo:** Diminuzione step, aumento batch, rimozione NSP, MLM dinamico.
    *   **Sequenze di Pre-addestramento:** Lunghezza di 512 token.
    *   **Tokenizzazione BPE:** 50.000 sub-word.
    *   **Due Dimensioni del Modello:**
        *   RoBERTa-base (12 layer, 12 teste di attenzione, 768 dimensioni nascoste)
        *   RoBERTa-large (24 layer, 16 teste di attenzione, 1024 dimensioni nascoste)
    *   **Prestazioni Superiori a BERT:** Su GLUE, SQuAD v2.0 e RACE.
*   **5.3 Conclusioni:** Maggiore potenza di calcolo e dati migliorano il pre-addestramento.

---

**Schema Riassuntivo: SpanBERT e S-BERT**

**1. SpanBERT: Generalizzazione del Masking in BERT**

*   **1.1. Nuovo Approccio di Pre-Addestramento:**
    *   Schema di mascheramento, obiettivo di addestramento e procedura di addestramento delle sequenze differenti rispetto a BERT.
*   **1.2. Mascheramento di Span Adiacenti:**
    *   Mascheramento di span adiacenti casuali invece di singoli token.
    *   Lunghezza dello span: distribuzione geometrica sbilanciata (favorisce lunghezze brevi).
    *   Punto di partenza del mascheramento: selezione casuale.
    *   Mascheramento di parole complete.
*   **1.3. Obiettivo di Confine dello Span (SBO - Span-Boundary Objective):**
    *   Evita il mascheramento parziale di parole.
    *   Predizione di ogni token dello span usando solo i token ai confini dello span.
    *   Incoraggia il modello a registrare più informazioni sullo span nelle codifiche dei confini.
    *   Rappresentazione di ogni token dello span: codifiche dei confini + embedding di posizione relativa.
    *   Minimizzazione della perdita di cross-entropia (come in MLM).
    *   Perdita complessiva: somma delle perdite MLM e SBO per ogni token nello span.

**2. S-BERT: BERT per la Similarità di Frasi (Sentence Transformer)**

*   **2.1. Problema con BERT standard per la similarità:**
    *   Embedding di output di BERT (incluso il token CLS) sono un sottoprodotto, non l'obiettivo principale.
*   **2.2. Ottimizzazione per la codifica di frasi:**
    *   Modifica del pre-addestramento per migliorare l'apprendimento della similarità.
    *   Input: coppie di frasi (cross-encoding) invece di singole frasi (single-sentence encoding).
    *   Pre-addestramento per riconoscere la similarità tra le frasi.
*   **2.3. Implementazione:**
    *   Due frasi in input separate da `[SEP]`.
    *   Aggiunta di una testa di classificazione a BERT.
    *   Output della testa: score di similarità.
    *   Tutte le frasi processate sulla stessa sequenza di input.
*   **2.4. Further Pretraining, non Finetuning:**
    *   Il modello pre-addestrato può essere fine-tunizzato su un nuovo dataset.
*   **2.5. Similarity-Learning e Apprendimento Contrastivo:**
    *   Il modello impara a identificare similarità tra istanze semanticamente simili.
    *   Il modello impara ad allontanare esempi semanticamente differenti.

**3. Estensione di BERT per il Similarity Learning: Approcci Principali**

*   **3.1. Cross-encoding:**
    *   Singolo modello BERT.
    *   Input: coppia di frasi.
    *   Valutazione della similarità tramite una funzione obiettivo.
*   **3.2. Rete Siamese:**
    *   Due istanze dello stesso modello BERT (pesi condivisi).
    *   Ogni istanza riceve una frase in input.
    *   Calcolo della similarità tramite una funzione obiettivo applicata agli output delle due istanze.
    *   Genera *sentence embeddings* ottimizzati per la misurazione della similarità.
        *   **3.2.1. Processo della Rete Siamese:**
            *   Generazione degli embedding (es. average pooling).
            *   Aggiunta di residui (opzionale).
            *   Feed-forward.
            *   Calcolo della similarità (softmax o loss basata sull'apprendimento contrastivo).

*   **3.3. Obiettivo Comune:**
    *   Ottenere rappresentazioni vettoriali (sentence embeddings) che catturino la similarità semantica.

---

**Schema Riassuntivo**

**1. Funzioni Obiettivo per l'Apprendimento della Similarità Semantica**

   *   **1.1 Triplet Loss:**
        *   Confronta un'istanza target con un'istanza positiva (simile) e una negativa (diversa).
        *   Obiettivo: Distinguere tra similarità e dissimilarità.
   *   **1.2 Multiple Negative Ranking Loss:**
        *   Spinge il modello ad apprendere una similarità tra istanze positive maggiore della similarità tra istanze negative, di almeno un certo margine.
        *   Un margine più piccolo rende il modello più robusto.

**2. Fine-tuning di Modelli Pre-addestrati**

   *   **2.1 Vantaggi del Fine-tuning:**
        *   Richiede meno dati e converge più velocemente rispetto all'addestramento *from scratch*.
   *   **2.2 Parameter-Efficient Fine-tuning:**
        *   Approccio più efficiente dal punto di vista computazionale e dei parametri.
        *   **2.2.1 Lightweight Fine-tuning:**
            *   Congela alcuni pesi del modello pre-addestrato.
            *   Aggiunge nuovi layer da apprendere.
        *   **2.2.2 Adapted Tuning:**
            *   Aggiunge *adapter* (moduli aggiuntivi) sopra il modello pre-addestrato.
            *   Congela i pesi del modello pre-addestrato.
            *   Aggiorna solo i pesi degli *adapter*.
        *   **2.2.3 Prefix-Tuning e Prompt Tuning:**
            *   Non modificano l'architettura del modello pre-addestrato.
            *   Agiscono sull'input (aggiungendo token speciali) o aggiungendo parametri di prefisso in alcuni layer vicini all'input.
            *   **Vantaggio:** Ogni elemento di un batch durante l'inferenza potrebbe utilizzare un modello finetuned in modo diverso.

**3. Pretraining di Encoder-Decoder**

   *   **3.1 Approccio Generalizzato al Language Modeling:**
        *   Un prefisso di ogni input è fornito all'encoder e non è predetto
        $$h_{1},\dots,h_{r}=\text{Encoder}(w_{1},\dots,w_{r})$$
        $$h_{r+1},\dots,h_{r}=\text{Decoder}(w_{1},\dots,w_{t}),h_{1},\dots,h_{r}$$
        $$y_{i}\approx Ah_{i}+b,i>T$$
        *   La porzione di encoder beneficia di un contesto bidirezionale, mentre la porzione di decoder è usata per allenare l'intero modello.
   *   **3.2 Tecniche di Denoising e Span Corruption:**
        *   Generalizzazione del masked language modeling di BERT.
        *   Mascheramento di *span* di testo invece di singoli token.
        *   Obiettivo: Generazione di testo e ripristino degli *span* mascherati.
   *   **3.3 Token Sentinella:**
        *   Sostituzione degli *span* mascherati con token sentinella unici (identificati da un ID specifico).
        *   Target del pre-training: Associare a ciascun token sentinella il suo *bound* (testo corretto per riempire lo *span* mascherato).

---

## Schema Riassuntivo: Modelli di Linguaggio (T5, GPT, GPT-2, GPT-3)

**1. T5 (Text-to-Text Transfer Transformer)**
    *   **1.1 Architettura:** Encoder-Decoder
    *   **1.2 Addestramento:** Instruction Training (formulazione di task come problemi text-to-text)
        *   1.2.1 Fondamentale per la generazione del linguaggio.
        *   1.2.2 Token sentinella ricostruito come sequenza di testo.
        *   1.2.3 Ogni maschera ha una sua identità.

**2. Generative Pre-trained Transformer (GPT)**
    *   **2.1 Architettura:** Decoder-only Transformer
        *   2.1.1 12 layer (come BERT), 117 milioni di parametri.
        *   2.1.2 Stati nascosti a 768 dimensioni, layer feed-forward nascosti a 3072 dimensioni.
    *   **2.2 Tokenizzazione:** Codifica Byte-Pair con 40.000 merges.
    *   **2.3 Dataset:** BooksCorpus (oltre 7000 libri).
        *   2.3.1 Permette di apprendere dipendenze a lunga distanza.
    *   **2.4 Valutazione:** Natural Language Inference (NLI)
        *   2.4.1 Task di entailment: input di coppie di frasi etichettate (Entailment, Contraddittorio, Neutrale).
    *   **2.5 Input:** Sequenza di token con token di inizio frase e token di inizio decoder (fine frase per il primo token).

**3. GPT-2**
    *   **3.1 Architettura:** Decoder Transformer.
    *   **3.2 Dataset:** 40 GB di testo.
    *   **3.3 Parametri:** 1.5 miliardi.
    *   **3.4 Novità:** Zero-shot learning.

**4. Zero-shot Learning**
    *   **4.1 Definizione:** Capacità di eseguire task senza esempi positivi durante l'addestramento.
    *   **4.2 Utilità:**
        *   4.2.1 Limitazione dei dati.
        *   4.2.2 Difficoltà nella definizione del contesto.
    *   **4.3 Approccio:** Intervento sul prompt invece del fine-tuning.

**5. GPT-3**
    *   **5.1 Parametri:** 175 miliardi.
    *   **5.2 Dati:** Oltre 600 GB.
    *   **5.3 Novità:** Few-shot learning.

**6. Few-shot Learning (In-context Learning)**
    *   **6.1 Definizione:** Fornire esempi nel contesto dell'input (prompt).
    *   **6.2 Prestazioni:** Migliorano con l'aumento degli esempi.
    *   **6.3 Rendimento decrescente:** A partire da 3-4 esempi.

---
