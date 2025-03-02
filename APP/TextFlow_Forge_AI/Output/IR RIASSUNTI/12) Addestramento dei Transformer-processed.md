
## Preprocessing nei Language Model: Tokenizzazione e Approcci Principali

Il preprocessing nei Language Model si concentra principalmente sulla tokenizzazione, semplificando la pipeline e evitando complesse tecniche di riduzione della dimensionalità.  La fase di pre-tokenizzazione gestisce il dato grezzo, distinguendo parole, punteggiatura e spazi, e trattando la case delle lettere.

La **tokenizzazione**, cruciale per la preparazione dei dati, crea un vocabolario di *sub-word token* appresi dal modello, anziché definiti a priori.  Il tokenizzatore, un *learner*, ottimizza una funzione obiettivo per determinare la migliore strategia di sub-wording. Il vocabolario include caratteri, simboli, parole intere e parti di parole.  Durante l'inferenza, ogni parola viene suddivisa in una sequenza di token noti; un token può coincidere con una parola intera.  Diverse strategie di sub-wording esistono, alcune preservando gli spazi (con token speciali), con variabilità nella reversibilità della gestione degli spazi.

Tre approcci principali alla tokenizzazione sono utilizzati:

### Byte-Pair Encoding (BPE)

BPE inizia con parole uniche e caratteri ASCII come token candidati.  Iterativamente, unisce la coppia di token più frequente.  Una variante, il **byte-level encoding**, usa byte come simboli. BPE preserva gli spazi con un token speciale e codifica le emoji usando caratteri Unicode.

### WordPiece

WordPiece usa un approccio iterativo di pattern matching, guidato dalla frequenza dei token più piccoli per creare quelli più lunghi.  Utilizza "##" per indicare lo split di un termine che appare come singola parola nel corpus.  Tende a minimizzare il numero di token generati e non preserva gli spazi.

### Differenze tra BPE e WordPiece

| Caratteristica          | BPE                                      | WordPiece                                   |
|--------------------------|-------------------------------------------|--------------------------------------------|
| Criterio di selezione   | Coppia di token più frequente             | Score: frequenza della coppia / frequenza dei singoli token |
| Lunghezza dei sub-word | Tendenzialmente più corti                  | Tendenzialmente più lunghi                   |
| Gestione "unknown"      | Solo caratteri non presenti nel vocabolario | Parole non scomponibili in sub-word noti     |


WordPiece, a differenza di BPE, dà priorità al merge di coppie le cui parti individuali sono meno frequenti nel vocabolario.

---

### Tokenizzazione Unigram

L'algoritmo Unigram, a differenza di BPE e WordPiece, inizia con un vocabolario completo e rimuove iterativamente i simboli meno informativi, quelli che causano il minor incremento della *likelihood loss* del corpus.  A differenza degli altri, non unisce token, ma li rimuove.  In fase di tokenizzazione, sceglie lo split con la massima *negative log likelihood*.  Non preserva gli spazi multipli, ma gestisce bene simboli come le emoji.  La tabella seguente riassume le differenze tra Unigram, BPE e WordPiece:

| Modello | Addestramento | Passo di addestramento | Apprende | Codifica | Esempi |
|---|---|---|---|---|---|
| BPE | Vocabolario piccolo, unisce token | Unione token più frequenti | Regole di unione e vocabolario | Divide in caratteri, applica unioni | GPT-2, RoBERTa, BART |
| WordPiece | Vocabolario piccolo, unisce token | Unione token con punteggio migliore (frequenza, preferenza per token meno frequenti) | Vocabolario | Sottoparola più lunga nel vocabolario | BERT |
| Unigram | Vocabolario grande, rimuove token | Rimuove token che minimizzano la loss | Vocabolario con punteggio per token | Suddivisione più probabile | ALBERT |


### Pre-addestramento dei Modelli Linguistici

I modelli *Word2Vec*, *context-free*, rappresentano singole parole indipendentemente dal contesto.  I *Transformer*, invece, incorporano il contesto.  Nel NLP moderno, quasi tutti i parametri sono inizializzati tramite pre-addestramento, un metodo che maschera parti dell'input e addestra il modello a ricostruirle. Questo crea rappresentazioni robuste del linguaggio, inizializzazioni efficaci e distribuzioni di probabilità per la generazione di testo.  Il pre-addestramento genera molteplici embedding intermedi per ogni token, ognuno dipendente dal contesto.  I task risolvibili includono ricostruzione sintattica, semantica lessicale e analisi del sentiment, spesso prevedendo token successivi o completando frasi.


### Paradigma Pretrain-Finetune

I modelli linguistici, definiti *word-models*, possiedono una conoscenza del mondo dipendente dalla dimensione e dai dati di addestramento.  Il pre-addestramento genera modelli di *Narrow AI*, successivamente specializzati tramite *fine-tuning* per compiti specifici.  L'obiettivo finale è l'AGI (Intelligenza Artificiale Generale).  I Transformer sono architetture chiave per la creazione di modelli pre-addestrati, come illustrato nell'immagine: ![[12) Addestramento dei Transformer-20241125164038232.png]]  Il *Language Modeling* è un compito di pre-addestramento comune per i decoder.

---

## Riassunto del Pre-addestramento di BERT

Questo documento descrive il pre-addestramento di BERT, un modello di linguaggio basato su Transformer.  A differenza del *language modeling* tradizionale, BERT utilizza il *Masked Language Modeling* (MLM) per ottenere una rappresentazione bidirezionale del testo.

**Masked Language Modeling (MLM):**  Il MLM consiste nel mascherare casualmente il 15% dei token di input con "[MASK]", sostituirne il 10% con token casuali e lasciare invariato il restante 10%. Il modello viene poi addestrato a predire i token mascherati, considerando il contesto circostante.  Questo approccio impedisce al modello di imparare scorciatoie e lo forza a costruire rappresentazioni robuste del linguaggio.

**BERT Architecture:** BERT utilizza un encoder Transformer bidirezionale.  Oltre al MLM, BERT impiega un ulteriore compito di pre-addestramento chiamato *Next Sentence Prediction* (NSP), che prevede di determinare se due sequenze sono consecutive in un documento.

**Tokenizzazione e Rappresentazione dell'Input:** BERT utilizza WordPiece per la tokenizzazione, con un vocabolario di 30.000 token.  La rappresentazione dell'input è la somma di *positional embeddings* e *segment embeddings*.  Due token speciali, `[CLS]` e `[SEP]`, vengono utilizzati rispettivamente per indicare l'inizio della sequenza e la separazione tra frasi multiple. L'embedding `[CLS]` rappresenta l'intera sequenza.

**Applicazione a diversi task:** Per applicare BERT a diversi task, si aggiunge una *testa di classificazione* all'ultimo layer.  L'embedding `[CLS]` viene tipicamente utilizzato come input per la classificazione del testo, oppure si può utilizzare un *average pooling* sugli embedding di tutti i token. BERT è disponibile in due varianti: una con 12 layer (768 dimensioni) e una con 24 layer (1024 dimensioni). I layer più profondi catturano relazioni semantiche a lungo raggio.  Il *fine-tuning* avviene partendo dagli embedding dell'ultimo livello.

---

## Riassunto di BERT e sue varianti

Questo documento descrive BERT (Bidirectional Encoder Representations from Transformers), il suo fine-tuning e alcune varianti migliorative come RoBERTa.

### Task a livello di sequenza con BERT

BERT può essere utilizzato per diversi task aggiungendo un layer specifico in cima al modello e poi effettuando un fine-tuning.  Per task di classificazione (es. entailment), si usa l'embedding del token speciale `[CLS]`; per task a livello di token (es. *sequence tagging*), si usano gli embedding di tutti i token di input.

### BERT come estrattore di features

Gli embedding di BERT, in particolare quelli degli ultimi quattro layer, possono essere integrati in altri modelli, come BiLSTM, migliorandone le prestazioni.

### Conoscenza codificata in BERT

Le informazioni contenute nei diversi layer di BERT sono gerarchicamente organizzate: i layer inferiori codificano informazioni sull'ordine delle parole, quelli intermedi informazioni sintattiche, mentre quelli superiori informazioni specifiche del task. Le informazioni semantiche sono distribuite su tutti i layer.

### Fine-tuning di BERT e esempi di task

BERT è stato con successo fine-tuned su numerosi task, ottenendo risultati allo stato dell'arte.  Esempi includono: QQP (rilevazione di parafrasi), QNLI (inferenza linguistica naturale), SST-2 (analisi del sentiment), CoLA (grammaticalità), STS-B (similarità semantica), MRPC (parafrasi), e RTE (inferenza testuale).  `![[]]`

### Estensioni di BERT: RoBERTa e altre

Diverse varianti di BERT sono state sviluppate, tra cui RoBERTa e SpanBERT.

### RoBERTa: Miglioramenti rispetto a BERT

RoBERTa migliora BERT principalmente attraverso un addestramento più lungo e più dati, rimuovendo la predizione della successiva frase (NSP) e utilizzando un *masked language modelling* dinamico.  Utilizza dataset più ampi (CC-News, Stories, OpenWebText) e sequenze più lunghe (512 token).  Impiega una tokenizzazione BPE con 50.000 sub-word.  Esistono due versioni: RoBERTa-base e RoBERTa-large.  RoBERTa supera BERT in diversi benchmark (GLUE, SQuAD v2.0, RACE), dimostrando che maggiore potenza di calcolo e dati migliorano significativamente le prestazioni del pre-training.

---

### SpanBERT: Un nuovo approccio al pre-addestramento di BERT

SpanBERT migliora BERT generalizzando il meccanismo di *masking*. Invece di mascherare singoli token, maschera *span* di token adiacenti, di lunghezza variabile (preferibilmente brevi) e sempre composti da parole complete.  Questo nuovo schema di mascheramento introduce un nuovo obiettivo di addestramento: lo **Span-Boundary Objective (SBO)**.  SBO prevede di predire ogni token all'interno dello *span* mascherato utilizzando solo i token di confine (il token prima dell'inizio e quello dopo la fine dello *span*).  Questo forza il modello a codificare informazioni rilevanti sullo *span* nei token di confine. La perdita totale è la somma della perdita MLM (Masked Language Modeling) e SBO.


### S-BERT: BERT per la similarità di frasi

S-BERT è un *sentence transformer* che ottimizza BERT per compiti di similarità tra frasi.  A differenza di BERT, che codifica token individuali, S-BERT si concentra sulla codifica di intere frasi.  Invece del *fine-tuning*, S-BERT utilizza un *further pre-training* con un approccio di *cross-encoding*: vengono utilizzate coppie di frasi come input, separate da un token `[SEP]`, e una testa di classificazione calcola uno score di similarità. Questo processo sfrutta il *similarity learning*, tipicamente implementato con un paradigma di *apprendimento contrastivo*.


### Estensioni di BERT per il Similarity Learning

Esistono due approcci principali per adattare BERT al *similarity learning*:

1. **Cross-encoding:** Un singolo modello BERT riceve in input una coppia di frasi, e la similarità è valutata tramite una funzione obiettivo appropriata.

2. **Rete Siamese:** Due istanze dello stesso modello BERT (con pesi condivisi) elaborano separatamente le due frasi.  Gli *embeddings* generati (tipicamente tramite *average pooling*) possono essere arricchiti con residui e poi passati a una rete *feed-forward*. La similarità è calcolata tramite una funzione *softmax* o una perdita basata sull'apprendimento contrastivo.  Questo approccio genera *sentence embeddings* ottimizzati per la misurazione della similarità.

In entrambi i casi, l'obiettivo è ottenere *sentence embeddings* che catturino efficacemente la similarità semantica tra le frasi.

---

### Funzioni Obiettivo per l'Apprendimento della Similarità Semantica

Due funzioni obiettivo principali vengono utilizzate per apprendere la similarità semantica: la *Triplet Loss* e la *Multiple Negative Ranking Loss*.  La *Triplet Loss* confronta un'istanza target con una positiva (simile) e una negativa (dissimile), mirando a distinguere tra similarità e dissimilarità. La *Multiple Negative Ranking Loss*  impone che la similarità tra istanze positive superi quella tra istanze negative di un certo margine, con margini più piccoli che aumentano la robustezza del modello.


### Fine-tuning di Modelli Pre-addestrati

Il fine-tuning di modelli pre-addestrati come BERT è più efficiente dell'addestramento *from scratch*, richiedendo meno dati e epoche.  Esistono diverse tecniche di *parameter-efficient fine-tuning*:

* **Lightweight Fine-tuning:** Congela i pesi del modello pre-addestrato e aggiunge nuovi layer da apprendere.
* **Adapted Tuning:** Aggiunge *adapter* (moduli aggiuntivi) sopra il modello pre-addestrato, congelando i pesi originali e aggiornando solo quelli degli *adapter*.
* **Prefix-Tuning e Prompt Tuning:** Non modificano l'architettura, ma agiscono sull'input aggiungendo token speciali o parametri di prefisso in layer vicini all'input, permettendo l'utilizzo di un modello finetuned in modo diverso per ogni elemento di un batch durante l'inferenza.  `![[]]`


### Pre-training di Encoder-Decoder

Il pre-training di modelli encoder-decoder si basa su un'estensione del *masked language modeling*.  Invece di mascherare singoli token, si mascherano *span* di testo (sezioni di testo) utilizzando una tecnica di *denoising* e *span corruption*.  

L'encoder processa un prefisso di input ($$h_{1},\dots,h_{r}=\text{Encoder}(w_{1},\dots,w_{r})$$), beneficiando di un contesto bidirezionale. Il decoder ($$h_{r+1},\dots,h_{r}=\text{Decoder}(w_{1},\dots,w_{t}),h_{1},\dots,h_{r}$$)  viene utilizzato per addestrare l'intero modello, con l'output approssimato da  $$y_{i}\approx Ah_{i}+b,i>T$$.

Ogni *span* mascherato è sostituito con un token sentinella unico. L'obiettivo del pre-training è associare a ciascun token sentinella il suo *bound*, ovvero predire il testo corretto per riempire lo *span* mascherato.

---

## Riepilogo dei Modelli Linguistici T5, GPT, GPT-2 e GPT-3

Questo documento riassume le caratteristiche chiave dei modelli linguistici T5, GPT, GPT-2 e GPT-3, evidenziando le differenze architettoniche e le capacità emergenti.

### T5 (Text-to-Text Transfer Transformer)

T5 è un modello encoder-decoder addestrato con la tecnica dell'**instruction training**.  Questa tecnica formula tutti i task come problemi di testo-a-testo, permettendo al modello di apprendere molteplici task contemporaneamente.  A differenza del masked language modelling, il token sentinella viene trattato come una sequenza di testo da ricostruire.

### GPT (Generative Pre-trained Transformer)

GPT è un modello *decoder-only* con 12 layer, 117 milioni di parametri, stati nascosti a 768 dimensioni e layer feed-forward nascosti a 3072 dimensioni.  Utilizza una tokenizzazione Byte-Pair con 40.000 merges ed è stato addestrato su BooksCorpus (oltre 7000 libri).  Viene valutato su benchmark come la *Natural Language Inference* (NLI), per compiti di *entailment*. L'input al decoder è una sequenza di token, incluso un token di inizio frase e un token di inizio/fine frase per il primo token.

### GPT-2

GPT-2 mantiene l'architettura *decoder-only* di GPT, ma con un dataset di addestramento di 40 GB di testo e 1.5 miliardi di parametri.  La novità principale è l'emergere di capacità di **zero-shot learning**.

### Zero-shot Learning

Lo *zero-shot learning* permette al modello di eseguire task senza aver visto esempi positivi durante l'addestramento, senza necessità di aggiornamenti del gradiente durante l'inferenza. È particolarmente utile quando i dati sono limitati o il contesto del task è difficile da definire, permettendo di guidare la generazione tramite modifiche al *prompt*.

### GPT-3

GPT-3 rappresenta un'ulteriore scalabilità, con 175 miliardi di parametri e un dataset di oltre 600 GB.  Questa scalabilità porta all'emergere del **few-shot learning**.

### Few-shot Learning (In-context Learning)

Il *few-shot learning*, o *in-context learning*, consiste nel fornire esempi nel *prompt* per guidare il modello nel task. Le prestazioni migliorano con l'aumentare degli esempi, ma mostrano un rendimento decrescente oltre 3-4 esempi.

---
