
##### Preprocessing nei Language Model

* **Semplificazione della Pipeline:** Riduzione della complessità, senza task di riduzione della dimensionalità spinta sulla sintassi.

* **Pre-tokenizzazione:**
	* Trattamento del dato grezzo.
	* Distinzione parole/delimitatori (punteggiatura, spazi).
	* Gestione della case delle lettere.

* **Tokenizzazione:**
	* **Apprendimento del Vocabolario:** Il tokenizzatore apprende un vocabolario di *sub-word token* (token).
	* **Strategia di Sub-wording:** Apprendimento di una strategia ottimizzando una funzione obiettivo.
	* **Composizione del Vocabolario:** Caratteri, simboli non alfabetici, parole intere e parti di parole.
	* **Inferenza:** Splitting delle parole in sequenze di token noti; un token può coincidere con la parola intera.
	* **Gestione degli Spazi:** Variabilità nella gestione degli spazi (reversibilità?). Metodi basati su sequenze di parole, caratteri o byte.

* **Principali Approcci:**
* **Byte-Pair Encoding (BPE):**
	* **Addestramento:** Partenza da parole uniche e caratteri ASCII nel corpus.
	* **Merging Iterativo:** Unione iterativa delle coppie di token più frequenti.
	* **Byte-level Encoding (variante):** Simboli trattati come byte (GPT, BERT).
	* **Gestione Spazi ed Emoji:** Spazi con token speciali, emoji con caratteri Unicode.
* **WordPiece:**
	* **Pattern Matching Iterativo:** Creazione di token più lunghi a partire da quelli più frequenti.
	* **Prefisso "##":** Indica lo split di un termine che appare come singola parola nel corpus.
	* **Minimizzazione dei Token:** Tendenza a minimizzare il numero di token generati.
	* **Gestione Spazi:** Non preserva gli spazi.
* **Differenze con BPE:**
	* **Criterio di Selezione:** WordPiece usa uno score (frequenza coppia / frequenza singoli token), dando priorità a coppie con parti individuali meno frequenti.
	* **Lunghezza Sub-word:** WordPiece tende a sub-word più lunghe; token "unknown" per parole non scomponibili. BPE usa "unknown" solo per caratteri singoli non presenti.

##### Unigram Tokenizzazione

* **Approccio:** Algoritmo probabilistico inverso rispetto a BPE e WordPiece. Inizia con un vocabolario ampio e rimuove iterativamente i simboli meno informativi.
* **Fase di training:** Rimuove simboli che minimizzano l'incremento della *likelihood loss*.
* **Fase di inferenza:** Tutti i simboli vengono splittati.
* **Criterio di split:** Massimizza la *negative log likelihood*.
* **Gestione spazi:** Non preserva spazi multipli, ma cattura simboli speciali (emoji).

##### Model Pretraining

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

##### Paradigma Pretrain-Finetune

* **Modelli Linguistici:** Sono *word-models* con conoscenza del mondo dipendente da *model size* e dati di addestramento.
* **Utilizzo modelli pre-addestrati:** Crea modelli di *Narrow AI* specializzati tramite *fine-tuning*.
* **Obiettivo:** Raggiungere l'AGI (Intelligenza Artificiale Generale) prima della singolarità tecnologica.
* **Architettura chiave:** Transformer, utilizzati per generare modelli pre-addestrati.
* **Compiti di pre-addestramento:** Includono *Language Modeling* per i decoder.

##### Pre-addestramento degli Encoder

* **Masked Language Modeling (MLM):**
	* Sostituzione di una frazione di parole con "[MASK]".
	* Predizione delle parole mascherate.
	* Apprendimento di $p_\theta(x|\tilde{x})$, dove $\tilde{x}$ è la versione mascherata di $x$.
	* $h_1, ..., h_T = \text{Encoder}(w_1, ..., w_T)$ (rappresentazione dell'encoder)
	* $y_i \sim Aw_i + b$ (output)
	* Perdita calcolata solo sulle parole mascherate.
	* Analogo a un compito di "filling-in mask".

* **BERT (Bidirectional Encoder Representations from Transformers):**
	* Utilizza MLM per una rappresentazione bidirezionale profonda del testo.
	* Maschera casualmente il 15% dei token:
	* 80% sostituiti con "[MASK]".
	* 10% sostituiti con token casuali.
	* 10% lasciati invariati.
	* **Next Sentence Prediction (NSP):** Determina se due sequenze sono consecutive nel documento originale.

##### Tokenizzazione e Rappresentazione dell'Input (BERT)

* **Tokenizzazione:** Utilizza WordPiece (vocabolario di 30.000 token).
* **Lunghezza degli embeddings:** 768.
* **Rappresentazione dell'input:** Somma di *positional embeddings* e *segment embeddings*.
* **Token speciali:**
	* `[CLS]`: Rappresentazione dell'intera sequenza.
	* `[SEP]`: Separa le frasi in input.

##### Applicazione di BERT a diversi task

* **Classificazione del testo:**
	* Aggiunta di una testa di classificazione all'ultimo layer.
	* Utilizzo dell'embedding `[CLS]` come input per la classificazione.
	* Alternativamente, utilizzo di *average pooling* su tutti gli embedding di output.
	* **Varianti di BERT:** 12 layer (768 dimensioni) e 24 layer (1024 dimensioni).
	* **Specializzazione:** Utilizzo degli embedding dell'ultimo livello per il *fine-tuning*.
	* **Gerarchia di apprendimento:** Layer più vicini all'input apprendono relazioni a corto raggio, mentre quelli più vicini all'output apprendono relazioni semantiche (a lungo raggio).

##### Task a Livello di Sequenza con BERT

* **Aggiunta di Layer:** Aggiunta di un layer specifico per il task sopra il modello BERT, seguito da fine-tuning.
* **Utilizzo dell'embedding [CLS]:** Input per task di classificazione (es. entailment).
* **Utilizzo degli embedding dei token:** Input per task a livello di token (es. sequence tagging, question answering).

##### BERT come Feature Extractor

* **Integrazione con altri modelli:** Gli embedding di BERT (soprattutto degli ultimi 4 layer) migliorano le prestazioni di modelli come BiLSTM.

##### Conoscenza Codificata in BERT

* **Informazione sull'ordine delle parole:** Layer inferiori.
* **Informazione sintattica:** Layer intermedi.
* **Informazione specifica del task:** Layer superiori.
* **Informazione semantica:** Distribuita su tutti i layer.

##### Fine-tuning di BERT e Task

* **Successo e Versatilità:** Fine-tuning su diversi task con risultati allo stato dell'arte.
* **Esempi di Task:** QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE.

##### Estensioni di BERT

* **Miglioramenti nel pre-addestramento:** RoBERTa e SpanBERT.

##### RoBERTa

* **Differenze principali da BERT:**
	* Solo MLM come task di pre-training (rimozione di NSP).
	* Maggiori dati di addestramento (CC-News, Stories, OpenWebText).
	* Addestramento molto più lungo (MLM dinamico, sequenze più lunghe).
	* Tokenizzazione BPE con 50.000 sub-word.
	* Due dimensioni del modello: RoBERTa-base e RoBERTa-large.
	* **Risultati:** Prestazioni superiori a BERT su GLUE, SQuAD v2.0 e RACE. Dimostrazione dell'importanza di maggiore potenza di calcolo e dati nel pre-training.

##### SpanBERT

* **Differenza principale da BERT:** Maschera sequenze contigue di parole durante il pre-addestramento.

##### SpanBERT

* **Nuovo Approccio di Pre-addestramento:**
	* Mascheramento di span adiacenti di lunghezza variabile (distribuzione geometrica sbilanciata).
	* Mascheramento di parole complete.
	* Selezione casuale del punto di partenza dello span.
* **Obiettivo di Confine dello Span (SBO):**
	* Predizione di ogni token nello span usando solo i token ai confini.
	* Utilizzo delle codifiche di output dei confini e dell'embedding di posizione relativa del token target.
	* Minimizzazione della perdita di cross-entropia (MLM + SBO).

##### S-BERT: BERT per la Similarità di Frasi

* **Sentence Transformer:**
	* Migliora la codifica di intere frasi rispetto all'utilizzo di embedding CLS o di output di BERT.
	* Utilizza *cross-encoding* (coppie di frasi come input).
	* Aggiunge una testa di classificazione per calcolare lo score di similarità.
	* *Further Pretraining*, non fine-tuning.
* **Similarity Learning (Apprendimento Contrastivo):**
	* Impara a identificare similarità tra frasi semanticamente simili.
	* Impara a distinguere frasi semanticamente differenti.

##### Estensione di BERT per il Similarity Learning

* **Approcci Principali:**
	* **Cross-encoding:** Un singolo modello BERT con coppia di frasi in input.
	* **Rete Siamese:** Due istanze dello stesso modello BERT (condivisione dei pesi), una per ogni frase.
* **Processo della Rete Siamese:**
	* Generazione di embedding (es. average pooling).
	* Aggiunta di residui (opzionale).
	* Feed-forward.
	* Calcolo della similarità (softmax o loss contrastiva).
	* **Obiettivo:** Ottenere *sentence embeddings* che catturino la similarità semantica.

##### Funzioni Obiettivo per Apprendimento Similarità Semantica

* **Triplet Loss:** Confronta un'istanza target con una positiva (simile) e una negativa (dissimile) per distinguere similarità e dissimilarità.
* **Multiple Negative Ranking Loss:** Impone che la similarità tra istanze positive superi quella tra istanze negative di un certo margine. Margini più piccoli aumentano la robustezza del modello.

##### Fine-tuning di Modelli Pre-addestrati

* **Full Fine-tuning:** Fine-tuning di tutti i parametri del modello. Richiede meno dati e converge più velocemente rispetto all'addestramento *from scratch*.
* **Parameter-Efficient Fine-tuning:** Approcci più efficienti dal punto di vista computazionale e dei parametri.
* **Lightweight Fine-tuning:** Congela alcuni pesi del modello pre-addestrato e aggiunge nuovi layer.
* **Adapted Tuning:** Aggiunge *adapter* (moduli aggiuntivi) sopra il modello pre-addestrato; solo i pesi degli *adapter* vengono aggiornati.
* **Prefix-Tuning e Prompt Tuning:** Non modificano l'architettura; agiscono sull'input aggiungendo token speciali o parametri di prefisso. Vantaggio: ogni elemento di un batch può usare un modello finetuned in modo diverso.

##### Pre-training di Encoder-Decoder

* **Approccio:** Generalizza il masked language modeling. Utilizza tecniche di *denoising* e *span corruption*.
* **Meccanismo:** Maschera *span* di testo (non solo token singoli). L'obiettivo è la generazione di testo, ripristinando gli *span* mascherati.
* **Token Sentinella:** Ogni *span* mascherato è sostituito con un token sentinella unico. Il target è associare ogni token sentinella al suo *bound* (testo corretto).
* **Formulazione Matematica:**
	* $$h_{1},\dots,h_{r}=\text{Encoder}(w_{1},\dots,w_{r})$$
	* $$h_{r+1},\dots,h_{r}=\text{Decoder}(w_{1},\dots,w_{t}),h_{1},\dots,h_{r}$$
	* $$y_{i}\approx Ah_{i}+b,i>T$$
L'encoder beneficia di un contesto bidirezionale, il decoder allena l'intero modello.

##### Modelli Linguistici di Grandi Dimensioni: T5, GPT, GPT-2, GPT-3

##### T5 (Text-to-Text Transfer Transformer)

* Architettura: Encoder-Decoder
* Addestramento: Instruction Training (task multipli formulati come problemi testo-a-testo)
* Gestione token sentinella: Trattato come sequenza di testo, non con masked language modelling.

##### GPT (Generative Pre-trained Transformer)

* Architettura: Decoder-only Transformer (12 layer, 117 milioni di parametri, stati nascosti a 768 dimensioni, layer feed-forward nascosti a 3072 dimensioni)
* Tokenizzazione: Byte-Pair Encoding (40.000 merges)
* Dataset: BooksCorpus (>7000 libri)
* Valutazione: Benchmark NLI (Natural Language Inference) per il task di entailment (Entailment, Contraddittorio, Neutrale)
* Input: Sequenza di token con token di inizio frase e token di inizio/fine frase per il decoder.

##### GPT-2

* Architettura: Decoder Transformer
* Dataset: 40 GB di testo
* Parametri: 1.5 miliardi
* Capacità emergente: Zero-shot learning

##### Zero-shot Learning

* Definizione: Esecuzione di task senza esempi positivi durante l'addestramento. Nessun aggiornamento del gradiente durante l'inferenza.
* Vantaggi:
* Utile per problemi text-to-text (es. risposta a domande).
* Alternativa al fine-tuning in caso di:
* Limitazione dei dati.
* Difficoltà nella definizione del contesto.
* Implementazione: Modifica del prompt per guidare la generazione.

##### GPT-3

* Parametri: 175 miliardi
* Dati: >600 GB
* Capacità emergente: Few-shot learning (In-context learning)

##### Few-shot Learning (In-context Learning)

* Definizione: Fornire esempi nel prompt per guidare il modello.
* Efficacia: Miglioramento delle prestazioni con l'aumento degli esempi, ma con rendimenti decrescenti dopo 3-4 esempi.

