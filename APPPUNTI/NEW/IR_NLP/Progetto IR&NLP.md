
Sviluppo di un agente conversazionale in grado di analizzare contenuti politici provenienti da social network, integrando tecniche di Sentiment Analysis e Retrieval-Augmented Generation (RAG).

Il sistema prevede diverse fasi:

1. **Raccolta e Pre-elaborazione dei Dati**: Il sistema acquisisce post da diverse piattaforme social (reddit o X), utilizzando API o tecniche di scraping dove necessario, e applica un filtro per rimuovere contenuti irrilevanti (es. spam, duplicati, o post fuori tema rispetto all'argomento di interesse, in questo caso la politica).
 (Eventualmente integrare questa fase nel progetto del secondo modulo del corso, per ottimizzare il carico di lavoro e sfruttare le sinergie tra i due progetti)

2. **Analisi del Sentiment e Indicizzazione**: Ogni post viene analizzato da un modello fine-tuned sul task di Sentiment Analysis, che assegna un’etichetta di sentiment: positivo, negativo o neutro. Questi dati vengono quindi indicizzati nel sistema RAG, rendendoli poi facilmente recuperabili in base a query specifiche. Come modello di riferimento, si valuta l'adozione di [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B) (nella versione con 1.5 miliardi di parametri).

3. **Retrieval e Generazione della Risposta**: In risposta a una query dell'utente, il sistema utilizza il RAG per recuperare i post rilevanti, filtrandoli ulteriormente in base al sentiment espresso. Ad esempio, se l'utente desidera sapere quali sono le critiche più frequenti rivolte a un politico X, il sistema seleziona i post che menzionano il politico, estrae quelli con sentiment negativo, e ne sintetizza i contenuti.
 La risposta generata non si limita a riportare i dati grezzi, ma offre una sintesi chiara e strutturata delle opinioni prevalenti, focalizzandosi sui temi specifici della query (ad esempio, controversie su dichiarazioni o politiche proposte).
 Per garantire un'elaborazione priva di bias, si prevede l'integrazione di un LLM non censurato come [dolphin-mistral 7b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-7B-32k), che evita distorsioni ideologiche o filtri imposti dal modello.

4. **Implementazione della Pipeline con Cheshire Cat AI**: La pipeline del sistema prevede l'uso del framework open-source [Cheshire Cat AI](https://cheshirecat.ai/), progettato per facilitare la creazione di agenti conversazionali avanzati. Cheshire Cat AI gestisce l'interazione tra il RAG, il modello di Sentiment Analysis e il Large Language Model utilizzato per la conversazione con memoria. La struttura del framework si basa su due catene principali: una *Procedurale* e una _di Memoria_.

 - **Catena Procedurale**: Questa catena si occupa di eseguire azioni specifiche, come l'interazione con strumenti esterni o l'elaborazione di task concreti, come l'analisi del sentiment o la selezione di post rilevanti. Ogni azione è definita per rispondere a una query o a una richiesta da parte dell'utente, ottimizzando la gestione del flusso di lavoro.

 - **Catena di Memoria**: Questa catena è responsabile per il recupero e l'elaborazione dei contesti rilevanti dalle memorie dichiarative ed episodiche. La memoria dichiarativa contiene fatti generali o conoscenze di base, mentre la memoria episodica conserva informazioni specifiche basate su interazioni precedenti o esperienze passate. 

 Il framework è pre-configurato per interagire anche con **Ollama** come backend, consentendo di eseguire i modelli di linguaggio localmente, senza la necessità di dipendere da server esterni.

 **Cheshire Cat AI** integra diverse funzionalità provenienti da **Langchain**, . Le principali funzionalità utilizzate includono:

 1. **Gestione delle Chains**: Cheshire Cat adotta il concetto di *LLM chains* per combinare modelli linguistici e processi in pipeline, permettendo l'elaborazione sequenziale dei dati.

 2. **Agent Framework**: Incorporando l'architettura degli *agenti* di Langchain, Cheshire Cat consente agli agenti di prendere decisioni basate sul contesto su quale catena eseguire, migliorando la flessibilità nell'interazione.

 3. **Memoria (Memory)**: Sia Langchain che Cheshire Cat utilizzano sistemi di memoria avanzati per mantenere il contesto delle conversazioni, con Langchain che supporta memorie vettoriali (vector memory) e Cheshire Cat che gestisce memorie procedurali(funzioni python), episodiche(conversazioni passate dell'utente) e dichiarative(documenti caricati).

 4. **Integrazione con Modelli e Strumenti Esterni**: Cheshire Cat sfrutta le capacità di Langchain di integrare modelli di linguaggio e strumenti esterni come database vettoriali (Qdrant) e API REST. Inoltre, Langchain permette di integrare facilmente altre risorse esterne, rendendo il framework estremamente versatile.

https://medium.com/@rslavanyageetha/vader-a-comprehensive-guide-to-sentiment-analysis-in-python-c4f1868b0d2e
