Sviluppo di un agente conversazionale in grado di analizzare contenuti provenienti da social network, integrando tecniche di Sentiment Analysis e Retrieval-Augmented Generation.

Il sistema si articola in più fasi:

1. Raccolta e Pre-elaborazione dei Dati:  
    I post vengono acquisiti da diverse piattaforme social e sottoposti a un processo di filtraggio per eliminare contenuti irrilevanti, come spam o post non pertinenti. Se possibile, proporrei di integrare questa fase nel progetto relativo al secondo modulo del corso, per ottimizzare il carico di lavoro e sfruttare le sinergie tra i due progetti.
    
2. Analisi del Sentiment e Indicizzazione:  
    Ogni post viene analizzato da un modello di Sentiment Analysis fine-tuned, che assegna un’etichetta di sentiment — positivo, negativo o neutro. Questi dati vengono quindi indicizzati nel sistema RAG, rendendoli facilmente recuperabili in base a criteri specifici.
    
3. Retrieval e Generazione della Risposta:  
    Quando un utente invia una query, il sistema utilizza il RAG per recuperare i post rilevanti, filtrandoli ulteriormente in base al sentiment. Ad esempio, se l’utente cerca informazioni sulle preoccupazioni legate a un argomento specifico, il sistema seleziona i post con sentiment negativo e ne sintetizza i temi principali, fornendo una risposta chiara e mirata.
    

Per implementare questa pipeline, intendo utilizzare il framework open-source Cheshire Cat AI ([cheshirecat.ai](https://cheshirecat.ai/)), che facilita la cr
eazione di un agente garantendo l'interazione tra il RAG, il modello di Sentiment Analysis e il Large Language Model per conversare con la memoria. 




**Obiettivo**: Sviluppare un sistema di Information Retrieval che combina Sentiment Analisys e tecniche di retrieval basate su modelli RAG (Retrieval-Augmented Generation). Il progetto mira a facilitare il recupero e l'organizzazione di informazioni rilevanti da post sui social network, rispondendo a query specifiche con un filtro basato su argomenti e sentiment.

1. **Raccolta e indicizzazione dei post**:
    - La fase di raccolta fa parte del progetto per il secondo modulo. La raccolta avverrà da diversi social network. È prevista una fase di filtraggio e pulizia di post non rilevanti (spam, post di topic non rilevanti, e altro)
    - Ogni post viene acquisito dalla piattaforma e indicizzato in un database.
    - **Classificazione del sentiment**: Ogni post viene analizzato da un modello di sentiment analysis (fine-tuned), che lo etichetta come **positivo**, **negativo** o **neutro**.
	    - Questa etichetta sarà poi usata per filtrare i risultati nella fase 2.
      
2. **Organizzazione con RAG**:
    - Il RAG permette di recuperare i post rilevanti sulla base della query dell'utente.
    - Si inserisce un filtro che estrae solo i post con un determinato sentiment.
    - A questo punto, il RAG fa un retrieval dei post, utilizzando gli embedding per trovare quelli più simili alla query (ad esempio, recuperando post che parlano di "energia rinnovabile" se la query riguarda l'energia).
    - Quindi il sistema genera una risposta (ad esempio, un riassunto) dei post trovati, basandosi su sentiment e argomento.

3. **Chashire Cat AI:**
	- Utilizzare questo framework open source come endpoint per interagire con l'applicazione.
	- In particolare, verrà usato inizialmente per creare una pipeline per l'aggiornamento della memoria e l'acquisizione del sentiment.
	- Il sistema integra già un RAG e la possibilità di inserire un LLM tramite API(per comunicare coi post etichettati).
	  
- **Query esempio: "quali sono le preoccupazioni riguardo l'argomento X?"**
	- Quando viene inviata una query, il modello di retrieval seleziona i post pertinenti all'argomento X (per esempio, "energia rinnovabile").
	- Successivamente, il modello di sentiment filtra solo quelli con un sentiment negativo.
	- Il sistema restituisce quindi una sintesi dei pensieri negativi riguardo all'argomento X.


# Architettura di un Agente Conversazionale RAG con Sentiment Analysis utilizzando Cheshire Cat AI

## 1. Architettura del Sistema

### Componenti Principali
- Raccolta Dati
- Pre-elaborazione 
- Sentiment Analysis
- Indicizzazione RAG
- Retrieval Intelligente
- Generazione Risposta

### Flusso di Interazione
1. **Acquisizione Dati**: Raccolta di contenuti da piattaforme social
2. **Pre-elaborazione**: Pulizia e filtraggio dei contenuti
3. **Sentiment Analysis**: Classificazione dell'umore dei testi
4. **Indicizzazione**: Arricchimento dei documenti con metadati sentimentali
5. **Retrieval Mirato**: Selezione di documenti rilevanti basata su query e sentiment
6. **Generazione Risposta**: Sintesi contestualizzata

### Ruolo di Cheshire Cat AI
- Framework di orchestrazione dell'intera pipeline
- Gestione dei plugin per Sentiment Analysis e RAG
- Interfaccia per l'integrazione di modelli LLM
- Supporto per l'estensibilità e la modularità

## 2. Raccolta e Pre-elaborazione dei Dati

### Sorgenti Dati
- Social Media API (Twitter, Reddit, Facebook)
- Forum tematici
- Blog e piattaforme di discussione

### Metodi di Filtraggio
- Rimozione spam tramite machine learning
- Identificazione contenuti irrilevanti mediante:
  - Analisi dei metadati
  - Rilevamento lingua
  - Soglie di qualità del contenuto

### Integrazione con Modulo di Raccolta Dati
- Vantaggi:
  - Ottimizzazione del carico computazionale
  - Standardizzazione del processo di acquisizione
  - Possibilità di arricchimento dei dati

## 3. Analisi del Sentiment e Indicizzazione

### Modello di Sentiment Analysis
- Architettura: Transformer fine-tuned
- Modello Base: BERT multilingua
- Fine-tuning su dataset specifici del dominio
- Classi di Sentiment:
  - Positivo
  - Neutro
  - Negativo
  - Sfumature emozionali

### Processo di Indicizzazione
- Arricchimento dei documenti con etichette sentimentali
- Calcolo di score sentimentali
- Memorizzazione in indice vettoriale con metadati
- Supporto per ricerche semantiche e sentimentali

## 4. Retrieval e Generazione della Risposta

### Meccanismo di Retrieval
- Ricerca semantica vettoriale
- Filtraggio basato su sentiment
- Ranking dei documenti tramite:
  - Rilevanza semantica
  - Coerenza sentimentale
  - Contesto della query

### Esempio di Generazione Risposta
Query: "Preoccupazioni su cambiamenti climatici"
1. Retrieval di post con sentiment negativo
2. Analisi dei temi ricorrenti
3. Sintesi delle principali argomentazioni
4. Generazione risposta oggettiva

### Metodo di Sintesi
- Tecnica di summarization extractive/abstractive
- Mantenimento dell'integrità informativa
- Contestualizzazione delle prospettive

## 5. Implementazione con Cheshire Cat AI

### Componenti Chiave
- Plugin personalizzati
- Integrazione modelli LLM
- Gestione memoria conversazionale
- Configurazione dinamica

### Ottimizzazione
- Caching intelligente
- Caricamento modulare dei componenti
- Supporto per modelli quantizzati

## 6. Valutazione del Sistema

### Metriche
- Precisione Sentiment Analysis
  - F1 Score
  - Confusion Matrix
- Qualità Retrieval
  - Precision@K
  - Mean Average Precision
- Valutazione Risposte Generate
  - Coerenza
  - Rilevanza
  - Copertura prospettive

### Metodologia
- Test set etichettato manualmente
- Valutazione cross-domain
- Confronto con baseline

## Conclusioni
Sistema innovativo che integra RAG, Sentiment Analysis e Cheshire Cat AI per un'esperienza conversazionale contestualmente ricca e sfaccettata.