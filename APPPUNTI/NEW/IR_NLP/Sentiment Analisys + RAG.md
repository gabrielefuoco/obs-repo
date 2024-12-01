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
