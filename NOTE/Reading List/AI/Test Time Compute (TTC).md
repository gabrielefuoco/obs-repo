TTC è una tecnica promettente per migliorare le capacità di ragionamento dei modelli linguistici. Questa tecnica introduce nuove sfide e opportunità, come la necessità di bilanciare le risorse computazionali e la possibilità di sviluppare modelli più efficienti. 

* **Definizione:** TTC è un processo che permette ai modelli linguistici di esplorare diverse tracce di ragionamento prima di fornire una risposta. 
* **Funzionamento:** Invece di generare direttamente una risposta, il modello genera diverse possibili tracce di ragionamento, partendo dallo stesso punto iniziale. 
* **Valutazione:** Le tracce di ragionamento vengono valutate tramite:
    * **Self-revision:** Il modello stesso valuta la validità del ragionamento e decide se proseguire o meno. Questo avviene tramite un prompt secondario che guida il modello nella valutazione.
    * **Verifier:** Un modello separato, addestrato a parte, valuta le tracce e assegna un punteggio. Il modello principale prosegue sulla traccia con il punteggio più alto.
* **Differenza da modelli tradizionali:** TTC introduce una fase aggiuntiva di ragionamento, a differenza dei modelli tradizionali che generano direttamente una risposta.

**Equiparazione di ricerca e ragionamento:**

* **Punto Critico:** TTC equipara la ricerca di possibili tracce di ragionamento al ragionamento stesso, in alcuni casi anche alla pianificazione. 
* **Implicazioni:** Questo sfuma la distinzione tra ricerca, pianificazione e ragionamento, tipica della psicologia cognitiva. 
* **Contesto:** I modelli linguistici, basati su reti neurali, non operano con la stessa distinzione tra questi concetti che è presente nella psicologia cognitiva.

**Nuove leggi di scala:**

* **Punto Critico:** TTC introduce una nuova variabile nelle leggi di scala dei modelli linguistici, ovvero il tempo dedicato all'esplorazione e alla valutazione delle tracce di ragionamento. 
* **Implicazioni:** 
    * **Modelli più piccoli:** Modelli più piccoli possono ottenere prestazioni simili a modelli più grandi, con un aumento dei tempi di inferenza.
    * **Risorse computazionali:** TTC richiede più tempo di calcolo rispetto ai modelli tradizionali, ma può ridurre il numero di parametri necessari.
* **Contesto:** Le leggi di scala tradizionali si basano su parametri, dati di addestramento e potenza di calcolo. TTC introduce il tempo di calcolo come una nuova variabile.
	* Le leggi di scala descrivono la relazione tra le risorse utilizzate per costruire un modello linguistico e le sue prestazioni

**Neurosimbolico:**

* **Punto Critico:** TTC rappresenta un passo verso l'integrazione di tecniche neurosimboliche nel deep learning. 
* **Analogia con AlphaGo/AlphaZero:** TTC ricorda il processo di generazione e valutazione di piani di azione utilizzato in AlphaGo e AlphaZero.
* **Integrazione di tecniche:** TTC integra tecniche di deep learning con elementi di ragionamento simbolico, come l'utilizzo di alberi di ricerca.
* **Contesto:** Le reti neurali codificano le informazioni in modo distribuito e opaco, mentre l'intelligenza artificiale simbolica si basa su entità e relazioni atomiche. TTC cerca di combinare questi due approcci.

**Token:**

* **Definizione:** Un token è un'unità di base del linguaggio, come una parola o un segno di punteggiatura.
* **Contesto:** I modelli linguistici generano testo tramite la produzione di token. TTC richiede la generazione di molti più token per esplorare diverse tracce di ragionamento.

**Albero di ricerca:**

* **Definizione:** Un albero di ricerca è una struttura dati che rappresenta tutte le possibili tracce di ragionamento.
* **Contesto:** TTC utilizza un albero di ricerca per esplorare le diverse tracce di ragionamento e valutare la loro promettentezza.

**Deep Learning:**

* **Definizione:** Deep learning è un tipo di apprendimento automatico che utilizza reti neurali profonde per apprendere da grandi quantità di dati.
* **Contesto:** I modelli linguistici sono basati su deep learning. TTC è una tecnica che si applica al deep learning per migliorare le capacità di ragionamento dei modelli linguistici.

---
## Glossario dei termini chiave

| Termine | Spiegazione |
|---|---|
| **Test Time Compute (TTC)** | Una tecnica che permette ai modelli linguistici di esplorare diverse tracce di ragionamento prima di fornire una risposta. |
| **Traccia di ragionamento** | Una sequenza di passaggi logici che un modello linguistico può seguire per arrivare a una conclusione. |
| **Self-revision** | Un processo in cui il modello linguistico valuta la validità del proprio ragionamento e decide se proseguire o meno. |
| **Verifier** | Un modello separato, addestrato a parte, che valuta le tracce di ragionamento e assegna un punteggio. |
| **Albero di ricerca** | Una struttura dati che rappresenta tutte le possibili tracce di ragionamento. |
| **Leggi di scala** | Descrivono la relazione tra le risorse utilizzate per costruire un modello linguistico e le sue prestazioni. |
| **Neurosimbolico** | Un approccio che integra tecniche di deep learning con elementi di ragionamento simbolico. |
| **Token** | Un'unità di base del linguaggio, come una parola o un segno di punteggiatura. |
| **Deep Learning** | Un tipo di apprendimento automatico che utilizza reti neurali profonde per apprendere da grandi quantità di dati. |
| **Inferenza** | Il processo di utilizzo di un modello addestrato per generare una risposta a un nuovo input. |
| **Parametri** | Variabili che definiscono le connessioni e i pesi all'interno di una rete neurale. |
| **Potenza di calcolo** | La capacità di un sistema di elaborare dati. |
| **Pianificazione** | Il processo di creazione di un piano di azione per raggiungere un obiettivo. |
| **Psicologia cognitiva** | Lo studio dei processi mentali come la memoria, il linguaggio e il ragionamento. |



