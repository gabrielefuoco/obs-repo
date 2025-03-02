
Apache Airflow è una piattaforma open-source per la creazione, pianificazione e monitoraggio di workflow di elaborazione dati, rappresentati come DAG (Directed Acyclic Graph) definiti in Python.  Questo approccio offre vantaggi come il controllo di versione, la collaborazione e la possibilità di scrittura di test. Airflow astrae la complessità, semplificando la combinazione di task e la definizione delle loro dipendenze, gestite dallo scheduler che le esegue su worker.  Supporta il parallelismo dei dati (molti task eseguono lo stesso codice su dati diversi) e dei task (task diversi eseguono in parallelo).

**Principi chiave:** dinamicità (pipeline configurate come codice Python), estensibilità (operatori per diverse tecnologie e componenti estendibili) e flessibilità (parametrizzazione tramite Jinja2).

**Architettura:**  `![[]]`.  Comprende Scheduler (gestisce l'attivazione dei workflow), Executor (gestisce l'esecuzione dei task su Worker), Webserver (interfaccia utente), DAG Directory (contiene i file DAG) e Database dei metadati (memorizza lo stato).

**Programmazione in Airflow:** Un DAG definisce le dipendenze tra i task. I task sono azioni (es. recupero dati, analisi).  Le dipendenze tra task sono indicate con `>>`.

**Tipi di task:** Operatori (task predefiniti), Sensori (attendono eventi esterni, modalità `poke` o `reschedule`), TaskFlow (funzioni Python personalizzate come task).

**Operatori:**  Esempi di operatori core includono `BashOperator`, `PythonOperator`, `EmailOperator`.  Un esempio mostra l'utilizzo di `SimpleHttpOperator`, `PythonOperator` e `EmailOperator` per ottenere e inviare informazioni meteo via email.

**Sensori:** Attendono eventi esterni.  `poke` occupa uno slot worker costantemente, `reschedule` solo durante i controlli. La scelta dipende dalla frequenza di controllo.

**TaskFlow:** Semplifica la creazione di DAG usando il decoratore `@task`.  Un esempio di applicazione di ensemble learning con TaskFlow illustra come suddividere un dataset, eseguire algoritmi di classificazione in parallelo (`train` task), e aggregare le predizioni tramite voto (`vote` task), usando task come `load` (carica dati) e `partition` (suddivisione dati).  `![[]]`.

---
