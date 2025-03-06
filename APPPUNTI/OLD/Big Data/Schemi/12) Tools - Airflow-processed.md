
**Apache Airflow: Schema Riassuntivo**

**1. Introduzione ad Apache Airflow**
    *   Piattaforma open-source per sviluppo, pianificazione e monitoraggio di workflow.
    *   Progetto Apache Software Foundation (dal 2019).
    *   Workflow definiti come DAG (Directed Acyclic Graph) in Python.
        *   Vantaggi:
            *   Controllo di versione.
            *   Sviluppo collaborativo.
            *   Scrittura di test.
    *   Alto livello di astrazione: combina task e dipendenze.
    *   Scheduler esegue task su worker in base alle dipendenze.
    *   Tipi di parallelismo:
        *   Parallelismo dei dati: stesso codice su diversi chunk di dati.
        *   Parallelismo dei task: task diversi in parallelo.

**2. Principi Fondamentali**
    *   Dinamicità: pipeline configurate come codice Python (generazione dinamica).
    *   Estensibilità: operatori per molte tecnologie, componenti estensibili.
    *   Flessibilità: parametrizzazione con Jinja2.

**3. Architettura di Apache Airflow**
    *   Componenti:
        *   Scheduler: attiva workflow pianificati, invia task all'Executor.
        *   Executor: gestisce l'esecuzione dei task (distribuzione su Worker).
        *   Webserver: interfaccia utente per ispezione, attivazione e debug.
        *   DAG Directory: contiene i file DAG.
        *   Database dei metadati: memorizza lo stato (usato da Scheduler, Executor, Webserver).

**4. Programmazione di Base**
    *   DAG: definisce dipendenze, ordine di esecuzione, ripetizione.
    *   Task: azioni da intraprendere (recupero dati, analisi, attivazione sistemi).

**5. Tipi di Task**
    *   Operatori: task predefiniti per costruire rapidamente un DAG.
    *   Sensori: operatori che attendono eventi esterni.
    *   TaskFlow: funzioni Python personalizzate come task.

**6. Operatori Airflow**
    *   Modello per task predefinito, definito dichiarativamente.
    *   Operatori core popolari:
        *   BashOperator: esegue comandi bash.
        *   PythonOperator: chiama funzioni Python.
        *   EmailOperator: invia email.
    *   Esempio:
        *   SimpleHttpOperator: richiesta GET (meteo).
        *   PythonOperator: formatta risposta HTTP.
        *   EmailOperator: invia email con informazioni.

**7. Sensori**
    *   Attendono il verificarsi di un evento.
    *   Modalità:
        *   poke: occupa slot worker durante l'esecuzione.
        *   reschedule: occupa slot solo durante il controllo, si mette in pausa.
    *   Scelta tra `poke` e `reschedule` dipende dalla frequenza di controllo.

**8. TaskFlow**
    *   API per semplificare la creazione di DAG con decoratore `@task`.

**9. Esempio Applicazione: Ensemble Learning con TaskFlow**
    *   Tecnica del voto.
    *   Passi:
        *   Suddivisione dataset in training e test.
        *   Esecuzione parallela di *n* algoritmi di classificazione sul training set.
        *   Meccanismo di voto per la classificazione dell'ensemble sul test set.
    *   Task (funzioni Python):
        *   `load`: carica il dataset (es. `load_breast_cancer` da `sklearn.datasets`).
        *   `partition`: suddivide il dataset in training e test.
        *   `train`: esegue il fitting di un modello `sklearn` sui dati di training.
        *   `vote`: aggrega le predizioni dei modelli tramite voto.

---
