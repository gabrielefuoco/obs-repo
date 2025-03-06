
Apache Airflow è una piattaforma open-source per lo sviluppo, la pianificazione e il monitoraggio di workflow. Progetto di primo livello della Apache Software Foundation dal 2019, permette di creare applicazioni di elaborazione dati come DAG (Directed Acyclic Graph) di task. I DAG sono definiti in Python, offrendo vantaggi quali:

* Archiviazione dei workflow nel controllo di versione, permettendo il ritorno a versioni precedenti.
* Sviluppo collaborativo di workflow da più persone contemporaneamente.
* Scrittura di test per la convalida delle funzionalità.

Airflow offre un alto livello di astrazione, semplificando la costruzione di workflow combinando task e specificando le dipendenze tra essi. Lo scheduler esegue i task su worker, considerando le dipendenze definite nel DAG. Il runtime supporta due tipi di parallelismo:

* **Parallelismo dei dati:** molti task eseguono in parallelo lo stesso codice su diversi chunk di dati.
* **Parallelismo dei task:** task (o operatori) diversi vengono eseguiti in parallelo.

### Principi principali di Apache Airflow

* **Dinamicità:** le pipeline sono configurate come codice Python, permettendo la generazione dinamica di pipeline.
* **Estensibilità:** Airflow include operatori per connettersi a molte tecnologie e tutti i componenti sono estensibili.
* **Flessibilità:** Airflow facilita la parametrizzazione dei workflow usando il motore di templating Jinja2.

#### Architettura di Apache Airflow

![[Pasted image 20250223174242.png|438]]

#### Componenti di Apache Airflow

* **Scheduler:** gestisce l'attivazione dei workflow pianificati e l'invio dei task all'Executor.
* **Executor:** gestisce l'esecuzione dei task. Gli Executor di produzione possono distribuire l'esecuzione su una serie di **Worker**.
* **Webserver:** fornisce un'interfaccia utente per ispezionare, attivare e debuggare DAG e task.
* **DAG Directory:** contiene i file DAG letti dallo Scheduler, dall'Executor e dai Worker.
* **Database dei metadati:** usato dallo Scheduler, dall'Executor e dal Webserver per memorizzare lo stato.

## Nozioni di programmazione di base in Apache Airflow

Un DAG definisce le dipendenze tra i task, il loro ordine di esecuzione e il comportamento di ripetizione. I task descrivono le azioni da intraprendere (recuperare dati, eseguire analisi, attivare sistemi, etc.). L'esempio seguente definisce un DAG "backup giornaliero" (inizio 1° gennaio 2023, esecuzione giornaliera) con quattro task (script bash), usando `>>` per indicare le dipendenze.

### Tipi di task in Apache Airflow

* **Operatori:** task predefiniti per costruire rapidamente un DAG.
* **Sensori:** operatori che attendono il verificarsi di un evento esterno.
* **TaskFlow:** funzioni Python personalizzate come task.

#### Operatori Airflow

Un operatore è un modello per un task predefinito, definito dichiarativamente in un DAG. Airflow offre molti operatori, alcuni integrati nel core o in provider preinstallati.

### Operatori Core

Alcuni operatori core popolari includono:

* **BashOperator:** esegue un comando bash.
* **PythonOperator:** chiama una funzione Python.
* **EmailOperator:** invia un'email.

#### Esempio di utilizzo degli operatori

Un esempio utilizza:

* **SimpleHttpOperator:** esegue una richiesta GET per ottenere informazioni meteo.
* **PythonOperator:** formatta la risposta HTTP.
* **EmailOperator:** invia un'email con le informazioni formattate.

## Sensori in Apache Airflow

I sensori attendono il verificarsi di un evento. Hanno due modalità:

* **poke:** occupa uno slot worker per tutta la durata dell'esecuzione.
* **reschedule:** occupa uno slot solo durante il controllo, mettendosi in pausa tra i controlli.

La scelta tra `poke` e `reschedule` dipende dalla frequenza di controllo (es. `poke` per controlli al secondo, `reschedule` per controlli al minuto).

**TaskFlow:** Per programmatori che usano codice Python semplice, l'API TaskFlow semplifica la creazione di DAG usando il decoratore `@task`.

## Esempio di Applicazione Ensemble Learning con TaskFlow

Questo esempio mostra come implementare un'applicazione di ensemble learning usando la tecnica del voto:

- Il dataset viene suddiviso in set di training e test.
- *n* algoritmi di classificazione vengono eseguiti in parallelo sul set di training.
- Un meccanismo di voto esegue la classificazione dell'ensemble sul set di test.

![[_page_17_Figure_2.jpeg|400]]

Il flusso di lavoro è composto dai seguenti task (funzioni Python):

• `load`: carica il dataset usando la funzione `load_breast_cancer` di `sklearn.datasets`; i dati e i target vengono restituiti in una tupla.

• `partition`: riceve il dataset caricato dal task `load` e lo suddivide in set di training e test; restituisce i dati suddivisi come un dizionario.

• `train`: riceve i dati di training e un'istanza di uno stimatore `sklearn`, ed esegue l'operazione di fitting; il modello fittato viene restituito come output.

• `vote`: riceve i dati di test e una lista di modelli fittati; usa i modelli per calcolare una lista di predizioni; successivamente, le predizioni vengono aggregate tramite voto, per ottenere la classificazione dell'ensemble.
