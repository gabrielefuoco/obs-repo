| **Termine**                      | **Definizione**                                                                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Apache Airflow**               | Piattaforma open-source per lo sviluppo, la pianificazione e il monitoraggio dei workflow di elaborazione dati.                                                        |
| **DAG (Directed Acyclic Graph)** | Struttura che rappresenta un workflow in Airflow, definendo le attività (task) e le loro dipendenze in un grafo aciclico diretto.                                      |
| **Task**                         | Unità di lavoro all'interno di un DAG, che rappresenta un'azione specifica come il recupero di dati, l'esecuzione di un comando o l'attivazione di un sistema esterno. |
| **Operator**                     | Template predefinito per un task in Airflow, che definisce un'operazione specifica come l'esecuzione di un comando Bash o l'invio di un'email.                         |
| **Sensor**                       | Tipo speciale di operator che attende il verificarsi di un evento esterno prima di procedere con l'esecuzione del task successivo nel DAG.                             |
| **TaskFlow**                     | API di Airflow che consente di definire i task come funzioni Python, semplificando la scrittura e la gestione dei DAG.                                                 |
| **Scheduler**                    | Componente di Airflow responsabile dell'attivazione dei DAG e dell'invio dei task all'executor in base alla pianificazione definita.                                   |
| **Executor**                     | Componente di Airflow responsabile dell'esecuzione dei task, delegando il lavoro ai worker.                                                                            |
| **Worker**                       | Processo o macchina che esegue effettivamente i task inviati dall'executor.                                                                                            |
| **Jinja**                        | Motore di templating utilizzato da Airflow per la parametrizzazione dei workflow, consentendo di utilizzare variabili e logica condizionale all'interno dei DAG.       |
| **XCom**                         | Meccanismo di comunicazione tra task in Airflow, che consente di passare dati da un task all'altro.                                                                    |
| **Ensemble Learning**            | Tecnica di apprendimento automatico che combina le previsioni di più modelli per ottenere prestazioni migliori rispetto a un singolo modello.                          |
| **Parallelismo dei Dati**        | Esecuzione di task paralleli sullo stesso codice, applicato a diversi sottoinsiemi di dati.                                                                            |
| **Parallelismo dei Task**        | Esecuzione simultanea di task diversi all'interno di un DAG.                                                                                                           |


**Apache Airflow** è una piattaforma open-source per lo sviluppo, la pianificazione e il monitoraggio dei workflow. Consente di creare applicazioni di elaborazione dati strutturate come DAG (Directed Acyclic Graph) di task.

### Caratteristiche principali
- **Workflow come codice Python**: I DAG vengono definiti in Python, consentendo:
  - Archiviazione dei workflow in controllo di versione (rollback a versioni precedenti).
  - Sviluppo collaborativo dei workflow da parte di più persone.
  - Scrittura di test per validare i workflow.
---
# Strumenti di Programmazione basati su Workflow

## Apache Airflow
- **Astrazione elevata**: Semplifica la costruzione dei workflow combinando task e specificando dipendenze tra di essi.
- **Scheduler**: Esegue i task su una serie di worker, tenendo conto delle dipendenze definite nel DAG.
- **Supporto per il parallelismo**:
  - **Parallelismo dei dati**: Esecuzione di task paralleli sullo stesso codice, applicato a diversi chunk di dati.
  - **Parallelismo dei task**: Esecuzione simultanea di task diversi.

#### Principi di Apache Airflow
- **Dinamicità**: I pipeline possono essere generati dinamicamente grazie alla flessibilità del codice Python.
- **Estensibilità**: Airflow supporta operatori che si connettono a molte tecnologie; ogni componente è estendibile per adattarsi all'ambiente utente.
- **Flessibilità**: Supporta la parametrizzazione dei workflow grazie al motore di templating **Jinja** 
#### Componenti di Apache Airflow
- **Scheduler**: Attiva i workflow programmati e invia task all'**Executor**.
- **Executor**: Gestisce l'esecuzione dei task, delegando i task ai worker in ambienti di produzione.
- **Webserver**: Fornisce un'interfaccia utente per ispezionare, attivare e fare debug su DAG e task.
- **Directory DAG**: Contiene i file DAG letti da Scheduler, Executor e Worker.
- **Database Metadata**: Utilizzato da Scheduler, Executor e Webserver per memorizzare lo stato dei task e dei DAG.

#### Nozioni di base
- Un **DAG** definisce:
  - Le dipendenze tra i task.
  - L'ordine di esecuzione dei task.
  - La gestione dei tentativi di esecuzione falliti.
- I **task** definiscono l'azione specifica, che può essere il recupero di dati, l'analisi, l'attivazione di sistemi esterni, ecc.
- Esempio di un DAG:
  - Un DAG chiamato "daily backup", inizia il 1° gennaio 2023 e viene eseguito quotidianamente.
  - Quattro task eseguono script bash diversi.
  - Le dipendenze tra i task sono espresse con `>>`, che stabilisce la sequenza di esecuzione.
  
```python
# Dichiarazione DAG
with DAG(dag_id="daily_backup", start_date=datetime(2023, 1, 1),
         schedule="0 0 * * *") as dag:
    # Definizione di quattro task che eseguono comandi bash
    task_A = BashOperator(task_id="task_A",
                          bash_command="mv /backup/*.tgz /backup/old")
    task_B = BashOperator(task_id="task_B",
                          bash_command="tar czf /backup/http_log.tgz /var/log/http")
    task_C = BashOperator(task_id="task_C",
                          bash_command="tar czf /backup/mail_log.tgz /var/log/mail")
    task_D = BashOperator(task_id="task_D",
                          bash_command="echo backup completed")

    # Definizione delle dipendenze tra i task
    task_A >> [task_B, task_C] >> task_D
```
**Esempio di dichiarazione DAG.**
```python
# Definizione delle dipendenze dei task
task_A >> [task_B, task_C]
[task_B, task_C] >> task_D
```
---
# Tipi di Task in Airflow
- **Operators**: Task predefiniti che i programmatori possono concatenare per costruire la maggior parte di un DAG.
- **Sensors**: Sottoclasse speciale di **Operators** che attende un evento esterno.
- **TaskFlow Tasks**: Funzioni Python personalizzate, impacchettate come task.

## Operators
- Un **Operator** è un template predefinito per un task, definito dichiarativamente all'interno di un DAG.
    
- Airflow offre un ampio set di **Operators**, molti inclusi nel core o nei provider pre-installati.
    **Operator più popolari del core**:
    - **BashOperator**: Esegue un comando Bash.
    - **PythonOperator**: Esegue una funzione Python arbitraria.
    - **EmailOperator**: Invia un'email.

### Esempio di Operators

```python
endpoint = "https://api.open-meteo.com/v1/forecast"
latitude = 39.30
longitude = 16.25
parameters = ["temperature_2m_max", "temperature_2m_min", 
              "precipitation_sum", "sunrise", "sunset", "windspeed_10m_max",
              "winddirection_10m_dominant"]
timezone = "Europe/Berlin"
today = pendulum.now().strftime("%Y-%m-%d")
weather_query = f"{endpoint}?latitude={latitude}&longitude={longitude}&daily={','.join(parameters)}&timezone={timezone}&start_date={today}&end_date={today}"

def build_body(**context):
    query_result = context['ti'].xcom_pull('submit_query')
    weather_info = json.loads(query_result)
    daily_info = weather_info["daily"].items()
    units = weather_info["daily_units"].values()
    list_info = [f"{k}:{v[0]} {unit}" for (k,v),unit in zip(daily_info, units)]
    body_mail = ", ".join(list_info)
    return body_mail

with DAG(dag_id="weather_mail", start_date=datetime(2023, 1, 1),
         schedule="0 0 * * *") as dag_weather:
    submit_query = SimpleHttpOperator(task_id="submit_query",
                                      http_conn_id='', endpoint=weather_query, method="GET",
                                      headers={})
    prepare_email = PythonOperator(task_id='prepare_email',
                                   python_callable=build_body, dag=dag_weather)
    send_email = EmailOperator(task_id="send_email", to="user@example.com",
                               subject="Weather today in Cosenza",
                               html_content="{{ti.xcom_pull('prepare_email')}}")
    
    submit_query >> prepare_email >> send_email
```

## Sensors
- I **Sensors** sono un tipo speciale di operator progettato per **attendere che un evento esterno si verifichi**.
- Due modalità di esecuzione:
    
    - **poke** (predefinita): Il sensor occupa uno slot del worker per tutta la durata dell'attesa.
    - **reschedule**: Il sensor occupa uno slot solo durante il controllo, dormendo tra un controllo e l'altro.
    
    **Quando utilizzare**:
    - Modalità **poke**: Per controlli frequenti (es. ogni secondo).
    - Modalità **reschedule**: Per controlli meno frequenti (es. ogni minuto).

## TaskFlow Tasks
- L'**API TaskFlow** permette di scrivere DAG puliti in Python utilizzando il decoratore `@task`.
- Esempio di TaskFlow con tre task:
    
    - **get_ip**
    - **compose_email**
    - **send_email**
    
    **Funzionamento**:
    - I primi due task (get_ip, compose_email) utilizzano TaskFlow e si passano automaticamente il valore di ritorno.
    - Il terzo task (send_email) è un operator tradizionale ma utilizza il valore di compose_email per impostare i suoi parametri, creando dipendenze automatiche tra i task.


### Esempio di TaskFlow tasks

```python
from airflow.decorators import task
from airflow.operators.email import EmailOperator

@task
def get_ip():
    return my_ip_service.get_main_ip()

@task
def prepare_email(external_ip):
    return {
        'subject': f'Server connected from {external_ip}',
        'body': f'Your server executing Airflow is connected from the external IP {external_ip}<br>'
    }

email_info = prepare_email(get_ip())

EmailOperator(
    task_id='send_email',
    to='example@example.com',
    subject=email_info['subject'],
    html_content=email_info['body']
)
```

---

# Esempio di Programmazione con Apache Airflow

### Applicazione di Ensemble Learning
L'**ensemble learning** è implementato utilizzando le **API TaskFlow** di Airflow. In questo esempio, viene applicata la tecnica del **voto** per costruire un modello ensemble.

#### Fasi del processo:
1. **Dividere il dataset**: Il dataset di input viene suddiviso in un set di addestramento e un set di test.
2. **Costruire i modelli**: Vengono addestrati *n* algoritmi di classificazione in parallelo sul set di addestramento per creare *n* modelli indipendenti.
3. **Classificazione ensemble**: Un sistema di voto combina le previsioni dei modelli, assegnando la classe più votata a ciascuna istanza del set di test.

### Workflow con Airflow
Il workflow è strutturato come una serie di **task Python**, ognuno dei quali rappresenta una fase del processo:

- **load**:
  - Carica il dataset utilizzando la funzione `load_breast_cancer` da `sklearn.datasets`.
  - Restituisce i dati e i target come una tupla.

- **partition**:
  - Riceve il dataset dal task *load*.
  - Divide i dati in set di addestramento e di test.
  - Restituisce i set divisi come un dizionario.

- **train**:
  - Riceve i dati di addestramento e un'istanza di uno stimatore di `sklearn`.
  - Addestra il modello e restituisce il modello addestrato.

- **vote**:
  - Riceve i dati di test e una lista di modelli addestrati.
  - Utilizza i modelli per fare previsioni sul set di test.
  - Le previsioni vengono aggregate tramite un meccanismo di voto per ottenere la classificazione finale.

Questo workflow dimostra come Airflow possa essere utilizzato per gestire flussi di lavoro complessi, come l'addestramento parallelo di modelli e l'aggregazione dei risultati tramite ensemble learning.
#### Codice di esempio in Airflow
```python
# istanziazione del DAG
@dag (
    schedule=None,
    start_date=pendulum.now(),
    catchup=False,
    tags=["example"],
)
def ensemble_taskflow():
    # carica e divide il dataset in set di addestramento e di test
    @task(multiple_outputs=True)
    def partition():
        X, y = load_dataset(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=13
        )
        train_data = (X_train.tolist(), y_train.tolist())
        test_data = (X_test.tolist(), y_test.tolist())
        return {"train": train_data, "test": test_data}

    # addestra un modello di classificazione sui dati di addestramento
    @task
    def train(model: sklearn.base.BaseEstimator, train_data: tuple):
        X_train, y_train = train_data
        model.fit(X_train, y_train)
        model_bytes = pickle.dumps(model)
        model_str = model_bytes.decode("latin1")
        return model_str

    # esegue la classificazione ensemble sui dati di test
    @task
    def vote(test_data: tuple, models: list):
        X_test, y_test = test_data
        pred_sum = np.array([0] * len(X_test))
        for model_str in models:
            model_bytes = model_str.encode("latin1")
            model = pickle.loads(model_bytes)
            pred_sum += model.predict(X_test)
        # la previsione è assegnata alla classe con la maggioranza dei voti
        n_models = len(models)
        threshold = np.ceil(n_models / 2)
        preds = [int(s >= threshold) for s in pred_sum]
        print(f"Accuracy is: {accuracy_score(y_test, preds):.2f}")

# flusso principale
partitioned_dataset = partition()
train_data = partitioned_dataset["train"]
test_data = partitioned_dataset["test"]

# addestra 5 classificatori indipendenti in parallelo
m1 = train(GaussianNB(), train_data)
m2 = train(LogisticRegression(), train_data)
m3 = train(DecisionTreeClassifier(), train_data)
m4 = train(SVC(), train_data)
m5 = train(KNeighborsClassifier(), train_data)

# calcola l'accuratezza tramite votazione sui dati di test
vote(test_data, [m1, m2, m3, m4, m5])

# avvia il DAG
ensemble_taskflow()
```

---
# FAQ su Apache Airflow

## 1. Cos'è Apache Airflow?

Apache Airflow è una piattaforma open-source per lo sviluppo, la pianificazione e il monitoraggio di flussi di lavoro (workflow). In Airflow, questi flussi di lavoro vengono creati come DAG (Directed Acyclic Graph), ossia grafici aciclici diretti, composti da task che definiscono le azioni da eseguire.

## 2. Quali sono i vantaggi di definire i workflow come codice Python in Airflow?

Definire i workflow come codice Python in Airflow offre diversi vantaggi:

- **Gestione del codice sorgente**: I workflow possono essere archiviati in sistemi di controllo di versione come Git, consentendo il rollback a versioni precedenti, il confronto delle modifiche e la collaborazione tra sviluppatori.
- **Sviluppo collaborativo**: Più persone possono lavorare contemporaneamente sullo stesso workflow, semplificando lo sviluppo di progetti complessi.
- **Testabilità**: I workflow Python possono essere sottoposti a test automatizzati per garantirne la correttezza e l'affidabilità.

## 3. Qual è la differenza tra operatori, sensori e task TaskFlow in Airflow?

- **Operatori**: Sono i blocchi di costruzione di base dei DAG di Airflow e rappresentano task predefiniti per azioni comuni come l'esecuzione di comandi, l'invio di email o l'interazione con database.
- **Sensori**: Sono una sottoclasse di operatori specializzati nell'attesa del verificarsi di eventi esterni, come la presenza di un file in un bucket cloud o il superamento di una certa ora.
- **Task TaskFlow**: Sono funzioni Python personalizzate decorate con @task che consentono di scrivere DAG più concisi e leggibili, semplificando il passaggio di dati tra i task.

## 4. Come funziona il meccanismo di scheduling in Airflow?

Airflow utilizza uno scheduler che monitora continuamente i DAG e attiva l'esecuzione dei task in base alle loro dipendenze e alla pianificazione definita. Lo scheduler delega l'esecuzione effettiva dei task a un executor, che può essere configurato per eseguire i task localmente o su un sistema distribuito.

## 5. Cosa sono i provider in Airflow e a cosa servono?

I provider sono pacchetti che estendono le funzionalità di Airflow con operatori, sensori e hook per interagire con servizi esterni specifici, come AWS, Google Cloud Platform o Snowflake. L'utilizzo dei provider semplifica l'integrazione di Airflow con diverse piattaforme e servizi.

## 6. Quali sono i principali componenti di Apache Airflow?

- **Scheduler**: Attiva i workflow programmati e invia i task all'executor.
- **Executor**: Gestisce l'esecuzione dei task, delegandoli ai worker.
- **Webserver**: Fornisce un'interfaccia utente per monitorare, attivare e fare debug di DAG e task.
- **Directory DAG**: Contiene i file DAG letti dallo scheduler, dall'executor e dai worker.
- **Database Metadata**: Archivia lo stato dei task e dei DAG, consentendo ad Airflow di riprendere l'esecuzione in caso di errori o interruzioni.

## 7. Come posso eseguire task in parallelo in Airflow?

Airflow supporta due tipi di parallelismo:

- **Parallelismo dei dati**: Esecuzione dello stesso task su diversi sottoinsiemi di dati, utile per elaborazioni batch su grandi volumi di dati.
- **Parallelismo dei task**: Esecuzione simultanea di task indipendenti, utile per ottimizzare la durata complessiva del workflow.

## 8. Cos'è l'API TaskFlow e quali sono i suoi vantaggi?

L'API TaskFlow è un modo moderno e conciso per scrivere DAG in Airflow utilizzando il decoratore @task. Tra i vantaggi dell'utilizzo di TaskFlow troviamo:

- **Maggiore leggibilità**: I DAG scritti con TaskFlow sono più facili da leggere e comprendere rispetto a quelli scritti con operatori tradizionali.
- **Gestione semplificata delle dipendenze**: TaskFlow gestisce automaticamente le dipendenze tra i task in base al flusso di dati.
- **Migliore organizzazione del codice**: Le funzioni Python decorate con @task possono essere raggruppate in moduli separati, migliorando l'organizzazione del codice.

### Quiz

1. **Quali sono i vantaggi di definire i workflow come codice Python in Apache Airflow?**
2. **Spiega la differenza tra parallelismo dei dati e parallelismo dei task in Airflow.**
3. **Descrivi la funzione dello Scheduler in Apache Airflow.**
4. **Cosa sono gli Operators in Airflow e fornisci un esempio di un Operator comunemente utilizzato.**
5. **In che modo i Sensors differiscono dagli Operators in Airflow?**
6. **Quali sono i vantaggi dell'utilizzo dell'API TaskFlow in Airflow?**
7. **Spiega il concetto di XCom in Apache Airflow e come viene utilizzato.**
8. **Descrivi brevemente il processo di Ensemble Learning.**
9. **Come viene implementato l'Ensemble Learning nell'esempio fornito utilizzando Airflow?**
10. **Quali sono i componenti principali di un'architettura Apache Airflow?**

### Chiave di Risposta del Quiz

1. Definire i workflow come codice Python consente il controllo di versione, la collaborazione tra sviluppatori e la scrittura di test per i workflow.
2. Il parallelismo dei dati esegue lo stesso task su diversi sottoinsiemi di dati, mentre il parallelismo dei task esegue task diversi contemporaneamente.
3. Lo Scheduler in Airflow attiva i workflow in base alla pianificazione definita e invia i task all'executor per l'esecuzione.
4. Gli Operators sono template predefiniti per i task in Airflow. Un esempio comune è il BashOperator, che esegue un comando Bash.
5. I Sensors, a differenza degli Operators, attendono il verificarsi di un evento esterno prima di procedere con l'esecuzione del task successivo nel DAG.
6. L'API TaskFlow semplifica la scrittura dei DAG, consentendo di definire i task come funzioni Python e gestendo automaticamente le dipendenze tra di essi.
7. XCom è un meccanismo che permette ai task di scambiarsi informazioni. Un task può "pushare" un valore XCom e un altro task può "pullarlo" per utilizzarlo.
8. L'Ensemble Learning combina le previsioni di più modelli per ottenere risultati più accurati rispetto a un singolo modello.
9. Nell'esempio, l'Ensemble Learning viene implementato addestrando diversi modelli in parallelo e quindi aggregando le loro previsioni tramite votazione.
10. I componenti principali di Airflow sono: Scheduler, Executor, Webserver, Directory DAG e Database Metadata