| **Termine** | **Definizione** |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Stream** | Un flusso continuo di dati potenzialmente infiniti. |
| **Tempo dell'evento** | Il momento in cui un evento si verifica nel mondo reale. |
| **Tempo di ingestione** | Il momento in cui un sistema riceve i dati di un evento. |
| **Tempo di elaborazione** | Il momento in cui un sistema elabora i dati di un evento. |
| **Serie temporale** | Una sequenza di dati ordinati temporalmente, spesso raccolti a intervalli regolari. |
| **Modello di cassa** | Un modello di dati in cui i valori possono essere solo incrementati (ad esempio, il numero di vendite). |
| **Modello di tornello** | Un modello di dati in cui i valori possono essere sia incrementati che decrementati (ad esempio, un saldo bancario). |
| **Algoritmo di streaming** | Un algoritmo progettato per elaborare dati in streaming con memoria e tempo di elaborazione limitati. |
| **Elaborazione agnostica del tempo** | Elaborazione di dati in streaming senza considerare il tempo degli eventi. |
| **Elaborazione approssimativa** | Tecniche che forniscono risposte approssimative ai calcoli sui dati in streaming, spesso utilizzando strutture dati probabilistiche. |
| **Finestra temporale** | Un intervallo di tempo definito per l'elaborazione di un sottoinsieme di dati in streaming. |
| **Watermark** | Un marcatore temporale utilizzato nei sistemi di elaborazione di stream per indicare che tutti i dati fino a un determinato tempo sono stati ricevuti. |
| **DAG (Direct Acyclic Graph)** | Un grafo diretto senza cicli, utilizzato per rappresentare le dipendenze tra le operazioni in un sistema di elaborazione di stream. |
| **Apache Storm** | Una piattaforma open source per l'elaborazione di stream di dati in tempo reale. |
| **Topologia (Storm)** | Un grafo diretto aciclico che definisce il flusso di dati e le operazioni in un'applicazione Storm. |
| **Spout (Storm)** | La sorgente dei dati in una topologia Storm, responsabile della lettura e dell'emissione dei dati. |
| **Bolt (Storm)** | Un nodo di elaborazione in una topologia Storm, responsabile dell'esecuzione di operazioni sui dati ricevuti da uno o più spout o altri bolt. |
| **Nimbus (Storm)** | Il nodo master in un cluster Storm, responsabile della distribuzione del codice, dell'assegnazione dei task e del monitoraggio del cluster. |
| **Supervisor (Storm)** | Un nodo worker in un cluster Storm, responsabile dell'esecuzione dei task assegnati da Nimbus. |
| **ZooKeeper** | Un servizio di coordinamento distribuito utilizzato da Storm per gestire lo stato del cluster e la comunicazione tra i nodi. |
| **Mini-batch** | Una tecnica per elaborare i dati in streaming suddividendoli in piccoli batch e elaborando ogni batch come un'unità indipendente. |

## Cos'è uno Stream?

- Uno **stream** è un flusso continuo di dati non limitati (potenzialmente infiniti). A differenza dei dati limitati, che sono finiti e statici, i dati non limitati crescono continuamente e devono essere elaborati man mano che vengono prodotti. Il modello di streaming opera con un **modello push**, in cui la fonte di dati controlla il flusso e l’elaborazione, spesso in un contesto **pubblica/sottoscrivi**.
- Il concetto di **tempo** è centrale in questi sistemi e può riferirsi a diversi momenti chiave: **tempo dell'evento** (quando l'evento si verifica), **tempo di ingestione** (quando i dati sono ricevuti), e **tempo di elaborazione** (quando i dati vengono effettivamente processati).

#### Tempo dell'evento vs. Tempo di elaborazione

- Il **tempo dell'evento** rappresenta il momento in cui un dato è stato prodotto (ad esempio, quando un utente compie un'azione).
- Il **tempo di ingestione** è quando i dati vengono ricevuti dal sistema.
- Il **tempo di elaborazione** è quando il sistema elabora quei dati.
 - È importante notare che questi tempi spesso non coincidono: un evento potrebbe verificarsi molto prima di essere elaborato dal sistema.

#### Serie Temporali

- Una **serie temporale** è un insieme di dati ordinati temporalmente. Questi dati vengono raccolti a intervalli regolari, formando una sequenza che descrive l'evoluzione di un fenomeno nel tempo.

#### Modelli di Cassa e Tornello

- Nel **modello di cassa**, i dati vengono aggiornati solo con valori positivi (es., incremento di vendite).
- Nel **modello di tornello**, i dati possono essere aggiornati con incrementi sia positivi che negativi (es., un saldo bancario che può aumentare o diminuire).

#### Algoritmi di Streaming

- Gli **algoritmi di streaming** sono progettati per processare flussi di dati in cui ogni elemento del flusso può essere esaminato una sola volta, poiché il volume di dati è troppo grande per essere conservato interamente in memoria. Questi algoritmi lavorano con risorse limitate (memoria e tempo di elaborazione).

#### Approcci all'Elaborazione di Stream

1. **Elaborazione agnostica del tempo**: Il tempo non è considerato rilevante (es. filtraggio o join interni).
2. **Elaborazione approssimativa**: Utilizza tecniche di stima per ottenere risultati sufficientemente accurati con risorse limitate.
3. **Finestre temporali**: Definiscono intervalli specifici di tempo per raggruppare gli eventi, basati su:
 - **Tempo di elaborazione**: Finestra definita in base a quando i dati vengono elaborati.
 - **Tempo dell'evento**: Finestra basata sul tempo in cui gli eventi si verificano.

#### Elaborazione Agnostica del Tempo

- In questo tipo di elaborazione, il tempo non è un fattore determinante, ed è usato in situazioni dove l'ordine temporale degli eventi non è importante, come nel **filtraggio** di dati o nell'**inner join** (unire dataset su un campo comune).

## Elaborazione Agnostica del Tempo

#### Esempio di Filtraggio

- **Obiettivo**: Filtrare log di traffico web per includere solo quelli provenienti da un dominio specifico.
- **Metodo**: Per ogni record che arriva, verifichiamo se appartiene al dominio desiderato e lo scartiamo se non lo è. Poiché ogni elemento viene processato singolarmente, il tempo in cui arriva non influisce sull'elaborazione. 
- **Conclusione**: Il tempo dell'evento non è rilevante; l'importante è la presenza o meno di un dominio specifico.

#### Esempio di Inner Join

- **Obiettivo**: Unire due fonti di dati non limitate.
- **Metodo**: Quando un valore arriva da una fonte, viene temporaneamente memorizzato. Quando un valore corrispondente arriva dall'altra fonte, i dati vengono uniti. La logica di join non dipende dal tempo di arrivo dei dati, ma solo dal fatto che entrambi i dati siano presenti.
- **Conclusione**: Anche in questo caso, il tempo non è un fattore rilevante per l’elaborazione.

## Elaborazione Approssimativa

- **Definizione**: Si utilizzano algoritmi che forniscono risposte approssimative basate su un riepilogo dei dati ("sketch"), anziché analizzare ogni elemento singolarmente.
- **Esempi**: 
 - **Top-N approssimativo**: Trova gli N elementi più frequenti senza dover esaminare tutti i dati.
 - **K-means in streaming**: Algoritmo di clustering che aggiorna i centri in modo incrementale man mano che nuovi dati vengono ricevuti.

## Finestre Temporali (Windowing)

- Le finestre temporali suddividono i flussi di dati in blocchi finiti per l’elaborazione. Esistono diversi tipi di finestre:

1. **Finestre Fisse**: Intervalli di tempo predefiniti, es. 5 minuti.
2. **Finestre Scorrevoli**: Finestre che si sovrappongono, es. ogni minuto ma analizzano i dati degli ultimi 5 minuti.
3. **Sessioni**: Finestre che raggruppano eventi che fanno parte della stessa sessione logica, ad esempio eventi consecutivi di un utente.

#### Finestre per Tempo di Elaborazione

- Il sistema accumula i dati in arrivo per un determinato intervallo di tempo di **elaborazione** e poi li invia per essere processati.
- **Esempio**: Accumulare dati per 10 minuti di tempo di elaborazione e poi processarli.

#### Finestre per Tempo dell'Evento

- Si basano sul **tempo dell'evento**, ovvero quando gli eventi si verificano, e richiedono di bufferizzare i dati per tener conto di eventuali ritardi.
- **Problema di completezza**: Non sempre è chiaro quando abbiamo ricevuto tutti i dati relativi a un intervallo temporale, causando difficoltà nell’elaborazione accurata.

## Operatori di Base per lo Streaming

1. **Aggregazione a Finestre**: Calcola aggregati su finestre temporali, ad esempio:
 - Velocità media
 - Numero di accessi a un URL in un certo periodo
2. **Join a Finestre**: Unisce i dati basati su una finestra temporale. Utilizzato per correlare eventi che si verificano nello stesso intervallo di tempo, come ad esempio:
 - Temperatura registrata in diverse stazioni meteorologiche nello stesso intervallo di tempo.

### Elaborazione di Eventi Complessi

- **Obiettivo**: Rilevare pattern o sequenze di eventi all'interno di uno stream di dati.
- **Evento complesso**: Una sequenza definita da condizioni logiche (es. valori dei dati) e temporali (es. avvenimenti entro un certo intervallo di tempo).

Esempio:
```
24°C, Stazione#1, 13:00
23°C, Stazione#2, 13:00
21°C, Stazione#1, 13:02
20°C, Stazione#1, 13:05
```
Pattern da rilevare:
```text
SEQ(A, B, C) CON
A.Temp > 23°C &&
B.Stazione = A.Stazione && B.Temp < A.Temp&&
C.Stazione = A.Stazione && A.Temp - C.Temp > 3
```

Questo pattern rileva una sequenza di eventi in cui la temperatura diminuisce successivamente alla Stazione#1.

- Gli eventi complessi possono essere costruiti usando operatori logici come:
 - **SEQ**: Definisce una sequenza di eventi.
 - **AND, OR, NEG**: Condizioni logiche. 
---
### Requisiti dello Streaming di Big Data

1. **Mantenere i dati in movimento**: Architettura che gestisce flussi di dati costanti.
2. **Accesso dichiarativo**: Linguaggi di alto livello come StreamSQL.
3. **Gestione delle imperfezioni**: Gestione di dati in ritardo, mancanti o non ordinati.
4. **Risultati prevedibili**: Coerenza e utilizzo corretto del tempo dell'evento.
5. **Integrazione dati stream e batch**: Combinare elaborazioni su dati storici e in tempo reale.
6. **Sicurezza e disponibilità**: Gestire guasti e assicurare la persistenza dei dati.
7. **Partizionamento e scalabilità**: Distribuzione dei dati e calcolo su larga scala.
8. **Elaborazione istantanea**: Risposta immediata agli eventi in arrivo.
---
### Elaborazione di Big Data

- **Database tradizionali**: Hanno sempre elaborato grandi quantità di dati, ma non sono adatti ai big data perché:
 - **Dati strutturati**: I database funzionano bene con dati strutturati.
 - **Big data**: Spesso non sono completamente strutturati e richiedono tecniche più flessibili rispetto a semplici operazioni di selezione, proiezione e join.

### Prima Soluzione: MapReduce (MR)

- **Vantaggi di MR**: Funziona bene con grandi quantità di dati statici.
- **Limitazioni per gli stream**: MR non è adatto agli stream continui, ma solo a finestre di dati molto grandi.
- **Problema**: I dati non si muovono tra le fasi di elaborazione, creando alta latenza e bassa efficienza.

## Mantenere i Dati in Movimento

#### Stream Discretizzati (Mini-batch)

- **Idea**: Eseguire elaborazioni batch su piccoli blocchi di dati raccolti nel tempo.
- **Pseudo-codice**:
  ```java
  while (true) {
    // Ottieni i prossimi pochi record
    // Esegui un calcolo batch
  }
  ```
- **Flusso**:
  ```
  Stream → Discretizzatore → Job → Job → Job → ...
  ```

#### Streaming Nativo

- **Esempio**:
  ```java
  while (true) {
    // Elabora il prossimo record
  }
  ```
- **Flusso**:
  ```
  Operatori di lunga durata → Flusso continuo di dati
  ```

- **Vantaggi**: Facile da implementare, coerenza e tolleranza ai guasti ben gestibili.
- **Svantaggi**: Difficoltà a gestire eventi con tempi specifici e sessioni.

---
### Architettura di Streaming Vera

- **Struttura**: Un programma di streaming è rappresentato come un **DAG (Direct Acyclic Graph)** di operatori e stream intermedi.
 - **Operatore**: Unità di calcolo che può mantenere uno stato.
 - **Stream intermedi**: Flussi logici di record passati tra operatori.

| **Trasformazioni di Stream** | **Descrizione** |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Trasformazioni di base | Operazioni standard di elaborazione su flussi di dati, come Map, Reduce, Filter e Aggregazioni. |
| Trasformazioni binarie | Operazioni binarie che combinano due flussi di dati, come CoMap e CoReduce. |
| Finestre flessibili | Le finestre temporali possono essere definite su intervalli di tempo, conteggi di elementi o variazioni di dati (Delta). |
| Operatori binari temporali | Unioni di flussi di dati basati su correlazioni temporali, come Join e Cross. |
| Iterazioni | Supporto nativo per iterazioni, che permettono di eseguire calcoli ripetuti su dati in arrivo. |

## Gestire le Imperfezioni - Watermark

- **Problema**: I dati possono arrivare in anticipo, in orario o in ritardo rispetto al tempo previsto.
- **Soluzione**: L'uso di **watermark**, che tracciano il progresso temporale degli eventi durante l'elaborazione.
 - **Definizione**: Un watermark è una funzione \( F(P) \to E \), dove:
 - \( P \): Punto nel tempo di elaborazione.
 - \( E \): Punto nel tempo dell'evento.
 - **Funzione**: Indica che il sistema crede di aver ricevuto tutti i dati con tempi di evento inferiori a \( E \). In altre parole, segnala che non ci si aspetta più dati precedenti a questo punto temporale.

### Watermark Perfetti vs Euristici

- **Watermark Perfetti**: Possibile se si ha conoscenza completa dei dati in ingresso, con tutti i dati in anticipo o in orario.
- **Watermark Euristici**: Utilizzano stime basate su informazioni parziali, ideali quando una conoscenza perfetta dei dati non è praticabile.
### Lezioni Apprese dal Batch

- **Batch Processing**:
 - Se un calcolo batch fallisce, è possibile ripetere l'operazione come una transazione, mantenendo la coerenza.
 - Il tasso di transazione è costante, il che semplifica la gestione degli errori.
 - **Domanda**: Possiamo applicare questi principi di affidabilità a una vera esecuzione in streaming?

### Creazione di Snapshot - Approccio Naive

- **Processo di Snapshot**:
 - Pausare l'esecuzione su determinati punti temporali $t_1, t_2, \dots.$
 - Raccogliere lo stato attuale del sistema (es. lo stato della memoria o dei calcoli).
 - Ripristinare l'esecuzione una volta terminata la raccolta dello snapshot.

### Partizionamento e Scalabilità Automatici

- **Tre Tipi di Parallelizzazione**:
 - I grandi sistemi di streaming devono supportare diversi tipi di parallelizzazione per bilanciare carico, distribuzione e prestazioni.
 - L'architettura deve garantire che tutti i tre tipi siano supportati per migliorare la scalabilità e l'efficienza.

## Apache Storm

- **Descrizione**: Piattaforma scalabile per l'elaborazione di flussi di dati in tempo reale.
 - **Calcolo basato su tuple**: Le unità di elaborazione sono tuple, che rappresentano elementi di dati.
 - **Topologia**: Un programma Storm è strutturato come un **grafo di topologia**, dove:
 - **Vertici**: Rappresentano i nodi di calcolo o le trasformazioni dei dati.
 - **Archi**: Rappresentano i flussi di dati tra i nodi.
 - **Stream**: Una sequenza illimitata di tuple (flussi di dati) che vengono processati continuamente.

- **Motore di Basso Livello**: È un sistema flessibile, che permette di gestire calcoli personalizzati e adattarsi a diverse architetture di elaborazione.

### Vantaggi di Apache Storm

- **Gratuito, semplice e open source**: Facile accessibilità per sviluppatori.
- **Multilinguaggio**: Può essere utilizzato con qualsiasi linguaggio di programmazione.
- **Velocità e Scalabilità**: Molto veloce e progettato per scalare orizzontalmente.
- **Tolleranza ai guasti**: Garantisce che i dati vengano elaborati correttamente anche in caso di errori.
- **Robustezza**: Si integra bene con molte tecnologie di database ed è robusto nel gestire grandi carichi di lavoro.

### Storm vs Hadoop

- **Somiglianze**:
 - Entrambi i sistemi utilizzano cluster per l'elaborazione distribuita.

- **Differenze**:
 - **Hadoop** esegue job **MapReduce**, che hanno una fine definita.
 - **Storm** esegue **topologie**, che elaborano dati in modo continuo e teoricamente senza fine.

### Modello Dati di Storm

- **Tupla**: L'unità base di dati in Storm, composta da un elenco di campi.
 - Ogni campo può avere diversi tipi di dati come byte, integer, float, ecc.
 - È possibile definire tipi di dati personalizzati tramite l'API di Storm.
 - **Accesso ai campi**: Puoi accedere ai valori della tupla tramite:
 - **Nome del campo**: `getValueByField(String)`
 - **Indice posizionale**: `getValue(int)`

### Concetti di Storm: Stream e Spout

- **Stream**: Una sequenza illimitata di tuple, rappresentata come coppie chiave-valore, ad esempio `<"UIUC", 5>`.
- **Spout**: La sorgente dei dati di uno stream. Ad esempio:
 - Una **API Twitterhose** può fungere da spout, emettendo un flusso continuo di tweet.

## Componenti di una Topologia Storm

#### Spout

- **Funzione**: Uno spout è la fonte di dati in una topologia Storm. Esso si occupa di:
 - Leggere o ascoltare dati da una fonte esterna (ad esempio, una coda o un API).
 - Pubblicare (emissione) i dati sotto forma di stream.
- Uno spout può emettere più stream contemporaneamente.
- **Metodi importanti**:
 - `nextTuple()`: Legge i prossimi dati disponibili e li emette come tuple nello stream.
 - `open()`: Configura lo spout all'inizio della topologia (ad esempio, per connettersi a una fonte esterna).

#### Bolt

- **Funzione**: I bolt sono i nodi che elaborano i dati ricevuti dallo spout o da altri bolt, eseguendo trasformazioni, filtri, unioni, ecc. Ogni bolt può:
 - Ricevere dati (tuple) da uno o più stream.
 - Applicare trasformazioni o elaborazioni ai dati.
 - Emissione di nuovi stream di output.
- I bolt possono essere concatenati per eseguire trasformazioni più complesse.
- **Metodi importanti**:
 - `execute(Tuple input)`: Viene eseguito per ogni tupla ricevuta. Contiene la logica di elaborazione.
 - `prepare(Map stormConf, TopologyContext context, OutputCollector collector)`: Configura il bolt, ad esempio per inizializzare risorse o settaggi necessari.

### Definizione di una Topologia Storm

- **Cos'è una Topologia**: È un'astrazione di un grafo diretto aciclico (DAG) che rappresenta il flusso di elaborazione dei dati. La topologia Storm viene distribuita su un cluster per elaborare i dati in tempo reale.
 - **Spout**: Fonte dei dati. Può leggere tuple da una coda ed emetterle come stream.
 - **Bolt**: Nodo che trasforma/elabora i dati. Consuma un certo numero di stream di input, li processa e può emettere nuovi flussi(stream).
- Ogni nodo (spout o bolt) viene eseguito in parallelo, migliorando l'efficienza di elaborazione.

### Architettura di Apache Storm

- **Processi Worker**: Ogni macchina in un cluster Storm può eseguire uno o più processi worker, che gestiscono task di spout e bolt.
- **Executor**: Sono i thread che eseguono le operazioni della topologia.
- **Task**: Sono le unità di lavoro gestite dagli executor.

### Componenti di Storm

1. **Nimbus**
 - È il **master** del cluster Storm, con responsabilità come:
 - Distribuire il codice dell'applicazione ai nodi worker.
 - Assegnare task ai worker.
 - Monitorare i task e riavviarli in caso di errori.
 - È **stateless** (senza stato), quindi memorizza tutte le informazioni critiche in **ZooKeeper**.
 - Solo un singolo nodo Nimbus è attivo in un cluster, e può essere riavviato senza interrompere i task in esecuzione.

2. **Supervisor**
 - Sono i nodi worker che eseguono i task assegnati da Nimbus.
 - Ogni nodo supervisor gestisce la creazione, l'avvio e la chiusura dei processi worker.
 - Anche il supervisor è **fail-fast** e memorizza il proprio stato in ZooKeeper.

3. **ZooKeeper**
 - ZooKeeper coordina il cluster distribuendo informazioni di configurazione tra Nimbus e i nodi supervisor.
 - Nimbus e i supervisor comunicano solo tramite ZooKeeper, garantendo che il sistema continui a funzionare anche se uno dei componenti viene temporaneamente interrotto.
 - Poiché tutti i dati di stato vengono memorizzati in ZooKeeper, è possibile riavviare Nimbus o un supervisor senza perdita di dati o di stato.

### Storm vs Hadoop

- **Storm**: Elabora dati in modo **continuo** e in tempo reale, tramite topologie.
- **Hadoop**: Esegue job **MapReduce**, che hanno una durata finita e terminano una volta completato il calcolo.
## Architettura

```
Processo Worker
┌───────────────┐
│    Task       │
│    Task       │
├───────────────┤
│    Task       │
│    Task       │
├───────────────┤
│    Task       │
│    Task       │
└───────────────┘
```

Una topologia in esecuzione consiste in molti processi worker distribuiti su molte macchine.

### Tolleranza ai guasti

- I **worker** inviano segnali di "heartbeat" (controllo di attività) a Nimbus tramite **ZooKeeper**.
- Quando un worker smette di funzionare, il **supervisor** lo riavvierà.
- Se un nodo supervisor smette di funzionare, **Nimbus** riassegnerà il lavoro ad altri nodi.
- Se **Nimbus** smette di funzionare, le topologie continueranno a funzionare normalmente, ma non sarà possibile eseguire nuove riassegnazioni dei task.
 - Questo è diverso da Hadoop, dove se il **JobTracker** smette di funzionare, tutti i job in esecuzione vengono persi.
- È preferibile eseguire ZooKeeper con almeno **3 nodi**, in modo da poter tollerare il fallimento di 1 server ZooKeeper.

## Una Topologia di Conteggio Parole di Esempio

### APACHE STORM

```
Sentence     Split       Word
Spout    →   Sentence →  Count   →  Report
              Bolt       Bolt       Bolt
```

- Sentence Spout: `{ "sentence": "my dog has fleas" }`
- Split Sentence Bolt:
  ```
  { "word": "my" }
  { "word": "dog" }
  { "word": "has" }
  { "word": "fleas" }
  ```
- Word Count Bolt: `{ "word": "dog", "count": 5 }`
- Report Bolt: stampa il contenuto

## Codice di Esempio per il Conteggio Parole

### APACHE STORM

```java
public class SentenceSpout extends BaseRichSpout {
    private SpoutOutputCollector collector;
    private String[] sentences = {
        "my dog has fleas", "i like cold beverages", 
        "the dog ate my homework", "dont have a cow man", 
        "i don't think i like fleas"
    };
    private int index = 0;

    public void open(Map config, TopologyContext context, SpoutOutputCollector collector) {
        this.collector = collector;
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("sentence"));
    }

    public void nextTuple() {
        this.collector.emit(new Values(sentences[index]));
        index++;
        if (index >= sentences.length) {index = 0;}
    }
}
```

```java
public class SplitSentenceBolt extends BaseRichBolt {
    private OutputCollector collector;

    public void prepare(Map config, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    public void execute(Tuple tuple) {
        String sentence = tuple.getStringByField("sentence");
        String[] words = sentence.split(" ");
        for(String word : words){
            this.collector.emit(new Values(word));
        }
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("word"));
    }
}
```

```java
public class WordCountBolt extends BaseRichBolt {
    private OutputCollector collector;
    private HashMap<String, Long> counts = null;

    public void prepare(Map config, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
        this.counts = new HashMap<String, Long>();
    }

    public void execute(Tuple tuple) {
        String word = tuple.getStringByField("word");
        Long count = this.counts.get(word);
        if(count == null) {
            count = 0L;
        }
        count++;
        this.counts.put(word, count);
        this.collector.emit(new Values(word, count));
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("word", "count"));
    }
}
```

```java
public class ReportBolt extends BaseRichBolt {
    private HashMap<String, Long> counts = null;

    public void prepare(Map config, TopologyContext context, OutputCollector collector) {
        this.counts = new HashMap<String, Long>();
    }

    public void execute(Tuple tuple) {
        String word = tuple.getStringByField("word");
        Long count = tuple.getLongByField("count");
        this.counts.put(word, count);
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        // questo bolt non emette nulla
    }

    public void cleanup() {
        List<String> keys = new ArrayList<String>(this.counts.keySet());
        Collections.sort(keys);
        for (String key : keys) {
            System.out.println(key + ": " + this.counts.get(key));
        }
    }
}
```
