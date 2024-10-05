| **Termine**                  | **Definizione**                                                                                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Big Data**                 | Insiemi di dati estremamente grandi e complessi che superano la capacità dei tradizionali metodi di elaborazione dati.                                     |
| **Social Media Analytics**   | L'analisi dei dati dei social media per ottenere informazioni sui comportamenti, le tendenze e le opinioni degli utenti.                                   |
| **PaRSoDA**                  | Parallel Social Data Analytics, una libreria Java che semplifica l'analisi parallela dei dati dei social media.                                            |
| **MapReduce**                | Un modello di programmazione per l'elaborazione parallela di grandi set di dati su cluster di computer.                                                    |
| **Spark**                    | Un framework di elaborazione distribuita open source per l'analisi di big data.                                                                            |
| **COMPSs**                   | Computing Platform for Shared-Memory Systems, un runtime HPC per l'esecuzione di applicazioni parallele.                                                   |
| **Trajectory Mining**        | L'analisi di sequenze di movimenti geospaziali per scoprire schemi di movimento e prevedere comportamenti futuri.                                          |
| **Sentiment Analysis**       | L'uso dell'elaborazione del linguaggio naturale per determinare l'atteggiamento emotivo espresso in un testo.                                              |
| **Region of Interest (ROI)** | Un'area geografica specifica che è di interesse o importanza per un particolare studio o analisi.                                                          |
| **Secondary Sort**           | Tecnica in Hadoop per controllare l'ordine dei valori all'interno di una chiamata alla funzione reduce, utilizzando una chiave composita.                  |
| **Map-side Join**            | Un tipo di operazione di join in Hadoop che viene eseguita durante la fase di mapping, in genere utilizzata per set di dati più piccoli.                   |
| **Reduce-side Join**         | Un tipo di operazione di join in Hadoop che viene eseguita durante la fase di riduzione, adatta per set di dati più grandi.                                |
| **DBSCAN**                   | Density-Based Spatial Clustering of Applications with Noise, un algoritmo di clustering che raggruppa punti dati in base alla loro densità.                |
| **KML**                      | Keyhole Markup Language, un formato di file utilizzato per visualizzare dati geografici in applicazioni di mappatura.                                      |
| **Emoji Sentiment Analysis** | L'analisi del sentiment basata sull'interpretazione delle emoji utilizzate in un testo.                                                                    |
| **Parsoda-py**               | Una versione Python di PaRSoDA che offre un linguaggio di programmazione più accessibile e integrazione con librerie Python come NumPy, Pandas e PyCOMPSs. |


### Motivazioni e Obiettivi
- **Big Data nei Social Media**: I social media come Twitter e Flickr generano enormi quantità di dati, che possono essere sfruttati per **estrarre informazioni utili** su comportamenti e dinamiche umane.
- **Crescita dell'analisi dei social media**: L'analisi dei social media è in rapido sviluppo, con applicazioni in:
  - Analisi dei sentimenti collettivi;
  - Analisi dei movimenti degli utenti (es. luoghi visitati e percorsi);
  - Comprensione del comportamento dei gruppi;
  - Studio delle dinamiche dell'opinione pubblica.
- **Sfide dei Big Data**: L'enorme dimensione e complessità dei dati richiedono **tecniche parallele e distribuite** per l'analisi.
- **Barriere d'ingresso**: Nonostante l'esistenza di framework paralleli come MapReduce e Spark, la loro adozione è limitata dalle **competenze di programmazione** richieste.
- **PaRSoDA**: La libreria **PaRSoDA (Parallel Social Data Analytics)** è stata creata per facilitare la costruzione di applicazioni di analisi dati parallele, rendendo più accessibile l'estrazione di conoscenza dai dati dei social media.

### Caratteristiche di PaRSoDA
- **Accessibilità**: PaRSoDA riduce le competenze di programmazione necessarie per sviluppare applicazioni di analisi dati.
- **Funzionalità avanzate**: Include strumenti per l'elaborazione e l'analisi dei dati dai social media, con particolare attenzione a:
  - Mobilità degli utenti;
  - Sentimenti degli utenti;
  - Tendenze degli argomenti.
- **Supporto ai Big Data**: Le funzioni sono basate sul modello MapReduce e sono progettate per l'esecuzione parallela su sistemi HPC e cloud.
- Le applicazioni PaRSoDA sono eseguite su cluster Hadoop, utilizzando YARN come gestore delle risorse e HDFS come file system distribuito.

### Fasi di PaRSoDA
1. **Acquisizione Dati**: Raccolta parallela di dati dai social media, memorizzati su un file system distribuito (HDFS).
2. **Filtraggio Dati**: Applicazione di funzioni per filtrare i dati raccolti.
3. **Mappatura Dati**: Trasformazione delle informazioni nei dati social media attraverso funzioni di mapping.
4. **Partizionamento Dati**: Suddivisione dei dati in shard, ordinati secondo chiavi primarie e secondarie.
5. **Riduzione Dati**: Aggregazione dei dati per shard utilizzando funzioni di riduzione specifiche.
6. **Analisi Dati**: Esecuzione di funzioni di analisi per estrarre informazioni rilevanti.
7. **Visualizzazione Dati**: Presentazione dei risultati tramite funzioni di visualizzazione personalizzabili.
---
### Come configurare le fasi in PaRSoDA
- PaRSoDA definisce un insieme di metodi che possono essere utilizzati per impostare ciascuna fase.
- Tabella: Principali metodi della classe _SocialDataApplication_.

| Fase                 | Funzione e descrizione                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Acquisizione dati    | `setcrawlers(Class[] functions, String[] params)` <br> Specifica le funzioni di crawling da utilizzare per l'acquisizione dei dati. L'array `functions` contiene le classi di crawling; `params[i]` contiene la stringa di configurazione della funzione `functions[i]`. |
| Filtraggio dati      | `setfilters(Class[] functions, String[] params)` <br> Specifica le funzioni e i parametri associati da utilizzare per eseguire il filtraggio dei dati.                                                                                                                   |
| Mappatura dati       | `setmapfunctions(Class[] functions, String[] params)` <br> Specifica le funzioni e i parametri associati da applicare nella fase di mapping.                                                                                                                             |
| Partizionamento dati | `setpartitioningkeys(String groupkey, String sortkey)` <br> Specifica le chiavi utilizzate per partizionare i dati in shards in base a una chiave primaria (`groupkey`) e ordinarli in base a una chiave secondaria (`sortkey`).                                         |
| Riduzione dati       | `setreducefunction(Class function, String params)` <br> Specifica la funzione e i parametri associati da utilizzare nella fase di riduzione.                                                                                                                             |
| Analisi dati         | `setanalysisfunction(Class function, String params)` <br> Specifica la funzione e i parametri associati da utilizzare per eseguire l'analisi dei dati.                                                                                                                   |
| Visualizzazione dati | `setvisualizationfunction(Class function, String params)` <br> Specifica la funzione e i parametri associati da utilizzare per la visualizzazione dei dati.                                                                                                              |
### Funzioni predefinite in PaRSoDA
- Per ogni fase, PaRSoDA fornisce un set di funzioni predefinite che possono essere eseguite.
- Esempi:
  - Per l'**acquisizione dati**, PaRSoDA fornisce implementazioni di crawling per raccogliere dati da alcuni dei social network più popolari (es. Flickr, Twitter).
  - Per il **filtraggio dati**, PaRSoDA fornisce funzioni per filtrare elementi geotaggati in base alla loro posizione, timestamp, parole chiave contenute, ecc.
    - `isGeoTagged`, `isInPlace`, `containsKeywords`, ...
- Gli utenti sono liberi di **estendere queste funzioni** con le proprie.

### Modello dei metadati in PaRSoDA
Per gestire gli elementi dei social media raccolti da diverse reti sociali, PaRSoDA definisce un modello di metadati per rappresentare i diversi tipi di elementi.

```json
{
  "basic": {
    "source": "twitter",
    "id": "111222333444555",
    "datetime": "2015-12-20T23:20:34.000",
    "location": {
      "lng": -0.1262,
      "lat": 51.5011
    },
    "user": {
      "userId": "12345",
      "username": "joedoe"
    }
  },
  "extra": {
    "inReplyToScreenName": "billsmith",
    "inReplyToUserId": 123456789,
    "hashtags": ["#code", "#mapreduce"],
    "inReplyToStatusId": 678712345678962848,
    "text": "(a billsmith that sounds great!",
    "retweets": 0,
    "isRetweet": false
  }
}
```

Il documento di metadati è composto da due parti:
- Una sezione `basic` che include campi comuni a tutti i social network;
- Una sezione `extra` che contiene campi specifici della sorgente.
---
### Secondary Sort in Hadoop
Il secondary sort è una tecnica che consente al programmatore di MapReduce di **controllare l'ordine** in cui i valori vengono visualizzati all'interno di una chiamata alla funzione reduce.

- Utilizza una chiave composita **[firstKey, secondKey]**.
- Il **partitioner** e il **group comparator** utilizzano solo la **firstKey**:
  - Il **partitioner** assegna tutti i record con la stessa firstKey a un singolo reducer (durante la **fase map**).
  - Dopo la **fase di shuffling**, i record vengono ricevuti dai nodi reducer, **raggruppati** usando la firstKey, e inviati al metodo reduce.
  
### Come eseguire un join in Hadoop
Un'operazione di **join** combina due dataset di grandi dimensioni in MapReduce.
- Se un dataset è più piccolo, viene distribuito a ogni nodo dati del cluster.
- Il mapper o il reducer utilizza il dataset più piccolo per cercare i record corrispondenti nel dataset grande, e li combina per generare i record di output.

### Map-side Join in Hadoop
Il **map-side join** viene eseguito dal mapper **prima** che i dati vengano consumati dalla funzione map.

- I dati devono essere **partizionati** e **ordinati**:
  - Gli input devono essere divisi in un numero uguale di partizioni.
  - Devono essere ordinati per la stessa chiave.
  - Tutti i record per una determinata chiave devono essere nella stessa partizione.
- Viene implementato tramite `CompositeInputFormat`, che può eseguire join su set di dati ordinati e partizionati allo stesso modo.

Il map-side join è generalmente **più veloce** del reduce-side join per piccoli dataset. Tuttavia, se i dati sono preparati con altri job MapReduce, la performance del map-side join diminuisce rispetto a un reduce-side join.

### Come fare una join in Hadoop
Durante la **fase di map**, i mapper leggono i dati dalle tabelle di join A e B.

- I mapper producono coppie **<chiave di join, valore di join>** come dati intermedi.
- Nella **fase di shuffle**, questi dati intermedi vengono ordinati e uniti.
- I **reducer** prendono questi dati ordinati e completano la join.

Il **reduce-side join** viene eseguito dal reducer e non richiede che il dataset sia in una forma strutturata o partizionato.

### Join di Repartizione con Sorting Secondario in Hadoop
Un approccio comune per eseguire una join tra due tabelle di dati in Hadoop è il **join di repartizione con sorting secondario**, che implementa una join lato-reduce.

- Il **join di repartizione** esegue una join relazionale tra due tabelle, A e B.
- Ogni task di map elabora una partizione di A o B. Ogni task emette una **chiave composta** costituita da:
  - Una **chiave di join**.
  - Una **chiave di tabella**.
- La **chiave di join** viene utilizzata durante la partizione per assegnare tuple con la stessa chiave di join allo stesso task di reduce.
- La **chiave di tabella** ordina le tuple, mettendo quelle di A prima di quelle di B.
- Il reducer, per ogni chiave di join, elabora prima le tuple di A, mantenendole in memoria, e poi le tuple di B per eseguire la join.
---
### Parsoda - Applicazione di estrazione delle Regioni di Interesse (ROI)

Un **punto di interesse (POI)** è una località significativa o interessante visitata dagli utenti (ad esempio, una piazza o un monumento).

- Una **regione di interesse (ROI)** rappresenta i confini geografici (un'area) di un POI.

Utilizzando un dataset di elementi geolocalizzati provenienti da social media come Flickr e Twitter, è possibile analizzare informazioni (ad esempio, testo, hashtag, dati geospaziali) per scoprire la ROI di un POI.

---

### Struttura dell'Applicazione di Estrazione delle ROI

1. **Acquisizione dati**:  
   Utilizziamo un file chiamato **"colosseo500m.json"** contenente tutti i post di Flickr pubblicati entro 500 metri dal Colosseo.

2. **Filtro dati**:  
   Dato un set di parole chiave relative al Colosseo (es. "colosseo, coliseo, anfiteatrum flavium"), filtriamo tutti gli elementi che:
   - Contengono almeno una parola chiave del set;
   - Hanno informazioni geospaziali valide.

3. **Mapping dati**:  
   Usiamo un **mapper di identità** predefinito, senza personalizzazioni aggiuntive.

4. **Partizione dati**:  
   La partizione dei dati avviene tramite la funzione predefinita, utilizzando un **approccio round robin** per distribuire i dati tra i nodi reducer.

5. **Riduzione dati**:  
   La funzione di riduzione estrae le **coordinate geografiche** (latitudine/longitudine) da ogni elemento e le restituisce in output.

6. **Analisi dati**:  
   Usiamo l'algoritmo **DBSCAN** per raggruppare i punti geospaziali ottenuti nella fase di riduzione. Questo passaggio consente di identificare un cluster che rappresenta (probabilmente) la **ROI** del Colosseo.

7. **Visualizzazione dati**:  
   Convertiamo il cluster ottenuto nel passaggio precedente in formato **KML**. Questo formato consente di visualizzare la ROI come un **poligono** su piattaforme di mappe come OpenStreetMap o Google Maps.

--- 

```java
public class RoiMiningMain {
    public static void main(String[] args) {
        String colosseumSynonyms = "colasse: colis: collis: collos: amphiteatrum flavium: amphitheatrum flavium + " +
                ": an colasaem: coliseo: coliseo: coliseo de roma: coliseo de roma: coliseu de roma: coliseum" +
                ": coliseum: coliseum: coliseus: colloseum: coloseu: colosseo: colosseo: colosseo: colosseu: colosseum + " +
                ": colosseum: colosseum: colosseum: colosseum: colosseum: colosseum: colosseum: colosseum: colosseum + " +
                ": culusseu: kolezyum: koliseoa: kolize: kolizej s: kolizey: kolizey: koliziej us: kolosej: kolosej + " +
                ": koloseo: koloseo: koloseum: koloseum: koloseum: koloseum: koloseum: koloseum: koloseum: koloseumi + " +
                ": kolosseum: kolosseum: kolosseum: kolosseum";
        
        SocialDataApp app = new SocialDataApp("ROI Mining - Colosseum");
        app.setOutputBasePath("output");
        app.setLocalFileSystem();
        app.setNumReducer(1);
        
        Class[] cFunctions = { FileReaderCrawler.class };
        String[] cParams = { "-i resources/colosseo500m.json" };
        app.setCrawlers(cFunctions, cParams);
        
        Class[] fFunctions = { IsGeoTagged.class, ContainsKeywords.class };
        String[] fParams = { "", "-separator : -keywords " + colosseumSynonyms };
        app.setFilters(fFunctions, fParams);
        
        Class rFunction = ReduceByCoordinates.class;
        String rParams = "-t 5";
        app.setReduceFunction(rFunction, rParams);
        
        Class aFunction = ExtractRois.class;
        String aParams = "-minpts 150 -eps 30";
        app.setAnalysisFunction(aFunction, aParams);
        
        Class vFunction = RoisToKml.class;
        String vParams = null;
        app.setVisualizationFunction(vFunction, vParams);
        
        app.execute();
    }
}
```

### Parsoda - Esempio di funzione di filtro

- Il listato seguente mostra una semplice funzione di filtro per verificare se un elemento di social media ha informazioni geospaziali valide.
- Il codice è molto semplice, trattandosi di una semplice funzione booleana (un predicato).

```java
public class IsGeoTagged extends AbstractFilterFunction {
    public IsGeoTagged() { super(); }
    
    public boolean test(AbstractGeoTaggedItem g) {
        if (g == null || g.getLocation() == null || (g.getLocation().getX() == 0 && g.getLocation().getY() == 0))
            return false;
        return true;
    }
}
```
---

### Parsoda - Altri Casi d'Uso Applicativi

I post sui social media sono spesso accompagnati da **coordinate geografiche** o altre informazioni (es. testo, campi di localizzazione) che permettono di identificare la posizione degli utenti.

- Gli utenti che si spostano tra diverse località producono un'enorme quantità di dati geo-referenziati, utili per comprendere i comportamenti di mobilità.
- Negli ultimi anni, c'è stato un crescente interesse nell'**estrazione di traiettorie** dai dati social geolocalizzati mediante tecniche di **trajectory mining**.
---
Un estratto del codice Parsoda per implementare un'analisi di itemset frequenti.

```java
public class SequentialPatternMain {
    public static void main(String[] args) {
        SocialDataApp app = new SocialDataApp("SPM - City of Rome");
        String[] cFiles = {"romeROIs.kml"};
        app.setDistributedCacheFiles(cFiles);
        Class[] cFunctions = {FlickrCrawler.class, TwitterCrawler.class};
        String[] cParams = {"-lng 12.492 -lat 41.890 -radius 10 -startDate 2014-11-01 -endDate 2016-07-31", 
                            "-lat 12.492 -lng 41.890 -radius 10 -startDate 2014-11-01 -endDate 2016-07-31"};
        app.setCrawlers(cFunctions, cParams);
        Class[] fFunctions = {IsGeoTagged.class, IsInPlace.class};
        String[] fParams = {"", "-lat 12.492 -lng 41.890 -radius 10"};
        app.setFilters(fFunctions, fParams);
        Class[] mFunctions = {FindPOI.class};
        String[] mParams = null;
        app.setMapFunctions(mFunctions, mParams);
        String groupKey = "user.userId";
        String sortKey = "dateTime";
        app.setPartitioningKeys(groupKey, sortKey);
        Class rFunction = ReduceByTrajectories.class;
        String rParams = "-t 5";
        app.setReduceFunction(rFunction, rParams);
        Class aFunction = PrefixSpan.class;
        String aParams = "-maxPatternLength 5 -minSupport 0.005";
        app.setAnalysisFunction(aFunction, aParams);
        Class vFunction = SortPrefixSpanBy.class;
        String vParams = "-k support -d desc";
        app.setVisualizationFunction(vFunction, vParams);
        app.execute();
    }
}
 ...
Class rFunction = ReduceByItemsets.class;
String rParams = "-t 5";
app.setReduceFunction(rFunction, rParams);
Class aFunction = FPGrowth.class;
String aParams = "-minSupport 0.005";
app.setAnalysisFunction(aFunction, aParams);
...
```

| Luogo                  | Supporto | Set di luoghi                                      | Supporto |
| ---------------------- | -------- | -------------------------------------------------- | -------- |
| Colosseo               | 21.7%    | Pantheon, Basilica di San Pietro, Colosseo         | 5.3%     |
| Basilica di San Pietro | 13.9%    | Fontana di Trevi, Basilica di San Pietro, Colosseo | 4.5%     |
| Trastevere             | 8.7%     | Foro Romano, Basilica di San Pietro, Colosseo      | 4.4%     |
| Pantheon               | 6.5%     | Musei Vaticani, Basilica di San Pietro, Colosseo   | 4.4%     |
| Fontana di Trevi       | 5.3%     | Fontana di Trevi, Pantheon, Colosseo               | 4.0%     |
Tabella: Top 5 luoghi visitati a Roma  
#### Valutazione di usabilità

| Fase                 | Parsoda | Hadoop |
| -------------------- | ------- | ------ |
| Main                 | 29      | 71     |
| Acquisizione dati    | 0       | 220    |
| Filtraggio dati      | 30      | 40     |
| Mapping dati         | 26      | 98     |
| Partizionamento dati | 0       | 120    |
| Riduzione dati       | 5       | 63     |
| Analisi dati         | 120     | 120    |
| Visualizzazione dati | 75      | 75     |
| Totale               | 285     | 803    |
**-65% righe di codice**

---
#### Parsoda - Compito
Modificare la funzione di mapping e le funzioni di riduzione per ridurre l'overhead di rete durante la fase di shuffling di MapReduce.
In particolare:
- Funzione di mapping: invece di inviare l'intero elemento geolocalizzato alla fase di riduzione, dovrebbe inviare solo le coordinate <lat, lng> di ogni elemento.
- Funzione di riduzione: deve solo memorizzare i punti su un file senza fare nulla su di essi (cioè, funzione identità).
---
## Parsoda-py: Abilitazione di Runtime HPC basati su Python per l'Analisi Parallela dei Dati

### Panoramica
- **Parsoda-py**: una nuova versione della libreria **Parsoda** sviluppata in Python.
- **Obiettivi**: migliorare l'architettura della libreria e la facilità di utilizzo con un linguaggio più accessibile come Python.
- **Caso di studio**: presentazione di un'applicazione pratica basata su sentiment mining.
### Motivazioni
Il porting di Parsoda in **Python** nasce dall'esigenza di:
- Semplificare la programmazione e aumentare l'usabilità.
- Integrare **runtime aggiuntivi** e utilizzare un vasto ecosistema di librerie Python (es. **NumPy**, **Pandas**).
- Abilitare il supporto alla **tipizzazione dinamica** di Python.

Parsoda-py è una **riprogettazione completa** della libreria Java, adottando un **design pattern bridge** per supportare diversi runtime di esecuzione.

### Versione Basata su Python
Parsoda-py offre quattro driver di esecuzione:
1. **ParsodaSingleCoreDriver**: esegue algoritmi sequenziali su un singolo core.
2. **ParsodaMultiCoreDriver**: utilizza i **thread pool** di Python per eseguire applicazioni multi-core localmente.
3. **ParsodaPySparkDriver**: esegue l'applicazione su un cluster **Spark** tramite **PySpark**.
4. **ParsodaPyCompssDriver**: esegue l'applicazione su un cluster **COMPSs** tramite **PyCOMPSs**.
### Supporto a COMPSs
Parsoda-py è esteso per supportare **COMPSs**, un runtime HPC sviluppato presso il **Barcelona Supercomputing Center (BSC)**, attraverso PyCOMPSs.

- Supporta l'uso di **Distributed Data Set (DDS)** in PyCOMPSs per gestire dataset distribuiti.
### Tipi di Crawler
Parsoda-py offre due tipi di **crawler** per leggere dati da una o più fonti:
1. **Crawler Locale**: legge i dati dal nodo master locale.
2. **Crawler Distribuito**: partiziona la sorgente dati, permettendo la lettura parallela dai nodi worker.

Per ogni crawler, è possibile definire il numero di **partizioni** o la dimensione del **chunk**.

---
### Caso di Studio Applicativo
Il seguente codice esegue un'applicazione di **sentiment mining** per analizzare il sentiment di un set di post sui social media basati sugli **emoji** contenuti:

```python
1 driver = ParsodaPyCompssDriver()
2 app = SocialDataApp("Sentiment Analysis", driver)
3 app.set_crawlers([DistributedFileCrawler("twitter.json", TwitterParser())])
4 app.set_filters([HasEmoji()])
5 app.set_mapper(ClassifyByEmoji("emoji.json"))
6 app.set_reducer(ReduceByEmojiPolarity())
7 app.set_analyzer(TwoFactionsPolarization())
8 app.set_visualizer(PrintEmojiPolarization("emoji_polarization.txt"))
9 app.execute()
```

- L'approccio in **Parsoda-py** consente un codice **conciso**, dove il programmatore dichiara le funzioni per ogni fase di analisi dei dati.
- L'applicazione può essere eseguita su diversi runtime cambiando semplicemente la classe del driver (es. **ParsodaPySparkDriver** per eseguire su Spark).

### Valutazione della Scalabilità
Con un dataset di **180 GB** di tweet e fino a **256 core CPU**, i test dimostrano che il tempo di esecuzione diminuisce in modo significativo all'aumentare del numero di core, con uno **speedup** quasi lineare fino a **64 core**.

### Applicazione di Trajectory Mining
L'esempio seguente mostra l'applicazione di **sequential pattern mining** su COMPSs:

```python
1 driver = ParsodaPyCompssDriver()
2 app = SocialDataApp("Sequential Pattern Mining", driver)
3 app.set_crawlers([DistributedFileCrawler("twitter.json", TwitterParser())])
4 app.set_filters([IsGeoTagged()])
5 app.set_mapper(FindPOI("romeROIs"))
6 app.set_secondary_sort_key(lambda x: x[0])
7 app.set_reducer(ReduceByTrajectories(5))
8 app.set_analyzer(GapBide(0.001, 0, 10))
9 app.set_visualizer(SortGapBide("trajectory_mining.txt", sort_key="support", mode="descending", min_length=3))
10 app.execute()
```
---
# FAQ su PaRSoDA

## 1. Qual è lo scopo principale di PaRSoDA?

PaRSoDA (Parallel Social Data Analytics) è una libreria software progettata per semplificare l'analisi parallela dei big data provenienti dai social media. L'obiettivo principale è rendere più accessibile l'estrazione di informazioni utili da questi dati, anche a utenti con competenze di programmazione limitate.

## 2. Quali sono le caratteristiche principali di PaRSoDA?

- **Accessibilità:** PaRSoDA riduce le competenze di programmazione necessarie per sviluppare applicazioni di analisi dei dati.
- **Funzionalità avanzate:** Offre strumenti per l'elaborazione e l'analisi dei dati social, con particolare attenzione alla mobilità degli utenti, ai sentimenti e alle tendenze degli argomenti.
- **Supporto Big Data:** Le funzioni sono basate sul modello MapReduce e progettate per l'esecuzione parallela su sistemi HPC e cloud.

## 3. Quali sono le fasi principali di un'applicazione PaRSoDA?

1. **Acquisizione dati:** Raccolta parallela di dati dai social media.
2. **Filtraggio dati:** Applicazione di filtri per selezionare i dati rilevanti.
3. **Mappatura dati:** Trasformazione delle informazioni nei dati social.
4. **Partizionamento dati:** Suddivisione dei dati in shard per l'elaborazione parallela.
5. **Riduzione dati:** Aggregazione dei dati per shard.
6. **Analisi dati:** Esecuzione di algoritmi di analisi per estrarre informazioni.
7. **Visualizzazione dati:** Presentazione dei risultati in modo intuitivo.

## 4. Come si configura un'applicazione PaRSoDA?

PaRSoDA definisce un insieme di metodi per impostare ogni fase dell'analisi. Ad esempio, setCrawlers() specifica le funzioni di crawling per l'acquisizione dati, setFilters() definisce i filtri da applicare e così via. Per ogni metodo, è possibile specificare funzioni predefinite o personalizzate.

## 5. Cosa sono i "metadati" in PaRSoDA e come vengono gestiti?

I metadati in PaRSoDA sono informazioni strutturate che descrivono gli elementi dei social media raccolti. PaRSoDA utilizza un modello di metadati flessibile per rappresentare elementi provenienti da diverse piattaforme social. Questo modello include campi comuni a tutti i social network (ad esempio, ID utente, timestamp) e campi specifici della sorgente (ad esempio, hashtag per Twitter, tag geografici per Flickr).

## 6. In che modo PaRSoDA gestisce l'elaborazione di grandi quantità di dati?

PaRSoDA utilizza il modello di programmazione MapReduce per l'elaborazione parallela dei dati. Ciò significa che i dati vengono suddivisi in blocchi più piccoli (shard) e elaborati contemporaneamente su più nodi di un cluster. Questo approccio consente a PaRSoDA di gestire dataset di grandi dimensioni in modo efficiente.

## 7. Quali sono alcuni esempi di casi d'uso di PaRSoDA?

PaRSoDA può essere utilizzato per una varietà di applicazioni di analisi dei dati social, tra cui:

- **Estrazione di regioni di interesse (ROI):** Identificazione di luoghi popolari e punti di riferimento in base ai dati di geolocalizzazione.
- **Analisi dei sentimenti:** Determinazione dell'atteggiamento generale (positivo, negativo o neutro) espresso nei post sui social media.
- **Estrazione di traiettorie:** Individuazione di schemi di movimento degli utenti e creazione di mappe di mobilità.
- **Scoperta di itemset frequenti:** Identificazione di gruppi di elementi che compaiono frequentemente insieme, come ad esempio luoghi visitati in sequenza.

## 8. Quali sono i vantaggi dell'utilizzo di PaRSoDA-py rispetto alla versione Java originale?

PaRSoDA-py, la versione Python di PaRSoDA, offre diversi vantaggi:

- **Maggiore facilità d'uso:** Python è un linguaggio di programmazione più accessibile rispetto a Java, rendendo PaRSoDA-py più facile da imparare e utilizzare.
- **Integrazione con l'ecosistema Python:** PaRSoDA-py può essere facilmente integrato con altre librerie Python popolari per l'analisi dei dati, come NumPy e Pandas.
- **Supporto per runtime aggiuntivi:** PaRSoDA-py supporta diversi runtime di esecuzione, tra cui Spark e COMPSs, consentendo agli utenti di scegliere l'ambiente più adatto alle proprie esigenze.
- **Tipizzazione dinamica:** La tipizzazione dinamica di Python semplifica la scrittura e la manutenzione del codice PaRSoDA-py.

In sintesi, PaRSoDA-py offre una maggiore flessibilità, facilità d'uso ed efficienza rispetto alla versione Java originale, rendendo l'analisi parallela dei dati social accessibile a un pubblico più ampio.

---
### Quiz
1. Quali sono le sfide principali dell'analisi dei Big Data sui social media?
2. In che modo PaRSoDA affronta le barriere d'ingresso nell'analisi parallela dei dati?
3. Descrivere le sette fasi del processo di analisi dei dati in PaRSoDA.
4. Quali sono le differenze tra funzioni predefinite e funzioni personalizzate in PaRSoDA? Fornire esempi.
5. Spiegare come PaRSoDA utilizza un modello di metadati per gestire elementi di social media eterogenei.
6. In che modo il Secondary Sort in Hadoop può essere utilizzato per ottimizzare le operazioni di join?
7. Confrontare e confrontare Map-side Join e Reduce-side Join in Hadoop.
8. Descrivere l'obiettivo e la struttura dell'applicazione di estrazione delle Regioni di Interesse (ROI) in PaRSoDA.
9. Quali sono i vantaggi dell'utilizzo di Parsoda-py rispetto alla versione Java originale?
10. Come Parsoda-py supporta diversi runtime di esecuzione per l'analisi parallela dei dati?

### Chiave di Risposta del Quiz
1. Le sfide principali dell'analisi dei Big Data sui social media includono l'enorme volume, la velocità e la varietà dei dati, che richiedono tecniche di elaborazione parallela e distribuita. Altre sfide includono la necessità di gestire dati non strutturati, la complessità dell'analisi del sentiment e la protezione della privacy degli utenti.
2. PaRSoDA riduce le barriere d'ingresso nell'analisi parallela dei dati fornendo un'interfaccia di programmazione di alto livello che semplifica lo sviluppo di applicazioni di analisi dei dati. Astrae la complessità della gestione di framework paralleli come MapReduce, rendendoli accessibili a un pubblico più ampio.
3. Le sette fasi del processo di analisi dei dati in PaRSoDA sono: (1) Acquisizione Dati, (2) Filtraggio Dati, (3) Mapping Dati, (4) Partizionamento Dati, (5) Riduzione Dati, (6) Analisi Dati e (7) Visualizzazione Dati.
4. Le funzioni predefinite in PaRSoDA sono funzioni pre-costruite per attività comuni, come l'acquisizione di dati da Twitter o il filtraggio di elementi geotaggati. Le funzioni personalizzate sono scritte dagli utenti per implementare funzionalità specifiche dell'applicazione.
5. PaRSoDA utilizza un modello di metadati con sezioni "basic" ed "extra" per rappresentare elementi di social media eterogenei. La sezione "basic" contiene campi comuni a tutti i social network, mentre la sezione "extra" contiene campi specifici della sorgente.
6. Il Secondary Sort in Hadoop può essere utilizzato per ottimizzare le operazioni di join garantendo che tutti i record con la stessa chiave di join vengano inviati allo stesso reducer. Ciò riduce la quantità di dati che devono essere trasferiti durante la fase di shuffle.
7. Map-side Join viene eseguito durante la fase di mapping ed è adatto per piccoli set di dati che possono essere caricati in memoria. Reduce-side Join viene eseguito durante la fase di riduzione ed è più adatto per set di dati più grandi.
8. L'applicazione di estrazione delle Regioni di Interesse (ROI) mira a identificare i confini geografici di un Punto di Interesse (POI) analizzando i dati dei social media geolocalizzati. Estrae coordinate geografiche, raggruppa i punti geospaziali e visualizza la ROI come un poligono su una mappa.
9. Parsoda-py offre un linguaggio di programmazione Python più conciso e accessibile, consentendo agli utenti con diverse competenze di programmazione di utilizzare la libreria. Inoltre, si integra con l'ecosistema Python, fornendo accesso a librerie come NumPy e Pandas.
10. Parsoda-py supporta diversi runtime di esecuzione, inclusi ParsodaSingleCoreDriver, ParsodaMultiCoreDriver, ParsodaPySparkDriver e ParsodaPyCompssDriver. Gli utenti possono passare da un runtime all'altro semplicemente modificando la classe del driver nel codice.

