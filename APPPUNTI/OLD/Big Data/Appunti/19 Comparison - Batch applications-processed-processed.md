
# Confronto tra strumenti di programmazione: applicazioni batch

### Applicazioni batch: Spark vs. Hadoop

Viene discussa un'applicazione per la scoperta automatica di pattern di mobilità utente da post di Flickr geotaggati generati nella città di Roma. L'applicazione mira a scoprire le traiettorie utente più frequenti attraverso specifiche località o aree di interesse per l'analisi, comunemente denominate **punti di interesse** (PoI).

Un PoI è una località considerata utile o interessante, come un'attrazione turistica o un'attività commerciale. Poiché le informazioni su un PoI sono generalmente limitate a un indirizzo o a coordinate GPS, è difficile abbinare le traiettorie ai PoI. Pertanto, è spesso utile definire le cosiddette **aree di interesse** (RoI) che rappresentano i confini delle aree dei PoI.

Una traiettoria può essere definita come una sequenza di RoI, che rappresenta uno schema di movimento nel tempo. Una **traiettoria frequente** è una sequenza di RoI frequentemente visitate dagli utenti.

### Implementazione con Apache Spark

Dopo aver raccolto un set di post geotaggati da Flickr, viene applicata una pre-elaborazione per filtrare i post e mappare ogni post geotaggato a una RoI. Quindi, viene applicata l'estrazione di traiettorie per estrarre pattern di mobilità frequenti nelle traiettorie degli utenti attraverso le RoI, al fine di comprendere meglio come le persone si muovono nella città di Roma.

I post sui social media spesso contengono informazioni sulla posizione o altri metadati utili:

* una descrizione testuale
* un insieme di parole chiave associate al post
* una coppia di latitudine/longitudine del luogo in cui il post è stato creato
* un ID che identifica l'utente che ha creato il post
* un timestamp che indica la data di creazione del post

![[|464](_page_2_Figure_10.jpeg)]

Si inizia con un'implementazione dell'applicazione utilizzando **Apache Spark** e il linguaggio Scala. Vengono definite due classi, **SingleTrajectory** e **UserTrajectory**, per rappresentare rispettivamente un singolo punto in una traiettoria (coppia `<PoI, timestamp>`) e un utente in una data data (coppia `<user_id, date>`). Queste classi definiscono i metodi *equals* e *hashCode* per il confronto tra traiettorie e utenti.

Il metodo principale legge il dataset di input JSON in un **DataFrame** Spark e applica funzioni per filtrare i punti dati non rilevanti (coordinate GPS non valide o località fuori Roma).

Il codice utilizza la classe **KMLUtils** per leggere un file Keyhole Markup Language (KML) contenente le coordinate di alcuni PoI a Roma e convertirlo in una mappa Scala dove le chiavi sono i nomi dei PoI (es., *colosseum*, *vaticanmuseums*, *mausoleumofhadrian*) e i valori sono le coordinate che definiscono i confini dell'area come poligono.

Il programma calcola le traiettorie degli utenti usando l'algoritmo **FPGrowth** e restituisce gli itemset frequenti e le regole di associazione che rappresentano le traiettorie ottenute.

#### Forme KML caricate:

```
Map(colosseum -> POLYGON((12.490838 41.891314,12.490249 41.890018,12.490204 41.889794,12.490321 41.889566,12.490494 41.889354,12.490967 41.889194,12.491686 41.88922,12.492165 41.889213,12.492505 41.88919,12.492683 41.889199,12.49333 41.88923,12.493665 41.88934,12.493783 41.889477,12.494041 41.88977,12.494276 41.89039,12.49422 41.89073,12.493732 41.890963,12.49302 41.891225,12.492193 41.89145,12.491616 41.891608,12.490971 41.891729,12.490908 41.89152,12.490838 41.891314)),
vaticanmuseums -> POLYGON((12.455288 41.906637,12.454548 41.906892,12.451833 41.906366,12.451571 41.905994,12.454205 41.902972,12.455181 41.903076,12.455288 41.906637)),
piazzacolonna -> POLYGON((12.479552 41.900674,12.47962 41.900517,12.48023 41.900682,12.480522 41.900758,12.480407 41.90106,12.480311 41.901285,12.479823 41.901185,12.479384...
piazzadelpopolo -> POLYGON((12.47534 41.911012,12.475206 41.910665,12.475384 41.910282,12.475788 41.91008,12.476215 41.909986,12.476639 41.910022,12.477036 41.910162,12.477325 41.910597,12.47739 41.91098,12.477296 41.911138,12.477138 41.911248,12.476891 41.911424,12.47655 41.911503,12.476038 41.911527,12.475716 41.911397,12.47534 41.911012))
```

#### Lettura del dataset JSON in un DataFrame Spark:

![[Pasted image 20250223180024.png]]

# Estrazione di Regole di Associazione da Traiettorie GPS: Confronto tra Spark e Hadoop

Questo documento descrive l'estrazione di regole di associazione da dati di traiettorie GPS, implementata inizialmente con Spark e successivamente con Hadoop MapReduce. L'analisi si basa su dati Flickr, filtrati e processati per estrarre informazioni rilevanti.

## Fase 1: Filtraggio dei Dati (Spark e Hadoop)

Sia l'implementazione Spark che quella Hadoop iniziano con una fase di filtraggio dei dati. Vengono applicati i seguenti filtri:

1. **Validità delle Coordinate GPS:** Vengono mantenuti solo i dati con longitudine e latitudine correttamente definite.
2. **Posizione Geografica:** Vengono mantenuti solo i post pubblicati a Roma. La funzione `filterIsInRome`, che sfrutta la libreria Java **Spatial4j** tramite la classe di utilità **GeoUtils**, verifica l'appartenenza di un punto ad un poligono definito.

Ulteriori filtri vengono applicati per accuratezza, latitudine, longitudine, `owner.id` e `dateTaken`.

## Fase 2: Estrazione di Regole di Associazione con FPGrowth (Spark)

Dopo il filtraggio, l'algoritmo **FPGrowth** (da **MLlib**) viene utilizzato per l'estrazione delle regole di associazione. Il metodo `mapCreateTrajectory` trasforma i dati in tuple `<UserTrajectory, SingleTrajectory>`, dove `SingleTrajectory` contiene l'ID utente e i PoI visitati.

Il metodo `computeTrajectoryUsingFPGrowth` prepara i dati di transazione come `RDD[Array[String]]`, rimuove le traiettorie vuote, raggruppa quelle dello stesso utente e le trasforma in un insieme di itemset unici usando la funzione `distinct`. Questo insieme viene passato a FPGrowth con parametri `minSupport` e `minConfidence` per generare itemset frequenti e regole di associazione.

**Esempio di dati di transazione preparati (`prepareTransaction`):**

```
(user 35716709@N04, day Nov 20, 2016,Set(poi villaborghese, timestamp Nov 20, 2016))
(user 61240032@N05, day Nov 29, 2016,Set(poi stpeterbasilica, timestamp Nov 29, 2016, poi colosseum, timestamp Nov 29, 2016, poi stpeterbasilica, timestamp Nov 29, 2016))
(user 99366715@N00, day Nov 30, 2016,Set(poi piazzanavona, timestamp Nov 30, 2016))
(user 52714236@N02, day Nov 30, 2016,Set(poi trastevere, timestamp Nov 30, 2016, poi piazzanavona, timestamp Nov 30, 2016, poi romanforum, timestamp Nov 30, 2016))
(user 92919639@N03, day Dec 10, 2016,Set(poi romanforum, timestamp Dec 10, 2016))
```

*Nota:* `x._2` è il terzo campo di un oggetto `SingleTrajectory`.

**Esempio di `transactions` risultanti:**

```
Array(villaborghese)
Array(stpeterbasilica, colosseum)
Array(piazzanavona)
Array(trastevere, piazzanavona, romanforum)
Array(romanforum)
…
```

**Esempio di itemset frequenti (`Frequent itemset (model.freqItemsets)`):**

```
{campodefiori}: 388
…
{piazzadispagna}: 211
{piazzadispagna,colosseum}: 34
{piazzadispagna,pantheon}: 51
{piazzadispagna,trevifontain}: 32
```

Il parametro `minSupport` utilizzato è 0.01.

## Fase 3: Implementazione con Hadoop MapReduce

La stessa applicazione viene implementata con Hadoop MapReduce, utilizzando tre componenti principali:

* **DataMapperFilter (Mapper):** Esegue il filtraggio dei dati, riutilizzando le classi `GeoUtils` e `KMLUtils` dell'implementazione Spark. Una nuova classe di utilità, **Flickr**, gestisce i dati Flickr.

* **DataReducerByDay (Reducer):** Estrae le regioni di interesse (RoI) per il mining delle traiettorie.

* **Programma principale:** Combina mapper e reducer, applicando l'algoritmo FPGrowth da **Apache Mahout**.

### DataMapperFilter

La classe `DataMapperFilter` estende la classe `Mapper` di Hadoop. Il metodo `setup` inizializza il poligono di Roma (`romeShape`) e una mappa di poligoni (`shapeMap`) usando `GeoUtils` e `KMLUtils`.

Il metodo `map` analizza i dati JSON, applica i filtri `filterIsGPSValid` e `filterIsInRome`, e scrive l'ID utente e l'oggetto Flickr arricchito con informazioni sulla RoI nel contesto di output.

### DataReducerByDay

Il reducer riceve, per ogni utente, l'insieme dei suoi post arricchiti con informazioni sulla RoI. Il metodo `concatenateLocationsByDay`, usando un comparator per ordinare per data, costruisce la sequenza delle RoI visitate in ogni giorno. Il metodo `reduce` genera la lista delle RoI per ogni giorno.

Infine, l'algoritmo FPGrowth viene utilizzato per estrarre le traiettorie frequenti.

Il metodo principale presentato combina tre componenti chiave: un mapper, un reducer e un'implementazione parallela dell'algoritmo FPGrowth. Questa combinazione permette di estrarre itemset frequenti da grandi dataset in modo efficiente. L'immagine seguente illustra il flusso di lavoro:

Il mapper si occupa della fase di pre-processing dei dati, suddividendo il dataset in parti più piccole e gestendo la conta delle occorrenze degli item in ogni partizione. Il reducer, successivamente, aggrega i risultati intermedi prodotti dal mapper, combinando le conte degli item e identificando gli itemset frequenti. L'utilizzo di un'implementazione parallela di FPGrowth ottimizza ulteriormente il processo, accelerando il calcolo degli itemset frequenti, particolarmente vantaggioso per dataset di grandi dimensioni.
