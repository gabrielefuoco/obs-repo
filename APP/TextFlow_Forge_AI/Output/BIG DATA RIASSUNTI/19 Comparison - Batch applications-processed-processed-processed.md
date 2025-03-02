
# Estrazione di Traiettorie Frequenti da Dati Flickr: Confronto Spark vs. Hadoop

Questo documento descrive l'implementazione di un'applicazione per la scoperta di pattern di mobilità utente a Roma, utilizzando dati geotaggati di Flickr. L'obiettivo è identificare le traiettorie più frequenti tra *punti di interesse* (PoI) o, più precisamente, *aree di interesse* (RoI) che delimitano i PoI.  Una traiettoria è definita come una sequenza di RoI visitate nel tempo, e una traiettoria frequente è una sequenza ripetutamente percorsa dagli utenti.

## Implementazione con Apache Spark

L'implementazione Spark, in Scala, utilizza due classi: `SingleTrajectory` (coppia `<PoI, timestamp>`) e `UserTrajectory` (coppia `<user_id, date>`), entrambe con metodi `equals` e `hashCode` per il confronto.  Il processo inizia leggendo un dataset JSON in un DataFrame Spark.  Dopo un filtraggio dei dati non validi (coordinate GPS errate o località fuori Roma), il programma utilizza la classe `KMLUtils` per leggere un file KML contenente le coordinate delle RoI (es. Colosseo, Musei Vaticani).  L'algoritmo `FPGrowth` viene poi applicato per calcolare le traiettorie frequenti, restituendo gli itemset frequenti e le regole di associazione che rappresentano tali traiettorie.

Un esempio delle forme KML caricate è mostrato di seguito:

```
Map(colosseum -> POLYGON((12.490838 41.891314,12.490249 41.890018,12.490204 41.889794,12.490321 41.889566,12.490494 41.889354,12.490967 41.889194,12.491686 41.88922,12.492165 41.889213,12.492505 41.88919,12.492683 41.889199,12.49333 41.88923,12.493665 41.88934,12.493783 41.889477,12.494041 41.88977,12.494276 41.89039,12.49422 41.89073,12.493732 41.890963,12.49302 41.891225,12.492193 41.89145,12.491616 41.891608,12.490971 41.891729,12.490908 41.89152,12.490838 41.891314)), vaticanmuseums -> POLYGON((12.455288 41.906637,12.454548 41.906892,12.451833 41.906366,12.451571 41.905994,12.454205 41.902972,12.455181 41.903076,12.455288 41.906637)), piazzacolonna -> POLYGON((12.479552 41.900674,12.47962 41.900517,12.48023 41.900682,12.480522 41.900758,12.480407 41.90106,12.480311 41.901285,12.479823 41.901185,12.479384... piazzadelpopolo -> POLYGON((12.47534 41.911012,12.475206 41.910665,12.475384 41.910282,12.475788 41.91008,12.476215 41.909986,12.476639 41.910022,12.477036 41.910162,12.477325 41.910597,12.47739 41.91098,12.477296 41.911138,12.477138 41.911248,12.476891 41.911424,12.47655 41.911503,12.476038 41.911527,12.475716 41.911397,12.47534 41.911012))
```

Un esempio della lettura del dataset JSON in un DataFrame Spark è mostrato nell'immagine: ![[Pasted image 20250223180024.png]]

## Fase di Filtraggio (Spark e Hadoop)

Sia l'implementazione Spark che quella Hadoop (menzionata ma non dettagliata) iniziano con una fase di filtraggio dei dati dai post Flickr.



---

Questo documento descrive l'estrazione di regole di associazione da dati di geolocalizzazione, implementata sia con Apache Spark che con Hadoop MapReduce.

**Fase 1: Filtraggio dei Dati**

I dati vengono filtrati per validità delle coordinate GPS e posizione geografica (Roma), utilizzando la libreria Java Spatial4j e la funzione `filterIsInRome`.  Ulteriori filtri vengono applicati su accuratezza, latitudine, longitudine, `owner.id` e `dateTaken`.

**Fase 2: Estrazione di Regole di Associazione con Spark e FPGrowth**

Dopo il filtraggio, i dati vengono trasformati in tuple `<UserTrajectory, SingleTrajectory>`.  La funzione `mapCreateTrajectory` crea queste tuple, dove `SingleTrajectory` contiene l'ID utente e i Points of Interest (POI) visitati.  La funzione `computeTrajectoryUsingFPGrowth` prepara i dati come `RDD[Array[String]]`, rimuove traiettorie vuote, raggruppa per utente e genera un insieme di itemset unici usando `distinct`.  Questo insieme viene passato all'algoritmo FPGrowth di MLlib con parametri `minSupport` (es. 0.01) e `minConfidence` per generare itemset frequenti e regole di associazione.  Un esempio di dati di transazione preparati è mostrato:

```
(user 35716709@N04, day Nov 20, 2016,Set(poi villaborghese, timestamp Nov 20, 2016))
...
```

e un esempio di itemset frequenti:

```
{campodefiori}: 388
...
{piazzadispagna,colosseum}: 34
```

**Fase 3: Implementazione con Hadoop MapReduce**

L'applicazione viene riprodotta con Hadoop MapReduce, usando:

* **DataMapperFilter (Mapper):** Filtra i dati usando `GeoUtils`, `KMLUtils` e una nuova classe di utilità `Flickr`, applicando i filtri `filterIsGPSValid` e `filterIsInRome`.
* **DataReducerByDay (Reducer):** Estrae le regioni di interesse (RoI) per ogni giorno, ordinando i post per data tramite `concatenateLocationsByDay`.
* **Programma principale:** Combina mapper e reducer, usando FPGrowth da Apache Mahout per l'estrazione delle traiettorie frequenti.

`DataMapperFilter` inizializza il poligono di Roma e una mappa di poligoni usando `GeoUtils` e `KMLUtils`. Il metodo `map` analizza i dati JSON, applica i filtri e scrive l'ID utente e i dati Flickr arricchiti nel contesto di output.  `DataReducerByDay` riceve i post arricchiti per utente e genera la lista delle RoI per ogni giorno.  Il flusso di lavoro prevede un mapper per il pre-processing e un reducer per l'aggregazione dei risultati, entrambi utilizzando FPGrowth per l'estrazione degli itemset frequenti.

---

L'implementazione parallela dell'algoritmo FPGrowth migliora significativamente l'efficienza del mining di itemset frequenti, soprattutto per dataset di grandi dimensioni.  Questa parallelizzazione accelera il calcolo, rendendo l'analisi più rapida rispetto ad un approccio sequenziale.

---
