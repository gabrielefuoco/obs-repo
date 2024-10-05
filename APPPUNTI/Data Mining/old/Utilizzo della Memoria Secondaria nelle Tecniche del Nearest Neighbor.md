Le tecniche del Nearest Neighbor (NN) sono molto utili per diversi scopi, come l'imputazione dei valori mancanti nei dataset e la generazione di suggerimenti personalizzati basati sui dati di navigazione web degli utenti. Quando i dataset diventano molto grandi, tali tecniche necessitano di utilizzare la memoria secondaria (ad esempio, dischi rigidi) per gestire efficacemente i dati.
![[Pasted image 20240604155225.png]]
### Strutture Dati Disk-Based

Per lavorare con dataset di grandi dimensioni che non possono essere completamente caricati in memoria primaria (RAM), vengono utilizzate strutture dati che operano efficientemente su memoria secondaria. Due strutture dati comuni sono:

1. **R-tree**:
   - Struttura dati ottimistica, tende a rispondere alle query in tempo logaritmico.
   - Organizza gli oggetti spaziali in iperrettangoli multidimensionali che possono sovrapporsi.
   - Simile ai B+tree, ma esteso per spazi multidimensionali.

2. **Vector Approximation File (VA-File)**:
   - Approccio pessimistico, analizza l'intero dataset ma lo fa molto velocemente.
   - Spesso usato per velocizzare l'accesso ai dati in spazi multidimensionali.

### R-tree

Gli R-tree sono strutture dati progettate per organizzare e gestire oggetti spaziali, come punti, linee e rettangoli, in spazi multidimensionali. Sono particolarmente utili per supportare query spaziali come la nearest neighbor search.

#### Caratteristiche degli R-tree

1. **Nodi e Rettangoli**:
   - Ogni nodo nell'R-tree rappresenta un insieme di oggetti spaziali.
   - Il nodo padre contiene il rettangolo che racchiude tutti gli oggetti dei nodi figli.

2. **Organizzazione Gerarchica**:
   - I rettangoli rappresentati nei nodi possono sovrapporsi.
   - Gli R-tree organizzano gli oggetti in una struttura gerarchica che facilita l'accesso efficiente ai dati.

#### Processo di Costruzione degli R-tree

La costruzione di un R-tree avviene con un approccio **bottom-up**, il che significa che parte dagli oggetti di base e li raggruppa progressivamente in nodi intermedi fino a formare la radice.

1. **Partizionamento degli Oggetti**:
   - Gli oggetti vengono partizionati in gruppi di piccola cardinalità (tipicamente 2-3 elementi per gruppo).

2. **Calcolo dei Rettangoli Minimi**:
   - Per ogni gruppo di oggetti, si calcola il rettangolo minimo che li racchiude tutti.

3. **Unione Ricorsiva**:
   - I rettangoli minimi vengono uniti ricorsivamente in nodi intermedi.
   - Questo processo continua fino a ottenere un singolo nodo radice che racchiude tutti gli oggetti.

#### Pro e Contro degli R-tree

**Pro**:
- **Supportano la Nearest Neighbor Search**: Sono efficienti nel trovare il punto più vicino a una data query.
- **Versatilità**: Funzionano sia per punti che per rettangoli.
- **Evitano Spazi Vuoti**: La struttura evita la creazione di spazi vuoti inutili.
- **Varianti**: Esistono molte varianti degli R-tree, come X-tree, SS-tree e SR-tree, che possono essere ottimizzate per specifiche esigenze.
- **Buone Prestazioni per Dimensioni Ridotte**: Funzionano bene per dataset con dimensioni ridotte.

**Contro**:
- **Prestazioni in Dimensioni Elevate**: Gli R-tree non sono molto efficienti per dataset con dimensioni elevate (alta dimensionalità), poiché la sovrapposizione dei rettangoli aumenta e la loro efficacia diminuisce.

### Conclusione

L'uso di strutture dati come gli R-tree e i Vector Approximation File è cruciale per gestire dataset di grandi dimensioni che non possono essere interamente caricati in memoria primaria. Gli R-tree, con il loro approccio gerarchico e la capacità di gestire oggetti spaziali multidimensionali, offrono un metodo efficiente per eseguire query spaziali, come la ricerca del vicino più vicino, pur con alcune limitazioni in spazi ad alta dimensione.