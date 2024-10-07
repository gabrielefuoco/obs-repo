### Motivazioni del Data Mining
* **Crescente quantità di dati:** la mole di dati raccolti è in continua crescita, rendendo difficile l'analisi manuale.
* **Complessità dei dati:** i dati sono sempre più complessi, con diverse tipologie e strutture.
* **Velocità di raccolta ed elaborazione:** i dati vengono raccolti ed elaborati a velocità sempre maggiori.

**Obiettivi:**
* **Sfruttare strumenti automatizzati:** per estrarre informazioni dai dati in modo efficiente.
* **Fondere metodi tradizionali con algoritmi sofisticati:** per ottenere risultati più accurati e completi.
* **Fini commerciali e scientifici:** per migliorare i processi decisionali e la ricerca.

**Motivazioni specifiche:**
* **Scalabilità:** la necessità di algoritmi in grado di gestire grandi quantità di dati.
* **Alta dimensionalità:** i dataset possono avere centinaia o migliaia di attributi, rendendo inefficaci le tecniche tradizionali.
* **Eterogeneità e complessità dei dati:** la diversità dei dati richiede algoritmi flessibili e adattabili.

### Cos'è un Pattern?
* **Modello:** un insieme di regole che descrivono i dati, astraendo e fornendo semantica a un insieme di dati.
* **Esprime un modello ricorrente:** identifica schemi e tendenze nei dati.

**Caratteristiche di un pattern:**
* **Validità sui dati:** deve essere supportato dai dati analizzati.
* **Comprendibilità semantica:** deve essere interpretabile e comprensibile dall'uomo.
* **Sconosciuto in precedenza:** deve fornire nuove informazioni non già note.
* **Utilità:** deve essere utile per prendere decisioni o per la ricerca.

### Tipi di Pattern
* **Regole associative:** determinano le regole di implicazione logica presenti in un database.
* **Classificatori:** forniscono un modello per la classificazione dei dati in base a classi predefinite.
* **Alberi decisionali:** classificatori che identificano le cause di un evento mediante una serie di domande-risposte.
* **Clustering:** raggruppa gli elementi di un insieme in classi non determinate a priori, basandosi sulle loro caratteristiche.
* **Serie temporali:** individuano pattern ricorrenti in sequenze di dati complesse.

### Compiti del Data Mining
* **Attività predittive:** prevedere il valore di un attributo in base ai valori di altri attributi.
* **Compiti descrittivi:** creare modelli che riassumono le relazioni presenti nei dati.

### CRISP-DM
* **Metodologia:** definisce i passi fondamentali di un progetto di data mining.
* **Sei fasi:**
    * Comprensione del dominio applicativo
    * Comprensione dei dati
    * Preparazione dei dati
    * Creazione del modello
    * Valutazione del modello
    * Deployment

### Cosa sono i dati?
* **Raccolta di oggetti:** descritti da una serie di attributi che ne descrivono le caratteristiche.

### Tipi di attributo
* **Nominali:** tipi enumerati, con dimensione discreta o binaria.
* **Ordinali:** tipi enumerati con un ordinamento (es. Alto-Medio-Basso), con dimensione discreta o binaria.
* **Di intervallo:** rappresentati da un numero reale con un'unità di misura associata (es. Celsius), con dimensione continua.
* **Di rapporto:** come quelli di intervallo, ma con la possibilità di applicare operatori come addizione e moltiplicazione, con dimensione continua.
* **Asimmetrici:** attributi binari in cui sono importanti solo i valori diversi da 0.

**Operatori:**
* **Diversità:** applicabile a tutti i tipi di attributi.
* **Ordinamento:** applicabile a attributi ordinali, di intervallo e di rapporto.
* **Additività:** applicabile a attributi di intervallo e di rapporto.
* **Moltiplicatività:** applicabile a attributi di rapporto.

### Caratteristiche importanti nei dati
* **Dimensione:** numero di attributi e numero di dati a disposizione.
* **Sparsità:** percentuale di valori significativi (diversi da 0).
* **Risoluzione:** livello di dettaglio dei dati.

### Tipi di dati
* **Tabellari:** raccolta di record con lo stesso insieme di attributi.
* **Matrici:** dati numerici rappresentati come vettori in uno spazio multidimensionale.
* **Documenti:** ogni riga rappresenta un documento, con attributi come le parole più significative.
* **Transazioni:** ogni riga contiene un insieme di oggetti, con numero di elementi variabile.
* **Grafi:** gli oggetti sono rappresentati da nodi, le relazioni da archi.
* **Dati ordinati:** dati con attributi in relazione con un ordine.
    * **Dati spazio-temporali:** oggetti vicini nello spazio o nel tempo tendono ad essere simili.
    * **Sequenze di dati:** dati costruiti da una sequenza di entità (es. informazioni genetiche).

## Esplorazione dei Dati

L'esplorazione dei dati è un'analisi preliminare che mira a identificare le principali caratteristiche di un dataset. Si utilizzano indicatori statistici e strumenti di visualizzazione per ottenere una comprensione iniziale dei dati.

### Indicatori Statistici

#### Misure di Posizione

* **Media:** Valore medio di un insieme di dati. Sensibile agli outlier.
* **Mediana:** Valore che occupa la posizione centrale in un insieme di dati ordinati. Se il numero di dati è dispari, è il valore centrale. Se il numero di dati è pari, è la media dei due valori centrali.
* **Range:** Differenza tra il valore minimo e il valore massimo di un attributo.

#### Misure di Dispersione

* **Varianza:** Misura di dispersione più comune. Indica quanto i valori si discostano quadraticamente dalla media.
    * $Var(x)=s^2_{x}=\frac{1}{n}\sum_{i=1}^n (x_{i}-\bar{x})^2$
* **Deviazione Standard:** Radice quadrata della varianza.
    * $DevStd(x)=s_{x}=\sqrt{ \frac{1}{n}\sum_{i=1}^n (x_{i}-\bar{x})^2}$
* **Covarianza:** Misura la relazione lineare tra due vettori di valori. Indica se sono direttamente proporzionali, inversamente proporzionali o incorrelati.
    * $CoVar(x,y)=\frac{1}{n}\sqrt{\sum_{i=1}^n (x_{i}-\bar{x})(y_{i}-\bar{y})}$
* **Deviazione Assoluta Media:** Misura la distanza dei valori rispetto alla media prendendo il valore assoluto e non il quadrato.
* **Deviazione Assoluta Mediana:** Calcola la mediana delle distanze dal valore medio.
* **Range Interquartile:** Calcola la distanza tra il 75° e il 25° percentile.

### Qualità dei Dati

La qualità dei dati è fondamentale per ottenere risultati coerenti. Il Data Mining si concentra sul rilevamento e la correzione dei problemi di qualità dei dati, oltre che sull'utilizzo di algoritmi che possano tollerare una scarsa qualità.

#### Problemi di Qualità dei Dati

* **Rumore:** Componente casuale di un errore di misurazione che distorce i valori o aggiunge oggetti spuri. L'eliminazione del rumore è difficile, quindi il Data Mining si concentra sullo sviluppo di algoritmi che producano risultati accettabili anche in presenza di errori.
* **Outlier:** Valori anomali che hanno caratteristiche diverse dalla maggior parte degli attributi del dataset o con valori insoliti rispetto a quelli tipici del determinato attributo.
* **Valori Mancanti:** Le strategie per affrontare questo problema sono:
    * **Eliminazione di oggetti o attributi:** Da fare con cautela, in quanto gli attributi eliminati possono essere critici per l'analisi.
    * **Stima dei valori mancanti:** I valori mancanti possono essere stimati usando i valori rimanenti. Se l'attributo è continuo, si utilizza il valore medio degli altri. Se è categorico, si prende il valore dell'attributo più comune.
    * **Ignorare i valori mancanti.**
* **Dati Inconsistenti:** Inseriti o rilevati male.
* **Dati Duplicati:** Uno o più valori potrebbero essere diversi e devono essere corretti (deduplicazione).

### Similarità e Dissimilarità

* **Similarità:** Misura numerica che esprime il grado di somiglianza tra due oggetti. È alta in caso di coppie di oggetti simili. Solitamente le similarità sono non negative e comprese tra 0 e 1.
* **Dissimilarità:** Esprime il grado di non-somiglianza tra due coppie di oggetti. È tipicamente definita combinando le vicinanze dei singoli attributi.

#### Misure di Similarità e Dissimilarità per Diversi Tipi di Attributi

* **Attributi Nominali:** Trasmettono solo informazioni sulla distinzione degli oggetti. 0 se uguali, 1 altrimenti.
* **Attributi Ordinali:** $|\frac{val_{1}-val_{2}}{\text{num di tuple -1}}|$
* **Attributi di Intervallo o Rapporto:** $|val_{1}-val_{2}|$

## Distanze

### Distanza Euclidea

In uno spazio multidimensionale, la distanza euclidea è data dalla seguente formula:

$d(x, y) = \sqrt{\sum_{k=1}^n (x_k - y_k)^2}$

Se si considerano attributi con scale di misura diverse, è necessario normalizzarli.

### Distanza di Minkowski

Generalizza la distanza euclidea.

$d(x, y) = \left(\sum_{k=1}^n |x_k - y_k|^r \right)^{1/r}$

* **r = 1:** Distanza di Manhattan, somma delle differenze assolute delle coordinate orizzontali e verticali dei due punti.
* **r = 2:** Distanza Euclidea.
* **r = infinito:** Distanza del Supremum, misura la distanza massima che esiste tra due valori della stessa componente.

### Distanza di Mahalanobis

Quando le variabili hanno scale diverse o gli attributi risultano correlati, si ha una generalizzazione della distanza euclidea che tenga conto della correlazione all'interno dei dati. Risulta utile quando gli attributi hanno differenti varianze e la distribuzione dei dati si approssima alla Normale.

$d(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$

Dove:

* $x$ e $y$ sono i vettori di dati che si desidera confrontare.
* $\Sigma$ è la matrice di covarianza dei dati.
* $T$ indica la trasposizione di un vettore.
*  $\Sigma^{-1}$ indica l'inversa della matrice di covarianza. 

### Proprietà delle distanze

Una distanza che soddisfa tutte le proprietà è detta **Metrica**.

* **Positività:** $d(x, y) \ge 0$ e $d(x, y) = 0$ se e solo se $x = y$.
* **Simmetria:** $d(x, y) = d(y, x)$.
* **Disuguaglianza Triangolare:** $d(p, r) \le d(p, q) + d(q, r)$.

### Range Queries

Query che recuperano tutti i dati che ricadono in un determinato intervallo di valori.

Dati un insieme di punti P e una range query con raggio r da un punto q, sfruttando la disuguaglianza triangolare è possibile limitare il numero di distanze d(xi,q) da calcolare per rispondere alla query:
$d(p,q)\leq d(p,x_{i})+d(q,x_{i})\to d(q,x_{i})\geq d(p,q)-d(p,x_{i}) \to \text{tutti i punti }x_{i} \text{per cui } d(p,q)-d(p,x_{i})>r$ devono essere scartati, poiché il punto $x_i$ non può essere un vicino più prossimo di q entro il raggio r; quindi, può essere scartato senza essere valutato ulteriormente.

$d(p,x_{i})\leq d(q,p)+d(p,x_{i})\to \text{tutti i punti }x_{i} \text{per cui } d(p,x_{i})+d(p,q)<r$ devono essere accettati senza essere valutati.

### Dissimilarità non metriche

È possibile definire la distanza d tra due insiemi A e B come size(A-B), ma tale valore non soddisfa le proprietà di simmetria e disuguaglianza triangolare. Dunque, possiamo ridefinire d come:

$d(A,B)=size(A-B)+size(B-A)$

### Proprietà delle similarità

* $s(p,q)=1 \ sse \ p=1$
* $s(p,q)=s(q,p)=\text{simmetria}$

Non esiste per la similarità un concetto equivalente alla disuguaglianza triangolare.

### Similarià tra vettori binari

Le misure di similarità tra oggetti caratterizzati da soli attributi binari sono dette Coefficienti di similarità e assumono valori compresi tra 0 e 1.

Abbiamo 4 possibili grandezze:

* M01=II numero di attributi in cui p=0 e q=1
* M10=II numero di attributi in cui p=1 e q=0
* M00=II numero di attributi in cui p=0 e q=0
* M11=II numero di attributi in cui p=1 e q=1

**Simple Matching Coefficent**

$SMC=\frac{M_{11}+M_{00}}{M_{OO}+M_{11}+M_{01}+M_{10}}$

Valuta allo stesso modo sia le presenze che le assenze e questo rappresenta un problema in caso di attributi binari asimmetrici (gli 0 (assenze) hanno meno importanza degli 1 (presenze)).

**Jaccard Coefficient**

Variante del SMC che viene spesso utilizzato per gestire gli oggetti costituiti da attributi binari asimmetrici.

$J=\frac{M_{11}}{M_{11}+M_{01}+M_{10}}$

## Similarità

### Similarità del Coseno

* Non considera le corrispondenze 00 e ci permette di lavorare anche con vettori non binari.
* $cos(x,y)=\frac{x \cdot y}{||x|| \cdot ||y||}$
* È una misura dell'angolo tra i due vettori, dunque è uguale a 0 se l'angolo è di 90° (non hanno elementi in comune).
* È utilizzata per calcolare la similarità tra i documenti, ma in questo caso occorre normalizzare le lunghezze dei vettori.

### Tanimoto (Extended Jaccard Coefficient)

* Utilizzata in presenza di attributi continui o di intervallo, con valori non binari.
* $EJ(x,y)=\frac{x \cdot y}{||x||^2+||y||^2-x \cdot y}$

### Similarità per dati con attributi eterogenei

In presenza di attributi eterogenei è necessario calcolare la similarità separatamente e quindi combinarla in modo che il risultato appartenga al range [0,1].

In caso di attributi asimmetrici, vanno rimossi i match 00.

Per gestire attributi di tipo diverso nel calcolo della similarità tra due oggetti x e y:

* Si calcola la similarità $s_k(x, y)$ per ciascun attributo k.
* Si definisce un parametro $\delta_k$ per ciascun attributo k:
    * $\delta_k = 0$ se l'attributo è asimmetrico con valori 0 o mancanti.
    * $\delta_k=1$ altrimenti.
* La similarità tra x e y si può definire come:
    * $similarity(x,y)=\frac{\left( \sum_{k=1}^n \delta_{k}s_{k}(x,y) \right)}{\sum_{k=1}^n \delta_{k}}$

Se gli attributi hanno una rilevanza diversa, è possibile aggiungere un peso $\omega_k$.

### Correlazione

Utilizzata per misurare l'esistenza di una relazione lineare tra coppie di oggetti binari o discreti; varia nel range [-1,1].

* **Correlazione di Pearson:**
    * $corr(x,y)=\frac{Cov(x,y)}{StDev(x)\cdot StDev(y)}$

La covarianza può riferirsi a:

1. _Covarianza di variabili casuali:_ parametro di popolazione, proprietà della distribuzione congiunta.
2. _Covarianza di un campione:_ statistica campionaria, stima della covarianza di popolazione e descrive il campione.

La differenza è nel denominatore: per la covarianza di popolazione si usa n, mentre per quella campionaria si usa n-1 per correggere la distorsione. Tuttavia, la correlazione è invariante rispetto al tipo di covarianza utilizzato.

### Confronto tra misure di prossimità

* La similarità del coseno rimane invariata rispetto alle operazioni di ridimensionamento ma non di traslazione.
* La distanza euclidea è suscettibile a entrambe.
* La misura di correlazione rimane invariata in entrambi i casi.

### Scatter Plot

Permettono di visualizzare graficamente la correlazione tra due vettori. Inoltre, quando sono disponibili etichette, permette di determinare se è possibile classificare gli oggetti in base ai valori di due attributi.

### Densità

Utilizzata per misurare il grado in cui gli oggetti dati sono vicini l'uno all'altro.

È strettamente correlata a quella di prossimità e viene usata per il rilevamento di cluster o anomalie.

La densità Euclidea è definita come il numero di punti per unità di volume, e può essere applicata in maniera:

* **Center based:** in cui corrisponde al numero di punti entro un certo raggio.
* **Grid based:** consiste nel dividere la regione in un numero di celle di uguale volume e definire la densità come il numero di punti nella cella.

## Preprocessing

Serie di azioni volta a consentire il funzionamento degli algoritmi di DM.

* **Aggregazione:** combina due o più attributi in uno solo al fine di diminuire la cardinalità del dataset.
* **Campionamento:** utilizzato perché processare l'intero dataset potrebbe essere troppo costoso.
    * Se il campione è rappresentativo, il risultato sarà equivalente a quello che si otterrebbe utilizzando l'intero dataset.
    * **Campionamento casuale semplice:**
        * Tutti gli elementi hanno la stessa probabilità di selezione.
        * _Senza reimbussolamento_: elementi selezionati non possono essere riscelti.
        * _Con reimbussolamento_: elementi selezionati possono essere riscelti più volte. Risultati simili al caso senza reimbussolamento se campione << popolazione.
    * **Campionamento stratificato:** si dividono i dati in più partizioni e si usa il campionamento semplice su ogni partizione. Utile se la popolazione è costituita da diversi tipi di oggetti.
* **Riduzione della dimensionalità:** applicato sugli attributi per evitare la cosiddetta _Curse of Dimensionality_, ovvero che al crescere della dimensionalità i dati diventano progressivamente più sparsi (risultato scadente), e per diminuire il tempo necessario.
* **Selezione degli attributi:** La selezione degli attributi riduce la dimensionalità dei dati rimuovendo attributi ridondanti o irrilevanti. Esistono diversi approcci:
    * **Esaustivo**: prova tutti i sottoinsiemi di attributi e sceglie il migliore (complessità 2n).
    * **Non esaustivi:** _embedded_ (parte dell'algoritmo di mining), _filtro_ (prima del mining), _euristici_ (approssimazione esaustiva).
L'obiettivo è eliminare ridondanze e caratteristiche non significative per il mining.
* **Creazione degli attributi:** attuata tramite l'estrazione delle caratteristiche o la combinazione di attributi.
* **Discretizzazione:** converte attributi continui in discreti, essenziale per alcune tecniche di data mining. Determina il numero di intervalli e i punti di separazione (_split point_).

## Tecniche di Discretizzazione

### Discretizzazione non supervisionata

* **Equi-larghezza:** suddivide l'intervallo di valori dell'attributo in intervalli di uguale ampiezza.
* **Equi-frequenza:** suddivide l'intervallo di valori dell'attributo in intervalli che contengono lo stesso numero di istanze.
* **K-mediani:** minimizza la distanza tra i punti appartenenti allo stesso raggruppamento.

### Discretizzazione supervisionata

* Massimizza la "purezza" degli intervalli rispetto alle classi, utilizzando misure come l'entropia per scegliere i migliori split point in modo ricorsivo.
* L'obiettivo è ridurre il numero di valori mantenendo informazioni rilevanti.

### Binarizzazione

* Effettua la rappresentazione di un attributo discreto mediante un insieme di attributi binari (costoso).

### Trasformazione degli attributi

* Mappa l'insieme di valori di un attributo in un nuovo insieme, associando ad ogni valore di partenza uno unico di arrivo, ovvero in formato più adatto per l'analisi dei dati.

#### Tipi di trasformazioni

* **Funzioni semplici:** (xk, log(x), ecc.) per enfatizzare proprietà dei dati o ridurre range eccessivi.
* **Normalizzazione:** per far rispettare certe proprietà e combinare variabili con range diversi:
    * **Max-Min:** riscala l'attributo A in modo che i nuovi valori siano nel range $[NewMin, NewMax],$ sensibile agli outlier.
    * **Z-score:** con media 0 e deviazione standard 1, meno sensibile agli outlier.
