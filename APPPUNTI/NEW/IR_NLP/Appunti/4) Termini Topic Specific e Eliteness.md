## Rilevanza e Incertezza nella Ricerca di Informazioni

Il concetto di rilevanza nella ricerca di informazioni è passato da un criterio binario (rilevante o non rilevante) a un approccio probabilistico. Questo cambiamento è dovuto alla presenza di **fattori latenti** che introducono incertezza sia nelle query degli utenti che nella rappresentazione dei documenti.

Il **Binary Independence Model (BIM)** è un modello che affronta questa incertezza concentrandosi sulla probabilità che un documento sia rilevante. Il BIM considera la presenza o l'assenza di un termine in un documento come indicatore di rilevanza.

Questo modello è efficace per la ricerca in archivi o messaggi, dove le query sono composte da poche parole chiave. Tuttavia, per gestire la complessità delle query più elaborate, come quelle utilizzate nella ricerca web, è necessario estendere il BIM.

L'estensione del BIM prevede l'integrazione del concetto di **frequenza dei termini (TF)**, che tiene conto del numero di volte in cui un termine appare in un documento. Questo approccio consente di calcolare la probabilità di rilevanza in modo più accurato, considerando l'incertezza presente nei termini utilizzati dagli utenti nelle query reali.

È importante notare che il BIM, nella sua forma originale, assume l'indipendenza tra i termini, un'ipotesi semplificativa che non riflette la realtà.

La **pseudo-rilevanza** è un'approssimazione che si basa su un sistema semi-automatico di attribuzione della rilevanza. Sebbene sia utilizzata con successo in ambiti come l'ottimizzazione SEO, è fondamentale la capacità di stimare la rilevanza in modo accurato.

Per migliorare la stima della rilevanza, si introduce un **prior bayesiano**, un fattore di smoothing che contribuisce a gestire l'incertezza e a catturare i fattori latenti nella rappresentazione dei documenti. Lo smoothing bayesiano serve a mitigare l'impatto dell'assunzione di indipendenza tra i termini, che è una semplificazione necessaria per gestire la complessità del problema.

## Termini Topic Specific e Eliteness

L'assunzione di indipendenza tra i termini, comunemente utilizzata nei modelli di recupero delle informazioni, presenta un problema significativo per i termini meno frequenti, ovvero quelli che si trovano nella "coda" della distribuzione. Questi termini, pur costituendo la maggior parte del vocabolario, vengono trattati come se fossero indipendenti, il che non corrisponde alla realtà.

Questo problema si accentua ulteriormente con i termini "elite", ovvero termini specifici di un determinato argomento (topic-specific). Questi termini, pur potenzialmente presenti in molti documenti, assumono un'importanza particolare nel contesto specifico di un documento. Ad esempio, in un articolo su Donald Trump, termini come "presidente", "Stati Uniti" o "elezioni" potrebbero essere considerati termini elite.

È importante sottolineare che i termini elite non coincidono necessariamente con le Named Entities, ovvero termini che identificano entità specifiche come persone, organizzazioni o luoghi geografici. Sebbene le Named Entities siano importanti, non sempre rappresentano i termini chiave per comprendere il tema principale di un documento.

## L'approssimazione di Poisson

L'approssimazione di Poisson è uno strumento utile per analizzare eventi rari in un intervallo di tempo o spazio. Per poterla applicare, è necessario che la sequenza di eventi sia molto lunga e che la probabilità di successo per ogni evento sia molto bassa.

**Generazione di documenti:** Nel contesto della generazione di documenti, l'approssimazione di Poisson può essere utilizzata per modellare la probabilità di osservare un evento posizione per posizione. Ogni occorrenza di parola viene campionata da una distribuzione multinomiale, che generalizza la distribuzione binomiale a più eventi.

**Regole empiriche:** Per determinare se una distribuzione può essere approssimata con una Poissoniana, si può utilizzare una regola empirica: si dovrebbero avere almeno 30 misurazioni, idealmente 50-100.

**Confronto con la distribuzione binomiale:** Esiste una regola empirica anche per scegliere tra una distribuzione di Poisson e una distribuzione binomiale. Se K è maggiore di 20 o 30, con una probabilità di successo dell'ordine di 1/K, si può utilizzare l'approssimazione di Poisson.

**Approssimazione nella coda della distribuzione:** L'approssimazione di Poisson è particolarmente utile per valutare la probabilità di non occorrenza di eventi, ovvero eventi che si trovano nella coda della distribuzione.

##### Derivazione dell'approssimazione:

La distribuzione delle frequenze dei termini segue una distribuzione binomiale ed è approssimata da una Poisson.
$$B_{T,P}(k)=\begin{pmatrix}T \\ K \end{pmatrix}p^k(1-p)^{t-k}$$
Possiamo approssimare $p^k(1-p)^{t-k}$ come $e^{-Tp}$, ponendo $\lambda=Tp$ e sviluppando il prodotto otteniamo:
$$p_{\lambda}(k)=\frac{\lambda^k}{k!}e^{-\lambda}$$
* Dove **k è il numero di occorrenze**
Questa formula mostra che per k grande, la probabilità di k eventi è approssimativamente proporzionale a una funzione gaussiana con media t e deviazione standard √t.

Per affermare che la possion vale come approssimazione di binomiale dobbiamo avere che:
- T deve essere molto grande.
- La probabilità di successo deve essere molto bassa (i termini devono stare nella coda della curva di Zipf).
* **Media = Varianza = λ = cf/T**
	* Se T è grande e p è piccolo, possiamo approssimare una distribuzione binomiale con una Poisson dove λ = Tp
* **"Intervallo fisso" implica una lunghezza di documento fissa**
	* Ad esempio, abstract di documenti di dimensioni costanti

La frequenza globale dei termini segue una legge di Zipf, prescinde dalla variabile documento.

### Limiti per i termini "topic specific"

L'approssimazione di Poisson, pur essendo valida per la maggior parte dei termini, presenta dei limiti quando si tratta di termini "topic specific", ovvero termini legati a un determinato argomento. Questi termini, essendo fortemente legati al contesto del documento, non sono generali e presentano un pattern di occorrenza diverso rispetto ai termini per cui l'approssimazione di Poisson è valida.

##### Caratteristiche dei termini "topic specific":

* **Mancanza di indipendenza:** I termini "topic specific" non sono indipendenti tra loro, ma tendono a co-occorrere in modo significativo.
* **Contestualità:** Questi termini sono fortemente legati al contesto del documento, presentando una forte proprietà di "contextual bound".
* **Occorrenza a gruppi:** I termini "topic specific" non si presentano necessariamente a coppie o triple, ma tendono a formare gruppi, evidenziando una coerenza e una condivisione di pattern di occorrenza.

##### Implicazioni per la modellazione:

L'approssimazione di Poisson non è adatta a modellare la probabilità di occorrenza dei termini "topic specific" a causa della loro dipendenza e contestualità. Un modello basato sulla Poissoniana non riesce a catturare la complessità del loro comportamento.

##### Soluzioni:

* **Approssimazione Okapi BM25:** Un'approssimazione alternativa, vantaggiosa per diversi aspetti, è implementata nel modello Okapi BM25.
* **Distribuzione Binomiale Negativa:** Un'altra soluzione è l'utilizzo della distribuzione binomiale negativa, che si adatta meglio alla stima della probabilità di occorrenza dei termini "topic specific".

## Distribuzione Binomiale Negativa

Vogliamo modellare la probabilità di osservare un certo numero *k* di insuccessi da una sequenza di trial Bernoulliani *i.i.d.* fino a osservare un certo numero *r* di successi (*r* è la stopping condition).

$$NB_{\ r,p}(k)= \begin{pmatrix}
k+r-1 \\
k
\end{pmatrix}p^r(1-p)^k$$

* *r* è il parametro di dispersione (numero di successi), *k* è il numero di failures da osservare, *p* è la probabilità di successo.

A seconda di come si interpreta il supporto, *k* e *r* assumono significato diverso. Noi stiamo osservando la probabilità di osservare *k* insuccessi.
La controparte di questa è quella in cui facciamo uno switch tra *k* e *r* **->** *r* diventa il numero di insuccessi e *k* di successi.

### Parametrizzazione della Binomiale Negativa

La media della distribuzione binomiale negativa è data da:

$$\mu=\frac{rp}{1-p}$$

dove *r* è il parametro di dispersione per l'evento successo e *p* è la probabilità di successo.

Possiamo ricavare *p* e (1-*p*) in funzione di *r* e *μ*:

* $rp=\mu-\mu p$
* $p=\frac{\mu}{\mu+r}$
* $(1-p)=\frac{r}{\mu+r}$

Sostituendo queste espressioni nella formula della binomiale negativa, otteniamo:

$$NB_{\ r,p}(k)= \begin{pmatrix}
k+r-1 \\
k
\end{pmatrix}\left( \frac{r}{\mu+r} \right)^r\left( \frac{\mu}{\mu+r} \right)^k$$

* *k+r* è il numero di trial (*T*), il-1 è perché si assume che l'ultimo evento sia un successo.

Introduciamo una dimensione che ci consenta di regolarci, in termini poissoniani, che un termini sia o non sia topic-specific, ovvero "elite"

### Termini Elite

Un termine "elite" è un termine, generalmente un sostantivo, che descrive un concetto, un tema o un topic presente nel corpus o nel testo in esame. La proprietà di "eliteness" è valutata in modo binario: un termine è o elite o non lo è.

**Attenzione:** l'essere "elite" dipende dalla rilevanza del documento al topic in questione. In altre parole, la "eliteness" di un termine è determinata dalla sua **term frequency** nel documento, ma solo se il documento è effettivamente rilevante per il topic.
### Eliteness

L'eliteness di un termine, ovvero la sua capacità di essere informativo per un determinato topic, è strettamente legata alla rilevanza del documento al topic stesso. Un documento può contenere molti termini "elite", ma solo quelli presenti in un documento rilevante al topic sono considerati realmente informativi.

##### Modellare l'eliteness:

* **Variabile nascosta:** Per ogni coppia documento-termine, si introduce una variabile nascosta, indicata come $E_i$ per il termine $i$, che rappresenta l'argomento. Un termine è elite in un documento se il documento tratta il concetto denotato dal termine.
* **Eliteness binaria:** Un termine o è elite o non lo è.

**Importanza dei "pattern" distribuzionali:** I termini "elite", essendo "topic specific", sono spesso classificati in gruppi: in un determinato contesto, alcuni termini possono avere "pattern" distribuzionali simili, ovvero frequenze di occorrenza simili.

**Il problema del "topic drift":** Un documento può contenere più termini "elite" che fanno riferimento a concetti diversi. Se consideriamo tutti i termini "elite" come appartenenti allo stesso concetto, potremmo non considerare altri concetti presenti nel testo. Questo fenomeno può portare al "topic drift", ovvero all'errata classificazione di un documento come rilevante ad un tema solo perché correlato, in termini distribuzionali, ad un altro tema, pur non essendo semanticamente affine.

**Esempio:** un documento su football americano potrebbe contenere termini "elite" relativi ad altri sport, come il baseball. Se non si tiene conto di questo "topic drift", il documento potrebbe essere erroneamente classificato come rilevante per il baseball.

![[4)-20241028152840027.png|227]]

### Retrieval Status Value (RSV) con termini "elite"

Il RSV con termini "elite" è simile alla derivazione bimodale:

$$RSV^{elite}=\sum_{i\in q}c_{i^{elite}}(tf_{i})$$

dove:

$$c_{i}^{elite}(tf_{i})=\log \frac{p(TF_{i}=tf_{i}|R=1)p(TF_{i}=0|R=0)}{p(TF_{i}=0|R=1)p(TF_{i}=tf_{i}|R=0)}$$

e usando l'eliteness:

$$p(TF_{i}=tf_{i}|R)=p(TF_{i}=tf_{i}|E=elite)p(E=elite|R)+p(TF_{i}=tf_{i}|E=elite)(1-p(E_{i}=elite|R))$$

Vogliamo personalizzare il RSV tenendo conto dei termini "elite". Nel termine i-esimo, il costo, tenendo conto della term frequency, è dato dalla probabilità. Stimiamo quindi l'odds ratio, considerando una nuova variabile TF che può assumere un preciso valore di term frequency. Nell'ultima parte, esplicitiamo la presenza o meno del concetto di "eliteness".

Data l'assunzione di rilevanza o meno, la probabilità di osservare una TF=tf è espressa come unione di due eventi che si verificano congiuntamente: la probabilità data l'assunzione che il termine i-esimo sia "elite" per la probabilità che, se vale l'assunzione di rilevanza, il termine sia "elite", più l'evento in cui il termine non è "elite", con la probabilità di osservare il termine, dato R, e E_i "elite".

### Modello a Due Poisson

I problemi riscontrati con il modello a 1-Poisson suggeriscono l'utilizzo di due distribuzioni di Poisson: la distribuzione varia a seconda che il termine sia "elite" o meno. La probabilità di osservare una term frequency (TF) pari a *k*, dato uno stato di rilevanza *R*, è data da:

$$p(TF_{i}=k|R)=\pi \frac{\lambda^k}{k!}e^{-\lambda}+(1-\pi) \frac{\mu^k}{k!}e^{-\mu}$$

dove:

* **π** è la probabilità che un documento sia "elite" per un termine.
* **λ** e **μ** sono i tassi delle due distribuzioni di Poisson (λ per i termini "elite", μ per gli altri). Questi sono parametri sconosciuti, così come π.

Considerando l'esistenza di termini che non si adattano bene a una singola distribuzione di Poisson, si utilizzano due distribuzioni per modellare il termine *i*-esimo con $TF=k_i$, come combinazione lineare di due distribuzioni di Poisson, a seconda che il termine sia "elite" o meno.

### Modello di Poisson (semplificato)

Il modello di Poisson presenta le seguenti proprietà:

* Aumenta monotonicamente con la term frequency (tf<sub>i</sub>).
* Asintoticamente si avvicina a un valore massimo.
* Il limite asintotico rappresenta il peso della caratteristica di "eliteness". *

La stima dei parametri per il modello a due Poisson è complessa. Si può approssimare con una semplice curva parametrica che possiede le stesse proprietà qualitative.

### Modello di Costo per Termini "Elite"

Il costo $c_i^{elite}$ per il termine *i*-esimo, considerando la term frequency, è definito come segue:

* $c_i^{elite}(0) = 0$
* $c_i^{elite}(tf_i)$ cresce monotonicamente, ma satura per valori alti di λ.

La stima dei parametri è preferibilmente effettuata tramite una funzione che presenti le stesse caratteristiche qualitative.

![[4)-20241028163529554.png|294]]

La figura mostra graficamente come la funzione $c_i^{elite}$ cresce monotonicamente con la term frequency, saturando per valori elevati di λ.

### Approssimazione della Poisson

Possiamo approssimare la distribuzione di Poisson con la seguente funzione:

$$\frac{tf}{k_1 + tf}$$

Se andiamo a graficarla, otteniamo:

![[4)-20241028163604531.png]]

In pratica:

* Per valori alti di $k_1$, gli incrementi in $tf_i$ continuano a contribuire significativamente al punteggio.
* I contributi diminuiscono rapidamente per valori bassi di $k_1$.

Per valori più alti di $k_1$, l'approssimazione della funzione è peggiore.

## Prime versioni di BM25

#### Versione 1: funzione di saturazione

La prima versione di BM25 utilizza la funzione di saturazione per calcolare il costo $c_i^{BM25v_1}$ del termine i-esimo:

$$c_{i}^{BM25v_{1}}(tf_{i})=c_{i}^{BIM} \frac{tf_{i}}{k_{1}+tf_{i}}$$

dove $c_i^{BIM}$ è il costo del termine i-esimo calcolato con il modello BIM.

#### Versione 2: semplificazione BIM a IDF

La seconda versione di BM25 semplifica il modello BIM utilizzando solo l'IDF (Inverse Document Frequency):

$$c_{i}^{BM25v_{2}}(tf_{i})=\log \frac{N}{df_{i}}\times \frac{(k_{1}+1)tf_{i}}{k_{1}+tf_{i}}$$

dove:

* $N$ è il numero totale di documenti nella collezione.
* $df_i$ è il numero di documenti che contengono il termine i-esimo.

## Estensioni del modello BM25

#### Prima estensione: funzione di saturazione

La prima estensione del modello BM25 utilizza la funzione di saturazione per tenere conto della term frequency. In pratica, si prende il costo $c_i$ del modello BM25 e si introduce un fattore di smoothing basato sulla funzione di saturazione.

#### Seconda estensione: stima di $r_i$ e utilizzo di $df$

La seconda estensione del modello BM25 utilizza solo la stima di $r_i$ (la probabilità che un documento sia rilevante dato il termine i-esimo) e la $df$ (la frequenza del termine i-esimo nella collezione). In questo caso, non si utilizza il costo $c_i$ completo.

### Normalizzazione della lunghezza del documento

Il modello BM25 può essere ulteriormente migliorato introducendo una normalizzazione della lunghezza del documento. Questo è importante perché i documenti possono avere lunghezze variabili, il che può influenzare il valore di $tf_i$.

#### Lunghezza del documento

* **dl** è la somma delle tf per tutti i termini.
* **avdl** è la lunghezza media dei documenti nella collezione.

I documenti più lunghi hanno maggiori probabilità di avere valori di $tf_i$ più grandi.

#### Motivi per la lunghezza variabile dei documenti

* **Verbosità:** suggerisce che il $tf_i$ osservato è troppo alto.
* **Ambito più ampio:** suggerisce che il $tf_i$ osservato potrebbe essere corretto.
* Una collezione di documenti reale probabilmente presenta entrambi gli effetti.

#### Normalizzazione della lunghezza del documento

* **Lunghezza del documento:**

$$dl=\sum_{i\in V}tf_{i}$$

* **avdl:** Lunghezza media dei documenti nella collezione.

* **Componente di normalizzazione della lunghezza:**

$$B=\left( (1-b)+b \frac{dl}{avdl} \right), 0\leq b\leq1$$

* **b = 1:** normalizzazione completa della lunghezza del documento.
* **b = 0:** nessuna normalizzazione della lunghezza del documento.

## Okapi BM25

Il modello Okapi BM25 è un'estensione del modello BIM (Best Match) che tiene conto della lunghezza dei documenti, normalizzando la term frequency (tf) in base ad essa.

La normalizzazione della term frequency rispetto alla lunghezza del documento è effettuata come segue:

$$tf_i' = \frac{tf_i}{B}$$

dove tf<sub>i</sub> è la term frequency del termine *i* e *B* è un parametro. La formula completa per il punteggio BM25 del termine *i* è:

$$c_i^{BM25}(tf_i) = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i'}{k_1+tf_i'} = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$

dove:

* `N` è il numero totale di documenti nella collezione.
* df<sub>i</sub> è il numero di documenti che contengono il termine *i*.
* k<sub>1</sub> e `b` sono parametri.
* `dl` è la lunghezza del documento.
* `avdl` è la lunghezza media dei documenti nella collezione.

La funzione di ranking BM25 è data dalla somma dei punteggi BM25 per ogni termine nella query:

$$RSV^{BM25} = \sum_{i \in q} c_i^{BM25}(tf_i)$$

dove `q` rappresenta l'insieme dei termini nella query.

Rispetto al modello BIM, BM25 migliora la valutazione della term frequency normalizzandola in base alla lunghezza dei documenti. La seconda formula presentata rappresenta una versione più sviluppata del modello, che include esplicitamente la normalizzazione della term frequency.

### Parametri del modello BM25

$$RSV^{BM25} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$

Il modello BM25 utilizza due parametri principali:

* **k1:** gestisce la pendenza della funzione di saturazione. Controlla lo scaling della term frequency.
* **k1 = 0:** modello binario.
* **k1 grande:** term frequency grezza.
* **b:** controlla la normalizzazione della lunghezza del documento.
* **b = 0:** nessuna normalizzazione della lunghezza.
* **b = 1:** frequenza relativa (scala completamente in base alla lunghezza del documento).

Tipicamente, k1 è impostato intorno a 1.2-2 e b intorno a 0.75. È possibile incorporare la ponderazione dei termini di query e il feedback di rilevanza (pseudo).

### Esempio di applicazione del modello BM25

Supponiamo una query "machine learning".

Supponiamo di avere 2 documenti con i seguenti conteggi di termini:

* **doc1:** learning 1024; machine 1
* **doc2:** learning 16; machine 8

Calcoliamo il punteggio tf-idf e BM25 per entrambi i documenti:

**tf-idf:** $log_2 (tf) \cdot \ log_{2} \left( \frac{N}{df} \right)$

* **doc1:** $11 * 7 + 1 * 10 = 87$
* **doc2:** $5 \cdot 7 + 4 \cdot 10 = 75$

**BM25:** $k_{1} = 2$

* **doc1:** $7 \cdot 3 + 10 \cdot 1 = 31$
* **doc2:** $7 \cdot 2.67 + 10 \cdot 2.4 = 42.7$

### Ranking con zone

**Zone:** Una zona è una sezione specifica di un documento. Ad esempio, in un articolo scientifico, le zone possono essere il titolo, l'abstract, l'introduzione, le sezioni di risultati e discussione, le conclusioni, le keyword e gli highlights.

#### Idea semplice

* Applicare la funzione di ranking preferita (BM25) a ciascuna zona separatamente.
* Combinare i punteggi delle zone utilizzando una combinazione lineare ponderata.

Tuttavia, questo sembra implicare che le proprietà di eliteness delle diverse zone siano diverse e indipendenti l'una dall'altra, il che sembra irragionevole.

#### Idea alternativa

* Assumere che l'eliteness sia una proprietà del termine/documento condivisa tra le zone.
* Ma la relazione tra eliteness e frequenze dei termini è dipendente dalla zona.
* Ad esempio, un uso più denso di parole di argomento elite nel titolo.

#### Conseguenza

* Combinare prima le prove tra le zone per ciascun termine.
* Quindi combinare le prove tra i termini.

### Calcolo di varianti pesate di frequenza dei termini totali e lunghezza del documento

Si calcolano varianti pesate della frequenza dei termini totali e della lunghezza del documento come segue:

$$
\begin{aligned}
\tilde{t} f_{i} &= \sum_{z=1}^{Z} v_{z} t f_{z i} \\
\tilde{dl} &= \sum_{z=1}^{Z} v_{z} l e n_{z}
\end{aligned}
$$

dove:

* $v_z$ è il peso della zona;
* $tf_{zi}$ è la frequenza del termine nella zona $z$;
* $len_z$ è la lunghezza della zona $z$;
* $Z$ è il numero di zone.

Si calcola anche la lunghezza del documento media ponderata:

$$
\begin{aligned}
avdl &= \frac{\text{average } d\tilde{l}}{\text{across all docs}}
\end{aligned}
$$

## Metodo per il calcolo delle varianti pesate

Il metodo per il calcolo delle varianti pesate si articola in tre fasi:

- **Calcolo della TF per zona:** La frequenza dei termini (TF) viene calcolata separatamente per ogni zona del documento.

- **Normalizzazione per zona:** La TF viene normalizzata in base alla lunghezza della zona.

- **Peso della zona:** A ciascuna zona viene assegnato un peso ($v_z$), che riflette la sua importanza nel contesto del documento. Questo peso è un parametro predefinito e non è apprendibile dal modello.

### Simple BM25F con zone

Interpretazione semplificata: la zona *z* è "replicata" *y* volte.

La formula per il punteggio RSV (Retrieval Status Value) nel modello Simple BM25F è:

$$RSV^{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i} $$

Tuttavia, si potrebbero voler utilizzare parametri specifici per ogni zona (k, b, IDF).

## Normalizzazione della lunghezza specifica per zona

Empiricamente, si è riscontrato che la normalizzazione della lunghezza specifica per zona (ovvero, *b* specifico per zona) è utile. La frequenza del termine modificata ($\tilde{tf}_i$) viene calcolata come:

$$
\tilde{tf}_i = \sum_{z=1}^Z v_z \frac{f_{z i}}{B_z}
$$

dove:

$$
B_z = \left( (1-b_z) + b_z \frac{\text{len}_z}{\text{avlen}_z} \right), \quad 0 \leq b_z \leq 1
$$

e `len_z` rappresenta la lunghezza della zona z e `avlen_z` la lunghezza media delle zone z.

La formula per il punteggio RSV nel modello BM25F, considerando la normalizzazione specifica per zona, è:

$$
\text{RSV}^{BM25F} = \sum_{i \in q} \log \frac{N }{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_{1}+tf_{i}}
$$

Si noti che questa formula differisce leggermente dalla formula del Simple BM25F, semplificando il denominatore. La differenza principale risiede nell'utilizzo di una normalizzazione della lunghezza specifica per zona, rappresentata da $B_z$.

## Classifica con caratteristiche non testuali

### Assunzioni

* **Assunzione di indipendenza usuale:**
	* Le caratteristiche non testuali sono indipendenti l'una dall'altra e dalle caratteristiche testuali.
	* Questa assunzione consente di separare il fattore nella derivazione in stile BIM:

$$\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$$

* **Le informazioni di rilevanza sono indipendenti dalla query:**
	* Questa assunzione è generalmente vera per caratteristiche come PageRank, età, tipo, ecc.
	* Consente di mantenere tutte le caratteristiche non testuali nella derivazione in stile BIM, dove vengono eliminati i termini non relativi alla query.

### Formulazione del RSV

Il RSV (Ranking Score Value) può essere calcolato come segue:

$$RSV=\sum_{i\in q}c_{i}(tf_{i})+\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$$

dove:

* $V_{f}(f_{j})=\log\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$
* $\lambda$ è un parametro libero aggiunto artificialmente per tenere conto delle ridimensionamenti nelle approssimazioni.

### Considerazioni sulla selezione di $V_j$

È necessario prestare attenzione nella selezione di $V_j$ a seconda di $f_j$. Ad esempio, la scelta di $V_j$ può spiegare perché $Rsv_{bm25} + log(\text{pagerank})$ funziona bene.
