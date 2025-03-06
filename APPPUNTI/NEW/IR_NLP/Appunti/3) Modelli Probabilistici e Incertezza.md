### Topic Modeling

Il **Topic Modeling**, in particolare il **Stochastic Topic Modeling**, ha avuto un grande impatto nella prima decade degli anni 2000. Si tratta di un'espressione generica che indica la modellazione e la rappresentazione dei dati testuali. La differenza principale con i modelli precedenti risiede nel fatto che il **Topic Modeling** rappresenta un **topic** come una **distribuzione di probabilità sullo spazio dei termini**. Allo stesso tempo, un documento è visto come una **miscela di distribuzioni di probabilità dei topic**.

### Basi di Conoscenza Lessicali

Le **Basi di Conoscenza Lessicali** sono un elemento importante del **Knowledge Management**. In particolare, ci concentreremo su **WordNet**, una risorsa lessicale che fornisce informazioni semantiche sulle parole. Verranno presentati spunti di ricerca recenti che utilizzano le basi di conoscenza lessicali per testare le capacità di comprensione dei Language Models in specifici task.

### Ritorno al Vector Space Model

Il **Vector Space Model** riemergerà nell'ambito dell'Information Retrieval. Vedremo un modello e le sue proprietà, insieme a una generalizzazione che rappresenta la versione probabilistica della funzione TF, specificamente per i task di retrieval.

**Premessa:** Un sistema tradizionale di Information Retrieval richiede un meccanismo di matching, ovvero un sistema di **ranking**.

## Incertezza nella Ricerca di Informazioni

Quando si parla di ricerca di informazioni, è importante considerare che la query dell'utente non sempre corrisponde a una richiesta precisa e chiara. Questo perché la traduzione in linguaggio naturale dell'esigenza informativa può essere imprecisa, soprattutto in termini di obiettivi di ricerca.

Per affrontare questa incertezza, è necessario introdurre una nozione di **incertezza** nei sistemi di recupero delle informazioni.

#### Un esempio dal Data Mining

Questa incertezza è affrontata introducendo modelli probabilistici, analogamente a quanto avviene nell' "uncertainty data mining", dove l'incertezza nei dati, ad esempio quelli sensoriali, viene gestita tramite approcci probabilistici.

##### Incertezza nelle Misurazioni Sensoriali

Ad esempio, i sensori ottici utilizzati per la rilevazione del particolato atmosferico sono più economici di quelli pneumatici, ma sono anche più sensibili all'umidità, portando a sovrastima. Questo evidenzia la difficoltà di ottenere valori precisi nelle misurazioni sensoriali.

### Introduzione all'Incertezza nei Dati

In un contesto di analisi dati, come la classificazione o il retrieval di informazioni, spesso si assume che i dati siano rappresentati da valori numerici. Tuttavia, è importante considerare che questi valori numerici possono essere affetti da incertezza.

Un modo per gestire questa incertezza è associare a ciascun valore una distribuzione di probabilità univariata, ad esempio una distribuzione gaussiana con media pari al valore stesso. In questo modo, si passa da vettori multidimensionali a insiemi di distribuzioni di probabilità, rendendo l'analisi più complessa ma anche più realistica.

### Strumenti per la Gestione dell'Incertezza

Per lavorare con distribuzioni di probabilità, si possono utilizzare strumenti come la divergenza di *Shannon-Jensen*, che fornisce una misura di distanza tra distribuzioni.

L'introduzione dell'incertezza nei dati è particolarmente importante in contesti come il data mining, il knowledge management e l'information retrieval. Questo perché l'incertezza permette di creare sistemi di retrieval più raffinati, in grado di soddisfare richieste complesse e di identificare il miglior set di risultati (result set) composto dai documenti più rilevanti.

## Modelli di Retrieval Probabilistici

Esistono diversi modelli di retrieval probabilistici, tra cui:

* **Probability Ranking Principle:** Questo principio, estremamente intuitivo, è alla base di molti modelli di retrieval. *Classifica i documenti in base alla probabilità di pertinenza*.
* **Binary Independence Model:** Questo modello presenta affinità con la teoria bayesiana e con il concetto di probabilità a priori. *Assume che i termini siano indipendenti l'uno dall'altro*.
* **Modello Okapi BM25:** Questo modello è una versione probabilistica della TF-IDF, utilizzata per il retrieval di informazioni, *tiene conto della frequenza dei termini e della lunghezza del documento*.
* **Reti bayesiane per il recupero di testo:** un approccio più generale che *consente di modellare le dipendenze tra i termini*. In generale, l'applicazione dell'inferenza bayesiana è fondamentale per la gestione dell'incertezza nei modelli di retrieval.

## Modelli Probabilistici per il Ranking di Documenti

L'utilizzo di modelli probabilistici per il ranking di documenti ha avuto un'evoluzione significativa negli ultimi anni. Sebbene questi modelli fossero già considerati "matematicamente eleganti" negli anni 2000, la loro applicazione pratica era limitata a causa delle tecnologie disponibili. Oggi, grazie a tecnologie più efficienti e accurate, i modelli probabilistici sono diventati uno strumento fondamentale per il ranking di documenti.

### Principio di Ranking Probabilistico:

Il principio di base è quello di **determinare il miglior set di risultati (result set) per una query**, composto da documenti altamente rilevanti. Questo implica non solo l'identificazione dei documenti rilevanti, ma anche il loro **ordinamento per rilevanza**.

##### Approccio Probabilistico:

L'approccio probabilistico al ranking di documenti si basa sul concetto di **probabilità di rilevanza di un documento rispetto a una query**. Invece di utilizzare euristiche per determinare la rilevanza, si cerca di **misurare la probabilità che un documento appartenga alla classe "rilevante" (R)**.

##### Formalizzazione del Problema:

Il problema può essere formalizzato come un problema di **classificazione**. Si vuole determinare la probabilità che un documento **X** appartenga alla classe **R** (rilevante) piuttosto che alla classe **non R** (non rilevante).

##### Notazione:

* **X:** Rappresentazione vettoriale del documento.
* **R:** Classe "rilevante".
* **R=1** significa "pertinente" e **R=0** "non pertinente"
* **P(R=1|X):** Probabilità che il documento X appartenga alla classe R.

##### Formula di Bayes:

La probabilità di rilevanza può essere calcolata utilizzando la formula di Bayes:

$$P(R=1|X) = \frac{{P(X|R=1) \cdot P(R=1)}}{P(X)}$$

Dove:

* **P(X|R=1):** Likelihood, ovvero la probabilità di osservare il documento X dato che è rilevante.
* **P(R=1):** Probabilità a priori che un documento sia rilevante.
* **P(X):** Probabilità di osservare il documento X.

##### Osservazioni:

* Il termine **P(X)** è costante rispetto alle classi e quindi non influenza il ranking.
* Il ranking dei documenti dipende quindi dai termini **P(X|R=1)** e **P(R=1)**.
## Classificazione e il Problema del Bias

L'assunzione di indipendenza condizionata può essere problematica nei dati testuali, a causa delle relazioni di dipendenza tra le dimensioni. È importante considerare questo aspetto quando si applica la classificazione a dati testuali.

Il problema della classificazione si basa sulla formula di *price*, dove il denominatore rimane costante indipendentemente dalla classe o dalla distanza.

##### Il termine di *likelihood*

- Il termine computazionalmente complesso nel contesto del *DICE* (con due sole classi) è il termine di *likelihood*. Per stimarlo, si utilizza la *maximum likelihood estimate*, che corrisponde alla frequenza relativa.
- Dato una classe *J*, la probabilità di osservare *X* (un'istanza specifica) si basa sull'assunzione di **indipendenza condizionata** tra le dimensioni. In altre parole, si assume che le dimensioni siano indipendenti tra loro, data la classe *J*.

##### Stima delle probabilità *a priori*

- Le probabilità *a priori* delle classi (ad esempio, probabilità di classe rilevante e probabilità di classe non rilevante) possono essere stimate osservando un *training set*.

##### Il problema del *bias*

- Il problema del *bias* applicato alla classificazione (sia per il testo che per altri tipi di dati) è particolarmente importante.

##### Validità dell'assunzione di indipendenza

- Questa assunzione può essere valida se la fase di *pre-process* ha ridotto la dimensionalità in modo da valorizzare l'indipendenza delle dimensioni. Tuttavia, in alcuni scenari, come quello dei dati testuali, l'assunzione di indipendenza è difficile da accettare.

##### Dipendenza nei dati testuali

- Nei dati testuali, le dimensioni sono spesso dello stesso tipo (ad esempio, parole) e presentano relazioni di dipendenza. Ad esempio, gli articoli sono dipendenti dai sostantivi, e la grammatica influenza le relazioni tra i termini.

## Probability Ranking e Binary Independence Model

Come dimostra il classificatore naïve bayesiano, è possibile ottenere buone prestazioni in determinati task anche con una risoluzione a "grana grossa". La chiave sta nella capacità di descrivere efficacemente gli elementi rilevanti per distinguerli da quelli non rilevanti. Questo concetto si applica anche all'information retrieval.

### Probability Ranking (PRP)

L'obiettivo del PRP è ordinare i documenti in base alla loro probabilità di rilevanza. Questo approccio si allinea con la regola di decisione ottimale dell'inferenza bayesiana: un documento è rilevante se la probabilità che soddisfi l'ipotesi di rilevanza è maggiore della probabilità che non la soddisfi.

##### Bisogna trovare p(R=1|x), ovvero la probabilità che un documento x sia pertinente.

* **p(R=1), p(R=0):** probabilità a priori di recuperare un documento pertinente (non pertinente) a caso.
* **p(x|R=1), p(x|R=0):** probabilità che, se viene recuperato un documento pertinente (non pertinente), sia x.

Formalmente, un documento D è rilevante rispetto a una query Q se e solo se:
$$P(R=1|D,Q) > P(R=0|D,Q)$$
Dove:

* **R=1** indica che il documento è rilevante.
* **R=0** indica che il documento non è rilevante.

Adottare il ranking probabilistico minimizza il rischio bayesiano, ovvero la loss 1-0/0-1, che corrisponde all'error rate.

### Binary Independence Model

Nel modello BIM, i documenti sono rappresentati da vettori binari di incidenza dei termini:

* **x** = vettore binario del documento.
* **x<sub>i</sub> = 1** se il termine i-esimo è presente nel documento x, altrimenti **x<sub>i</sub> = 0**.

Il modello assume l'indipendenza tra i termini, quindi la presenza di un termine in un documento non influenza la presenza di altri termini. Le query sono trattate come vettori di termini.

Per ogni documento D e query Q, il BIM calcola la probabilità di rilevanza $P(R=1|D,Q)$ e di non rilevanza $P(R=0|D,Q)$ per determinare l'ordinamento dei documenti.
L'obiettivo è solo il ranking

## Applicazione del Teorema di Bayes per il Ranking dei Documenti

Come già visto con altre applicazioni del teorema di Bayes, non siamo interessati al punteggio (score) o alle probabilità in sé, ma al **ranking** dei documenti.

La novità rispetto alla classificazione Bayesiana è che questa volta ricorreremo a **O(R|QX)**, ovvero la probabilità di osservare la classe **R** dato il documento **X** e la query **Q**.

**O(R|QX)** è un rapporto tra la probabilità che il documento **X** sia della classe **R** (R = 1) e la probabilità che il documento **X** non sia della classe **R** (R = 0). In formule:

$$O(R|Q\vec{X}) = \frac{P(R=1|Q\vec{X})}{P(R=0|Q\vec{X})}$$

Per ognuno di questi termini applichiamo il teorema di Bayes.

### Odds

Gli odds sono una misura di probabilità che rappresenta il rapporto tra la probabilità di un evento e la probabilità del suo complemento. In questo caso, gli odds ci aiutano a stabilire un ordinamento tra i documenti piuttosto che assegnare loro un punteggio.

Gli odds ci aiutano a rimanere concentrati sull'analisi specifica per stabilire un ordinamento tra i documenti.

##### Cosa sono gli Odds Ratio (OR)?

Gli Odds Ratio (OR) sono una misura di associazione relativa tra due variabili binarie. Sono utilizzati in statistica descrittiva per misurare la forza di associazione o di non indipendenza tra due variabili.

Consideriamo una tabella di contingenza 2x2 con le variabili X e Y:

| | Y=1 | Y=0 | Totale |
|--------|-----|-----|--------|
| X=1 | n11 | n10 | n1. |
| X=0 | n01 | n00 | n0. |
| Totale | n.1 | n.0 | N |

L'Odds Ratio è calcolato come:

$$OR = \frac{n_{11} \cdot n_{00}}{n_{10} \cdot n_{01}}$$

##### Applicazioni degli Odds Ratio:

Gli Odds Ratio sono utilizzati in diversi campi, tra cui:

* **Epidemiologia:** per stabilire il rischio di occorrenza di un evento relativo a diversi soggetti e popolazioni. Ad esempio, qual è il rischio di incidenza di una malattia rispetto ad un insieme di soggetti esposti a determinati fattori di rischio?
* **Ricerca sociale:** per studiare l'associazione tra variabili sociali.
* **Marketing:** per analizzare l'efficacia di campagne pubblicitarie.

## Rischio Relativo e Odds Ratio

Sono due misure utilizzate per valutare l'associazione tra un evento (ad esempio, lo sviluppo di una malattia) e una variabile (ad esempio, l'assunzione di un nuovo trattamento medico).

### Rischio Relativo

Il rischio relativo (RR) misura la probabilità di un evento in presenza di una variabile rispetto alla probabilità dello stesso evento in assenza della variabile.

##### Formula:

RR = (Probabilità di osservare X in presenza di Y) / (Probabilità di osservare X in assenza di Y)

##### Interpretazione:

* **RR = 1:** Non c'è differenza nel rischio tra l'occorrenza e la non-occorrenza di X.
* **RR < 1:** L'evento Y è meno probabile che accada in presenza di X.
* **RR > 1:** L'evento Y è più probabile che accada in presenza di X.

### Odds Ratio

L'odds ratio (OR) è un'altra misura di associazione che confronta le probabilità di un evento in due gruppi diversi.

##### Formula:

OR = (Odds di Y in presenza di X) / (Odds di Y in assenza di X)

##### Interpretazione:

* **OR = 1:** Non c'è differenza nelle odds tra i due gruppi.
* **OR < 1:** Le odds di Y sono più basse nel gruppo con X.
* **OR > 1:** Le odds di Y sono più alte nel gruppo con X.

### Relazione tra Rischio Relativo e Odds Ratio

Il rischio relativo e l'odds ratio sono misure correlate. In alcuni casi, l'odds ratio può essere utilizzato come una buona approssimazione del rischio relativo, soprattutto quando la prevalenza dell'evento è bassa.

### Esempio

Consideriamo l'incidenza di una malattia in soggetti esposti a un determinato fattore e in soggetti non esposti. L'odds ratio (OR) rispetto all'evento Y (incidenza della malattia) può essere calcolato come segue:

OR = (Odds di Y in presenza di X) / (Odds di Y in assenza di X)

dove:

* X = Esposizione al fattore
* Y = Incidenza della malattia
* θ1 = Odds di Y in presenza di X
* θ2 = Odds di Y in assenza di X

## Odds Ratio e Rischio Relativo: Un Confronto

La **Odds Ratio (OR)** è una misura di associazione tra un fattore di rischio e un evento. È definita come il rapporto tra le odds di un evento nel gruppo esposto al fattore di rischio e le odds dello stesso evento nel gruppo non esposto.

##### Proprietà importanti dell'Odds Ratio:

* **Non simmetrica:** L'OR non è simmetrica, ovvero scambiando le variabili X e Y, il valore dell'OR cambia.
* **Robusta:** L'OR può essere stimata anche in assenza delle incidenze, rendendola utile per studi retrospettivi.
* **Approssimazione del rischio relativo:** Per eventi rari, l'OR approssima molto bene il **rischio relativo (RR)**.

##### Rischio Relativo (RR):

Il RR è una misura di associazione che indica il rapporto tra il rischio di un evento nel gruppo esposto al fattore di rischio e il rischio dello stesso evento nel gruppo non esposto.

##### Confronto tra OR e RR:

* **Studi retrospettivi:** L'OR è preferibile al RR perché può essere stimata anche in assenza delle incidenze.
* **Studi prospettivi:** Il RR è preferibile all'OR perché fornisce una misura diretta del rischio.
* **Eventi rari:** L'OR approssima molto bene il RR per eventi rari.

##### Esempio:

Consideriamo uno studio epidemiologico che analizza l'associazione tra il fumo e il rischio di sviluppare una malattia.

* **Gruppo esposto:** Fumatori
* **Gruppo non esposto:** Non fumatori
* **Evento:** Sviluppo della malattia

Supponiamo che l'incidenza della malattia sia del 30% nel gruppo dei fumatori e del 10% nel gruppo dei non fumatori.

##### Calcolo dell'OR:

* **Odds di malattia nel gruppo dei fumatori:** 30/70 = 0.43
* **Odds di malattia nel gruppo dei non fumatori:** 10/90 = 0.11
* **OR:** 0.43 / 0.11 = 3.91

##### Interpretazione:

L'OR di 3.91 indica che i fumatori hanno un rischio di sviluppare la malattia 3.91 volte maggiore rispetto ai non fumatori.

##### Applicazione in contesti epidemiologici:

L'OR è un'utile misura di associazione per studiare l'impatto di fattori di rischio su eventi rari. Ad esempio, può essere utilizzata per valutare l'associazione tra l'esposizione a determinate condizioni ambientali e il rischio di sviluppare una malattia.

##### Nota:

L'assunzione di sparsità, ovvero la rarità degli eventi, è fondamentale per l'approssimazione dell'OR al RR. In caso di eventi frequenti, l'OR può sovrastimare il RR.

## OTS-I e Likelihood Ratio

L'OTS-I (Odd of Term Significance - I) è una query che, rispetto alla presenza o meno dell'ipotesi di rilevanza (gruppo dei rilevanti vs. gruppo dei non rilevanti), è equivalente a TOS (Odd of Term Significance). La probabilità di osservare un dato Q, data l'ipotesi di rilevanza, è calcolata come:
$$\frac{PR(Q | R=1)}{PR (Q | R=0)}$$
Dove:

* **PR (Q | R=1)** è la probabilità di osservare il dato Q dato che l'ipotesi di rilevanza è vera (R=1).
* **PR (Q | R=0)** è la probabilità di osservare il dato Q dato che l'ipotesi di rilevanza è falsa (R=0).

Questo rapporto è un fattore chiave nell'OTS-I. L'altro fattore è la **likelihood**, che rappresenta la probabilità di osservare un dato Q, data l'ipotesi di rilevanza o non rilevanza.

**Nota:** La likelihood è costante per ogni termine, quindi il focus principale è sulla stima del rapporto tra le probabilità di osservare un dato Q, dato che l'ipotesi di rilevanza è vera o falsa.

### Applicazione dell'ipotesi di indipendenza

Sotto l'ipotesi di indipendenza tra i termini, il rapporto di likelihood si trasforma in una produttoria di rapporti di likelihood per ogni termine:

$$OTS-i(Q)=
\frac{p(\vec{x} \mid R = 1, q)}{p(\vec{x} \mid R = 0, q)} = \prod_{i = 1}^{n} \frac{p(x_i \mid R = 1, q)}{p(x_i \mid R = 0, q)}
$$
Dove:

* **X<sub>i</sub>** rappresenta la presenza o assenza del termine i-esimo nel documento.

### Stima della produttoria di likelihood

Per stimare la produttoria di likelihood, si divide la produttoria in due parti:

- **Produttoria per i termini presenti nel documento:**
$$\begin{aligned}
\prod_{x_{i}=1} \frac{p\left(x_{i}=1 \mid R=1,q\right)}{p\left(x_{i}=1 \mid R=0,q\right)}
\end{aligned}$$
- **Produttoria per i termini assenti nel documento:**
$$\begin{aligned}
\prod_{x_{i}=0} \frac{p\left(x_{i}=0 \mid R=1,q\right)}{p\left(x_{i}=0 \mid R=0,q\right)}
\end{aligned}$$
### Notazione

Per semplificare la notazione, si utilizzano le seguenti abbreviazioni:

* **P<sub>i</sub>**: Likelihood rispetto alla presenza del termine i-esimo, data l'ipotesi di rilevanza.
* **R<sub>i</sub>**: Likelihood rispetto alla presenza del termine i-esimo, data l'ipotesi di non rilevanza.

Quindi, la produttoria di likelihood può essere riscritta come $\prod_{i=1}^{n} \left( \frac{P_i}{R_i} \right)$

## Probabilità di Osservazione dei Termini in un Documento

Questo testo tratta il concetto di probabilità di osservare un termine specifico in un documento, considerando due diverse assunzioni: **rilevanza** e **novità**.

##### Definizioni:

* **R con i:** Probabilità di osservare il termine iesimo nel documento, assumendo la rilevanza del documento.
* **t con i:** Probabilità di osservare il termine iesimo nel documento, assumendo la novità del termine.

##### Tabella di Contingenza:

La tabella di contingenza mostra la relazione tra le variabili "presenza/assenza del termine" e "rilevanza/non rilevanza" del documento.

| | Rilevanza (r=1) | Non Rilevanza (r=0) |
| ------------------ | --------------- | ------------------- |
| Presenza $(x_i=1)$ | $p_i$ | $r_i$ |
| Assenza $(x_i=0)$ | $(1-p_i)$ | $(1-r_i)$ |

##### Interpretazione:

* $p_i$: Probabilità di osservare il termine iesimo nel documento, dato che il documento è rilevante.
* $r_i$: Probabilità di osservare il termine iesimo nel documento, dato che il documento non è rilevante.
* $(1-p_i)$: Probabilità di non osservare il termine iesimo nel documento, dato che il documento è rilevante.
* $(1-r_i)$: Probabilità di non osservare il termine iesimo nel documento, dato che il documento non è rilevante.

##### Odds:

Rappresenta il rapporto tra la probabilità di un evento e la probabilità del suo complemento. In questo caso, gli odds vengono calcolati per tutti i termini della query, distinguendo tra **matching** e **non matching**.

* **Matching:** Il termine iesimo è presente sia nella query che nel documento (x con i = 1, q con i = 1).
* **Non Matching:** Il termine iesimo è presente nella query ma non nel documento (x con i = 0, q con i = 1).

##### Assunzioni:

* Il modello ignora completamente l'assenza di termini nella query (q con i = 0).
* La probabilità di occorrenza di un termine in un documento non viene valutata per i termini non presenti nella query.

## Derivazione della Formula Finale

Siamo arrivati a questo punto. Riprendiamo l'ultima formula ottenuta:

$$
\begin{aligned}
O(R \mid q, \tilde{x}) = O(R \mid q) \cdot \prod_{x_{i}=1} \frac{p(x_{i}=1 \mid R=1,q)}{p(x_{i}=1 \mid R=0,q)} \cdot \prod_{x_{i}=0} \frac{p(x_{i}=0 \mid R=1,q)}{p(x_{i}=0 \mid R=0,q)}
\end{aligned}
$$

Distinguiamo due parti importanti:

- **Produttoria dei termini "match":** Rappresenta i termini in cui il documento è rilevante per la query.
- **Produttoria dei termini "non-match":** Rappresenta i termini in cui il documento non è rilevante per la query.

Analizziamo ora l'ultimo passaggio aritmetico. Introduciamo una produttoria fittizia iterando su tutti i termini "match" (x con i = 1):
$$
\begin{aligned}
O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i = 1 \\ q_i = 1}} \frac{p_i}{r_i} \cdot \prod_{\substack{x_i = 0 \\ q_i = 1}} \frac{1 - p_i}{1 - r_i}
\end{aligned}
$$
Scomponiamo questi termini e li ridistribuiamo nelle due produttorie originali:

* Alcuni termini della produttoria fittizia vanno nella **prima produttoria** per formare un **odds ratio**:
* P / (1 - P): Probabilità di osservare il termine dato che il documento è rilevante, diviso la probabilità di osservare il termine dato che il documento non è rilevante.
* R / (1 - R): Probabilità di osservare il termine dato che il documento non è rilevante, diviso la probabilità di non osservare il termine dato che il documento non è rilevante.
* Nella **prima produttoria** abbiamo ora un **odds ratio** per ogni termine "match".
* Dalla produttoria fittizia rimane la quantità `1 con i / 1 con i`, identica al rapporto presente nella **seconda produttoria** (termini "non-match").

Possiamo quindi esprimere la produttoria finale come:
$$
\begin{aligned}
O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i=q_i=1}} \frac{p_i(1-r_i)}{r_i(1-p_i)} \cdot \prod_{\substack{q_i=1 }} \frac{1-p_i}{1-r_i}
\end{aligned}
$$

Questa produttoria finale considera sia i termini "match" che "non-match". I termini "non-match" erano già presenti nella seconda produttoria originale, mentre i termini "match" sono stati inclusi tramite la scomposizione della produttoria fittizia.

Ricapitolando, abbiamo ottenuto:

* **Odds ratio di R dato il primo termine:** Rappresenta la probabilità di rilevanza del documento dato il primo termine della query.
* **Produttoria sui termini indipendenti:** Rappresenta l'influenza dei termini che non dipendono dalla rilevanza del documento.

## Retrieval Status Value (RSV)

Il **Retrieval Status Value (RSV)** è un punteggio che viene assegnato a un documento per determinare la sua rilevanza rispetto a una query di ricerca. Questo punteggio si basa sul calcolo del logaritmo della produttoria degli **odds ratio** per ogni termine della query che corrisponde a un termine nel documento.

##### Calcolo dell'RSV:

- **Produttoria degli odds ratio:** Per ogni termine della query che corrisponde a un termine nel documento, si calcola l'**odds ratio**. L'odds ratio è il rapporto tra la probabilità che il termine appaia nel documento e la probabilità che appaia in un documento casuale.
- **Sommatoria dei logaritmi:** Si calcola la sommatoria dei logaritmi degli odds ratio per tutti i termini corrispondenti.
- **RSV:** Il risultato della sommatoria dei logaritmi è il **Retrieval Status Value (RSV)**.

##### Esempio:

Supponiamo di avere una query con due termini: "X" e "Q". Un documento contiene entrambi i termini.

* **Odds ratio per "X":** 1
* **Odds ratio per "Q":** 1

Il **RSV** per questo documento sarebbe:

```
log(1) + log(1) = 0
```

##### Interpretazione dell'RSV:

Un RSV più alto indica una maggiore rilevanza del documento rispetto alla query. Un RSV basso indica una minore rilevanza.

##### Stima delle probabilità:

Per calcolare gli odds ratio, è necessario stimare le probabilità di occorrenza dei termini nel documento e in un documento casuale. Questo può essere fatto utilizzando tecniche di stima della probabilità come la **smoothing di Laplace**.

## Analisi dei Dati di Addestramento dei Modelli Linguistici

L'analisi dei dati di addestramento dei modelli linguistici è un campo di ricerca attuale e complesso. Una sfida particolarmente impegnativa è la **"inversione del modello"**: risalire ai dati di addestramento originali a partire dal modello stesso e dai suoi output.

Seppur estremamente difficile, una simile scoperta avrebbe un impatto rivoluzionario. Il fatto che gruppi con ingenti risorse finanziarie stiano investendo in questa direzione è un segnale positivo dell'importanza di questa ricerca.

Attualmente, la **trasparenza** riguardo ai dati di addestramento dei modelli linguistici è limitata, specialmente per i modelli open source. Dichiarare genericamente che i dati provengono da "web, prosa e poesia" non è sufficiente: è necessario un livello di dettaglio molto più elevato.

Un'analisi efficace richiederebbe l'**addestramento di un modello specifico** in grado di riconoscere il contesto dei dati di addestramento originali. Questo processo è a sua volta molto complesso e necessita di dataset di addestramento altamente specifici.

## Simboli e Definizioni

Per comprendere meglio i concetti di NLP, è necessario introdurre alcuni simboli e definizioni:

* **RSV:** Questi simboli rappresentano il logaritmo del produttore di all special.
$$
RSV = \log \prod_{x_i=q_i=1} \frac{p_i(1-r_i)}{r_i(1-p_i)} = \sum_{x_i=q_i=1} \log \frac{p_i(1-r_i)}{r_i(1-p_i)}
$$
* **i:** Rappresenta il numero di termini, ovvero la dimensione del vocabolario.
* **n s N S:** Questi simboli sono utilizzati per indicare la dipendenza sia a livello di rappresentazione di testi che di quelli di binariet.
$$
RSV = \sum_{x_i=q_i=1} c_i; \quad c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}
$$

## Calcolo dell'Oz Ratio: Un Approfondimento

In questo contesto, stiamo analizzando un insieme di documenti, dove **N** rappresenta il numero totale di documenti. Il nostro obiettivo è determinare la rilevanza di un termine all'interno di questi documenti.

##### Definizioni:

* **S:** Numero totale di documenti rilevanti.
* **n:** Numero di documenti in cui un termine specifico è presente.
* **s:** Numero di documenti rilevanti che contengono il termine specifico.

##### Tabellina di Contingenza:

| Documents | Relevant | Non-Relevant | Total |
| --------- | -------- | ------------ | ----- |
| $x_i=1$ | s | n-s | n |
| $x_i=0$ | S-s | N-n-S+s | N-n |
| **Total** | **S** | **N-S** | **N** |

##### Calcolo delle Probabilità:

$$
p_i = \frac{s}{S} \quad \text{e} \quad r_i = \frac{(n-s)}{(N-S)}
$$
* **P(I):** Probabilità che il termine sia presente dato che il documento è rilevante.
* **R(I):** Probabilità che il termine sia presente dato che il documento non è rilevante.

##### Oz Ratio:

L'Oz Ratio è una misura che quantifica la forza dell'associazione tra un termine e la rilevanza di un documento. Si calcola come segue:
$$c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$$

##### Espressione in termini di Count:

Possiamo esprimere l'Oz Ratio in termini di count (n, s, N, S) sostituendo le probabilità P(I) e R(I) con le loro espressioni in termini di count:
$$
c_i = K(N,n,S,s) = \log \frac{s/(S-s)}{(n-s)/(N-n-S+s)}
$$
## $r_i$ e l'approssimazione dei non rilevanti

Partiamo da $r_i$, che rappresenta la probabilità di incidenza del termine di ricerca in documenti non rilevanti.

**Ipotesi:** È ragionevole assumere che l'insieme dei documenti non rilevanti possa essere approssimato dall'intera collezione. Questo perché i documenti rilevanti sono generalmente una piccola porzione rispetto all'intero insieme dei documenti, soprattutto in contesti come il web.

**Conseguenze:** Se approssimiamo i non rilevanti con l'intera collezione, R con I diventa la frazione N piccolo su N grande, dove:

* **N grande:** numero totale di documenti nella collezione.
* **N piccolo:** numero di documenti che contengono il termine di ricerca.

Quindi, $r_i$ è approssimabile a N piccolo diviso N grande, ovvero la frazione dei documenti nella collezione totale che contengono il termine di ricerca.

##### Oz di competenza:

$$\log \frac{1-r_i}{r_i} = \log \frac{N-n-S+s}{n-s} \approx \log \frac{N-n}{n} \approx \log \frac{N}{n} = IDF$$

**Approssimazione successiva:** N grande - N piccolo può essere ulteriormente approssimato a N grande, perché N piccolo è una porzione trascurabile rispetto a N grande.

**Validità dell'approssimazione:** Questa approssimazione è valida soprattutto in sistemi aperti come il web, dove la collezione di documenti è molto ampia. In corpus tematici, l'approssimazione potrebbe essere meno accurata a causa della presenza di termini altamente frequenti (legge di Zipf).

## Probabilità e Smoothing

È raro che un sostantivo o un verbo (termini di contenuto) appaia in un numero elevato di documenti. Il logaritmo del numero di documenti rilevanti in cui appare un termine diviso per il numero totale di documenti (log(n grande / n piccolo)) rappresenta la **norma inversa del termine**.

##### Incertezza e Probabilità

Stiamo trattando l'informazione in modo probabilistico, introducendo incertezza. Questo significa che non abbiamo un valore fisso, ma dobbiamo muoverci in un intorno. La **TFPF (Term Frequency-Probability Factor)** è una misura robusta perché è coerente con la legge di Zipf.

##### Smoothing

Introduciamo dei **parametri di smoothing** per evitare che il modello risponda in modo troppo generoso, amplificando eccessivamente le probabilità. Lo smoothing ha due effetti principali:

* **Penalizza gli eventi frequenti:** Riduce la probabilità di eventi che si verificano spesso.
* **Fa emergere eventi rari:** Aumenta la probabilità di eventi che si verificano raramente.

##### Smoothing nella Probabilità

Nella teoria della probabilità, in particolare nell'inferenza bayesiana, lo smoothing viene utilizzato per evitare che il modello si affidi eccessivamente ai dati osservati.

La formula di probabilità condizionale è la seguente: $$ P_{i}^{(h+1)} = \frac{\left|V_{i}\right|+\kappa p_{i}^{(h)}}{\left|V\right|+K} $$

* **$P_i^{(h+1)}$:** Probabilità di osservare l'evento $i$ all'iterazione $h+1$.
* **$|V_i|$:** Cardinalità dell'insieme $B$ contenente l'evento $i$. In altre parole, numero di volte in cui l'evento $i$ è presente nell'insieme $B$.
* **$|V|$:** Cardinalità dell'insieme $B$, ovvero il numero totale di elementi in $B$.
* **$K$:** Parametro di smoothing, utilizzato per gestire eventi non presenti in $B$ ed evitare probabilità nulle.
* **$p_i^{(h)}$:** Probabilità di osservare l'evento $i$ all'iterazione $h$.

##### Effetti dello Smoothing

Lo smoothing introduce dei **fattori nascosti** che influenzano la probabilità di occorrenza di un termine. Questi fattori nascosti rappresentano la correlazione o la dipendenza tra i termini stessi.

## Stima di $p_{i}$ e Relevance Feedback

### Introduzione

Dopo aver affrontato la stima di $r_{i}$, ci concentriamo ora su $p_{i}$, la likelihood di incidenza del termine nei documenti rilevanti. La stima di $p_{i}$ presenta diverse sfide e richiede l'utilizzo di diversi approcci.

### Approcci per la Stima di $p_{i}$

Esistono tre approcci principali per stimare $p_{i}$:

- **Stima da un sottoinsieme di documenti etichettati:** Questo approccio prevede la stima di $p_{i}$ da un sottoinsieme di documenti etichettati come rilevanti. Il meccanismo di feedback con l'utente o con un oracolo viene utilizzato per migliorare la stima.
- **Utilizzo di teorie e leggi:** Questo approccio si basa sull'utilizzo di teorie e leggi che consentono di approssimare $p_{i}$ in modo costante o attraverso una funzione dalla forma chiusa e nota.
- **Probabilistic Relevance Feedback:** Questo approccio si basa sul concetto di *probabilistic relevance feedback*, un'applicazione generale del feedback dell'utente per migliorare la ricerca di informazioni.

### Probabilistic Relevance Feedback

Il *probabilistic relevance feedback* si basa sul coinvolgimento dell'utente per raffinare il set di risultati (result set) e renderlo più accurato. L'utente fornisce feedback indicando quali documenti sono rilevanti e quali non lo sono, aiutando il sistema a migliorare la ricerca.

##### Obiettivo del Relevance Feedback:

L'obiettivo finale del *relevance feedback* è quello di migliorare la risposta alla query, rendendola più accurata. Questo si traduce in un intervento sulla query stessa, spesso attraverso l'espansione della query.

##### Espansione della Query:

L'espansione della query consiste nell'aggiungere nuovi termini alla query originale. Questi termini non sono scelti a caso, ma sono identificati come caratterizzanti i documenti rilevanti. Il sistema, iterazione dopo iterazione, identifica i documenti rilevanti e un meccanismo di estrazione di keyword identifica i termini importanti presenti in questi documenti. Questi termini vengono poi aggiunti alla query, migliorando la sua accuratezza.

## Relevance Feedback e Probabilistic Relevance Feedback

Il **relevance feedback** è un processo iterativo che mira a migliorare la qualità dei risultati di una ricerca. Inizialmente, il sistema restituisce un set di risultati basato su una query iniziale. L'utente fornisce un feedback, indicando quali risultati sono rilevanti e quali non lo sono. Questo feedback viene utilizzato per raffinare la query e ottenere risultati più pertinenti.

Il **fine ultimo** del relevance feedback è quello di **adattare il sistema di ricerca alle preferenze dell'utente**, fornendo risultati sempre più pertinenti e soddisfacenti.

### Probabilistic Relevance Feedback

Il **probabilistic relevance feedback** è una tecnica che utilizza un modello probabilistico per stimare la probabilità che un documento sia rilevante per una query. Questo modello si basa sulla **presenza o assenza di termini specifici** nei documenti e sulla **rilevanza dei termini** per la query.

##### Il processo di probabilistic relevance feedback prevede i seguenti passaggi:

- **Stima iniziale:** Si parte da una stima iniziale della probabilità che un documento sia rilevante (P(R|D)) e della probabilità che un termine sia rilevante (P(I|R)).
- **Identificazione dei documenti rilevanti:** L'utente identifica un set di documenti rilevanti tra quelli restituiti dal sistema.
- **Raffinamento della stima:** Il sistema utilizza il feedback dell'utente per raffinare la stima della probabilità di rilevanza dei termini e dei documenti.
- **Ricerca iterativa:** Il sistema utilizza le nuove stime per eseguire una nuova ricerca, restituendo risultati più pertinenti.

##### Il probabilistic relevance feedback si basa sul concetto di **inverso documento**, che indica la probabilità che un termine sia presente in un documento rilevante. Questa probabilità viene utilizzata per stimare la probabilità che un documento sia rilevante per una query.

##### Esempio:

Supponiamo che l'utente abbia identificato 5 documenti rilevanti tra 10 restituiti dal sistema. In questo caso, la frazione di documenti rilevanti è 5/10 = 0.5. Se il termine "informatica" è presente in 3 dei 5 documenti rilevanti, la probabilità che il termine "informatica" sia presente in un documento rilevante è 3/5 = 0.6.

##### Il probabilistic relevance feedback è un processo iterativo che consente al sistema di apprendere dalle preferenze dell'utente e di fornire risultati sempre più pertinenti.

## Approssimazione della Probabilità di Rilevanza

In questo contesto, si cerca di approssimare la probabilità di rilevanza di un termine in un documento, rappresentata dal rapporto tra la cardinalità dell'insieme dei documenti rilevanti (V) e la cardinalità dell'insieme di tutti i documenti (I).

##### Formula:

$$P(rilevanza | termine) ≈ \frac{|V|}{|I|} $$

##### Problema:

- Se l'insieme V è troppo grande, diventa computazionalmente costoso calcolare la probabilità di rilevanza.

##### Soluzione:

- Si introduce un meccanismo iterativo che raffina progressivamente l'insieme dei documenti candidati. Ad ogni passo, si stima la probabilità di rilevanza del termine in base all'insieme corrente di documenti candidati, utilizzando uno smoothing per tenere conto degli eventi non osservati.

##### Smoothing:

- Lo smoothing introduce una parte di probabilità che dipende da eventi non osservati, come la presenza o l'assenza del termine in altri documenti. Questo aiuta a evitare che la probabilità di rilevanza sia zero per termini che non sono presenti nell'insieme corrente di documenti candidati.

##### Parametro K:

- Il parametro K, utilizzato nello smoothing, è tipicamente un valore piccolo (5 o 10) e rappresenta un fattore di proporzionalità rispetto alla cardinalità dell'insieme dei documenti candidati.

##### Pseudo Relevance Feedback:

- In assenza di un feedback esplicito da parte dell'utente, si può utilizzare un feedback implicito basato sull'assunzione che i documenti in cima al ranking del sistema siano altamente rilevanti. Questo insieme di documenti viene utilizzato come base per il feedback implicito.

## Miglioramento della stima dei documenti rilevanti

Il processo di stima della rilevanza dei documenti può essere migliorato utilizzando un valore **K**, che rappresenta la cardinalità dell'insieme iniziale di documenti. Un valore **K** maggiore aiuta a ottenere una stima più accurata, evitando stime troppo basse.

Il modello **BM25**, utilizzato per il recupero delle informazioni, assegna punteggi di rilevanza ai documenti. Ad esempio, a due documenti con diverse frequenze di parole chiave, BM25 potrebbe assegnare un punteggio più alto al documento con una frequenza totale maggiore, anche se il secondo documento potrebbe essere più rilevante in base al contesto.

Questa differenza di punteggio è dovuta alla **parametrizzazione** utilizzata da BM25, che tiene conto della **normalizzazione della richiesta**.

### Esempio di Applicazione del Modello Vettoriale

##### Scenario:

Immaginiamo di avere due documenti, due "terri", con una grana molto grossa (cioè, con poche informazioni in comune).

##### Osservazioni:

* Nonostante la scarsità di informazioni, il modello vettoriale riesce a distinguere i due documenti, evidenziando la differenza nel loro trattamento.
* Questo esempio dimostra la capacità del modello vettoriale di gestire situazioni con dati limitati.

