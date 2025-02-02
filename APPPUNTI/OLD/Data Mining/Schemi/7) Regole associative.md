**Analisi delle Regole Associative:

I. **Scopo:** Identificare relazioni significative tra attributi in un dataset, scoprendo regole che indicano la co-occorrenza di elementi nelle transazioni.

II. **Concetti Chiave:**

   A. **Itemset Frequenti:** Insiemi di elementi che compaiono frequentemente insieme.
   B. **Regole di Associazione:** Relazioni tra itemset, indicando co-occorrenza (X → Y).
   C. **Variabile Binaria Asimmetrica:** Modello dove la presenza di un item è più importante della sua assenza.

III. **Itemset e Support Count:**

   A. **Definizioni:**
      1. Insieme di elementi (I):  $I = \{i_1, i_2, \dots, i_d\}$
      2. Insieme di transazioni (T): $T = \{t_1, t_2, \dots, t_n\}$
      3. Itemset (X): Sottoinsieme di I.

   B. **Metriche:**
      1. Support count (σ(X)): $\sigma(X) = |\{t_i | X \subset t_i, t_j \in T\}|$
      2. Support (s(X)): $s(X) = \frac{\sigma(X)}{N}$

   C. **Frequent Itemset:**  s(X) ≥ minsup (soglia definita dall'utente).

IV. **Regole di Associazione:**

   A. **Definizione:** X → Y (X e Y itemset disgiunti: $X \cap Y = \emptyset$)

   B. **Metriche:**
      1. **Supporto:** $S(X \cup Y) = \frac{\sigma(X \cup Y)}{N}$ (frazione di transazioni contenenti sia X che Y)
      2. **Confidenza:** $c(X \to Y) = \frac{\sigma(X \cup Y)}{\sigma(X)}$ (frequenza di Y in transazioni contenenti X)

   C. **Scopo:** Individuare regole forti (supporto e confidenza elevati rispetto alle soglie).


**Scoperta delle Regole di Associazione**

* **Obiettivo:** Trovare regole X → Y che soddisfano:
    * `Support ≥ minsup`
    * `Confidenza ≥ minconf`
* **Approccio Brute-Force:**
    * Calcola supporto e confidenza per tutte le possibili regole.
    * Numero di regole:  $R = 3^d - 2^{d+1} + 2$ (d = elementi distinti)
    * Numero di itemset: $2^d$
    * **Problema:** Complessità esponenziale, proibitivamente costoso.
* **Approccio Efficiente:**
    * **Fase 1: Generazione degli Itemset Frequenti:** Trova tutti gli itemset che soddisfano `minsup`.
    * **Fase 2: Generazione delle Regole:** Per ogni itemset frequente L, genera regole  `f → (L-f)` (f ⊂ L) con `Confidenza ≥ minconf`.
    * Sfrutta il fatto che `Support(X → Y) = Support(X ∪ Y)`.


**Generazione degli Itemset Frequenti**

* **Approccio Brute-Force:**
    * **Scopo:** Trovare tutti gli itemset frequenti.
    * **Procedura:** Genera tutti gli itemset candidati (escludendo vuoti e singoletti), conta il supporto di ogni candidato confrontandolo con ogni transazione.
    * **Complessità:** O(NMw)  (N = transazioni, M = itemset candidati ≈ $2^d$, w = dimensione massima transazione)
    * **Problema:** Complessità esponenziale in d.
* **Strategie per Ridurre la Complessità:**
    * Ridurre M (numero di candidati): Utilizzare il principio Apriori.
    * Ridurre N (numero di transazioni): Possibile per itemset grandi.
    * Ridurre NM (numero totale di confronti): Usare strutture dati efficienti.


**Principio Apriori**

* **Enunciato:** Se un itemset è frequente, tutti i suoi sottoinsiemi sono frequenti.
* **Proprietà:** Antimonotonia del supporto:  ∀X,Y: (X⊆Y) ⇒ s(X) ≥ s(Y)
* **Esempio:** Iterativo, genera itemset di dimensione crescente, scartando quelli che non raggiungono `minsup`.
* **Efficacia:** Riduce significativamente il numero di itemset candidati (es. da 41 a 13 con 6 elementi, 68% di riduzione).


**Misure di Validità delle Regole**

* **Supporto:** Frequenza di un itemset nel dataset.
* **Confidenza:** Probabilità che un itemset appaia dato un altro itemset.
* **Problema:** Numero esponenziale di itemset possibili, elevata complessità computazionale.



**Algoritmo Apriori: Scoperta di Itemset Frequenti**

I. **Obiettivo:** Trovare tutti gli *itemset* (insiemi di elementi) che appaiono in almeno una percentuale *minsup* di transazioni.

II. **Descrizione dell'Algoritmo (Iterativo):**

   A. **Fase 1: Inizializzazione:**
      1. Calcolo del supporto di ogni singolo *item* (1-itemset).
      2. Creazione di *F1*: insieme dei 1-itemset con supporto ≥ *minsup*.

   B. **Fase 2: Generazione e Pruning di Candidati:**
      1. Generazione di candidati *k*-itemset da *F<sub>k-1</sub>*.
      2. *Pruning*: Eliminazione di candidati che violano il principio Apriori (se un *itemset* è frequente, tutti i suoi sottoinsiemi devono esserlo).

   C. **Fase 3: Conteggio del Supporto:**
      1. Scansione del dataset per contare il supporto di ogni candidato *itemset*.

   D. **Fase 4: Eliminazione dei Candidati Infrequenti:**
      1. Eliminazione di candidati con supporto < *minsup*.

   E. **Fase 5: Terminazione:**
      1. L'algoritmo termina quando non vengono generati nuovi *itemset* frequenti.


IV. **Generazione di Candidati e Pruning:**

   A. **Forza Bruta:** Generazione di tutti i possibili *k*-itemset (es. $\begin{pmatrix}n \\k\end{pmatrix}$ per *n* *item* e cardinalità *k*).

   B. **Pruning:** Eliminazione di candidati contenenti almeno un *itemset* non frequente per ridurre il numero di verifiche.


**Generazione di Candidati (Apriori)**

* **Metodo  $F_{k−1}×F_{k−2}$**
    * Combina itemset frequenti di cardinalità *k-1* e *k-2*.
    * Processo:
        * Calcolo itemset frequenti di cardinalità 1 e 2.
        * Merge: Combinazione se prefissi identici.
        * Pruning: Eliminazione se contiene itemset non frequenti.
        * Conteggio supporto: Verifica frequenza itemset candidati.
* **Metodo $F_{k−1}×F_{k−1}$**
    * Combina itemset frequenti di cardinalità *k-1*.
    * Processo:
        * Merge: Combinazione se prefissi uguali.
        * Pruning: Eliminazione se contiene itemset non frequenti.
        * Conteggio supporto: Verifica frequenza itemset candidati.
* **Considerazioni generali:**
    * Alternative: tecniche basate su suffissi o combinazioni prefissi/suffissi.
    * Principio: due itemset si uniscono se differiscono di un solo elemento.
    * Vantaggi: maggiore efficienza rispetto alla forza bruta.


**Generazione di Regole Associative**

* **Partizionamento e Confidenza:**
    * Itemset frequente diviso in Corpo (lato sinistro) e Testa (lato destro).
    * Confidenza:  Frequenza(Itemset completo) / Frequenza(Corpo).
    * Esempio illustrativo di regole da {A, B, C, D}.
* **Proprietà di Anti-Monotonicità:**
    * Confidenza NON anti-monotona rispetto al numero di elementi nel corpo.
    * Confidenza anti-monotona rispetto al numero di elementi nella testa.  Esempio: $c(ABC \to D) \ge c(AB \to CD) \ge c(A \to BCD)$.
* **Pruning delle Regole:**
    * Applicazione del principio Apriori: se una regola non è frequente, le sue sotto-regole (ottenute spostando elementi dalla testa al corpo) non saranno adeguate.
* **Complessità:**
    * Il problema rimane esponenziale nonostante il pruning.


**Complessità degli Algoritmi per le Regole Associative**

I. **Sfide Principali:**
    a. **Esplosione Combinatoria:** Numero esponenziale di itemset e regole con l'aumento della dimensione del dataset.
    b. **Calcolo Costoso:** Calcolo del supporto e della confidenza richiede tempo significativo, soprattutto con dataset grandi.

II. **Fattori che Influenzano la Complessità:**
    a. **Soglie di Supporto/Confidenza:** Soglie basse aumentano il numero di itemset frequenti, incrementando la complessità.
    b. **Dimensionalità:** Numero di elementi nel dataset influenza la dimensione del reticolo di itemset.
    c. **Dimensione del Database:** Calcolo del supporto richiede scansione di tutte le transazioni.
    d. **Larghezza Media delle Transazioni:** Transazioni più lunghe aumentano la complessità.
    e. **Strutture di Memorizzazione:** Strutture dati efficienti migliorano le prestazioni.

III. **Calcolo del Supporto:**
    a. **Complessità:** *O*( *NMw* ) confronti, dove:
		N: numero di transazioni
		M: numero di itemset candidati ( ≈ $O(2^d)$ , *d* = dimensione massima degli itemset)
		w: larghezza massima della transazione

IV. **Ottimizzazione con Strutture ad Hash:**
    a. **Miglioramento del Calcolo del Supporto:** Accesso diretto tramite hash, evitando confronti inutili.
    b. **Esempio:** Utilizzo di alberi per organizzare gli itemset in base agli elementi (vedi immagini nel testo originale).

V. **Calcolo del Numero di Itemset Totali:**
    a. **Formula:**  $$ \binom{k}{n} = \frac{k!}{n!(k-n)!} $$  dove *k* è il numero di elementi e *n* è la dimensione degli itemset.

VI. **Hash Tree per la Generazione di Regole Associative:**
    a. **Struttura:** Albero che organizza gli itemset candidati in base ad una funzione di hash.
    b. **Esempio:**  (vedi immagini nel testo originale)  Funzione di hash *x*%3, nodi etichettati con valori di hash, itemset organizzati in base al valore di hash del primo elemento. Suddivisione dei nodi foglia se oltre un limite.
    c. **Matching tra Hash Tree e Transazione:** Confronto tra l'hash tree e un albero rappresentante la transazione per trovare itemset sottoinsiemi della transazione.
    d. **Vantaggi:** Riduzione del numero di confronti e miglioramento delle prestazioni.



**Rappresentazione Compatta degli Itemset Frequenti**

I. **Problema della Complessità:**
* Crescita esponenziale del numero di itemset con l'aumentare della dimensione del dataset.
* Inefficienza nel calcolo di tutti gli itemset.

II. **Itemset Massimali:**
* Definizione: Un itemset è massimale se nessun suo superset immediato è frequente (dato un supporto minimo).
* Vantaggi:
	* Riduzione della complessità computazionale.
	* Efficienza nel calcolo.

III. **Itemset Chiusi:**
* Definizione: Un itemset è chiuso se nessun suo superset immediato ha lo stesso supporto.

IV. **Principio Apriori:**
* Se un itemset è frequente, tutti i suoi sottoinsiemi lo sono anch'essi.
* Utilizzo: Eliminazione di itemset non massimali durante la scoperta di regole associative.

V. **Valutazione delle Regole Associative:**
* Misure oggettive:
	* **Supporto:** Proporzione di transazioni contenenti l'itemset.
	* **Confidenza:** Proporzione di transazioni contenenti l'itemset, dato l'antecedente.
* Misure soggettive (dipendenti dal contesto).

VI. **Problemi nella Valutazione:**
* **Supporto non omogeneo:** Difficoltà nella scelta della soglia di supporto (troppo basso → troppi itemset; troppo alto → perdita di regole importanti).
* **Pattern cross support:** Regole con item aventi supporto molto diverso, rendendole poco interessanti.


**Regole Associative: Analisi e Valutazione**

I. **Tabella di Contingenza**
* Definizione: Strumento per valutare regole associative del tipo X → Y.
* Valori:
	* $s_{XY}$: Support count di X e Y
	* $s_{\overline{X}Y}$: Support count di X e ¬Y
	* $s_{X\overline{Y}}$: Support count di ¬X e Y
	* $s_{\overline{X}\overline{Y}}$: Support count di ¬X e ¬Y
* Calcolo del supporto: $\text{supporto}_{xy}=\frac{S_{xy}}{N}$ (N = numero totale di oggetti)
* Criterio di valutazione:  Confidenza c(X → Y) > s_Y (supporto di Y)

II. **Indipendenza Statistica e Correlazione**
* Condizione per una regola interessante: P(Y|X) > P(Y)  ⇔ P(X, Y) > P(X) × P(Y)
* Correlazione:
	* Positiva: P(X, Y) > P(X) × P(Y)
	* Negativa: P(X, Y) < P(X) × P(Y)
	* Indipendenza: P(X, Y) = P(X) × P(Y)

III. **Misure di Correlazione**
* $\text{Lift}=\frac{P(Y|X)}{P(Y)}$ (per regole associative)
* $\text{Interest}=\frac{P(X,Y)}{P(X)\times P(Y)}$ (per itemset)
*  $\phi_\text{coefficient}=\frac{P(X,Y)-P(X)\times P(Y)}{\sqrt{ P(X) \times [1 - P(X)] \times P(Y) \times [1 - P(Y)] }}$ (per variabili binarie)

IV. **Pruning delle Regole**
* Utilizzo delle misure di correlazione per eliminare regole sotto una certa soglia.
* Esempio:  Il Lift non è sempre un indicatore affidabile della forza di una regola.  


**Proprietà delle Metriche di Valutazione delle Regole Associative**

* **1. Simmetria:**
    * **Simmetriche:** Risultato indipendente dall'ordine degli elementi (es: Jaccard, coseno, Lift).  Adatte per misurare la validità dell'itemset.
    * **Asimmetriche:** Risultato dipendente dall'ordine degli elementi (es: confidenza, Laplace). Adatte per misurare la validità della regola.

* **2. Variazione di Scala:**
    * **Insensibili:** Risultato invariato al variare della dimensione del dataset.
    * **Sensibili:** Risultato variabile al variare della dimensione del dataset.

* **3. Correlazione (Coefficiente ϕ):**
    * Misura la correlazione tra due variabili binarie.
    * Insensibile all'inversione dei valori delle variabili.

* **4. Addizione di Casi Nulli:**
    * **Sensibili:** Risultato variabile all'aggiunta di casi nulli (transazioni senza X né Y).
    * **Insensibili:** Risultato invariato all'aggiunta di casi nulli.


**Paradosso di Simpson**

* **Problema:** L'analisi di dati aggregati può portare a conclusioni errate se non si considerano variabili nascoste.

* **Esempio:** Relazione tra "Acquisto HDTV" e "Acquisto Macchina per Esercizi".
    * **Analisi Aggregata:** Confidenza più alta per chi compra HDTV, suggerendo una correlazione. 
    * **Analisi Stratificata (per Studenti e Lavoratori):** La relazione si inverte.  La confidenza è più alta per chi compra la macchina per esercizi *senza* comprare l'HDTV in entrambi i gruppi.  (Valori di confidenza specificati nel testo originale).

* **Spiegazione:** La variabile nascosta (stato di studente/lavoratore) influenza la relazione osservata. È fondamentale considerare le variabili nascoste e utilizzare l'analisi stratificata per evitare conclusioni errate.


**Effetti della Distribuzione del Supporto nell'Analisi delle Regole Associative**

I. **Problema della Distribuzione del Supporto Non Omogeneo**
* A. **Distorsione:** Pochi itemset ad alto supporto, molti a basso supporto.
* B. **Conseguenze di un supporto minimo elevato:** Perdita di itemset rari ma interessanti.
* C. **Conseguenze di un supporto minimo basso:** Calcolo computazionalmente costoso.
* D. **Esempio:**  Immagini illustrate nel testo originale (non riproducibili qui).

II. **Sfide nei Dataset con Supporto Non Omogeneo**
* A. **Problema:** Supporti molto diversi tra itemset portano a conclusioni errate sull'interesse delle regole.
* B. **Esempio:** Regole con alta confidenza ma basso supporto (poco significative) vs. regole con bassa confidenza ma alto supporto (interessanti ma non rilevate).  
* C. **Cross-Support:** Itemset con alto supporto individuale ma basso supporto congiunto.  Le regole associate sono poco interessanti nonostante l'alta confidenza.

III. **Cross-Support: Definizione e Soluzione**
* A. **Problema:** Regole con alta confidenza ma basso supporto (es. "caviale → latte"). 
* B. **Soluzione:** Utilizzo di una misura di cross-support per identificare e rimuovere itemset con supporti disomogenei.
* C. **Misura di Cross-Support:**  $r = \frac{min\{s(x_1), ..., s(x_n)\}}{max\{s(x_1), ..., s(x_n)\}}$ dove  `s(xi)` è il supporto dell'i-esimo elemento.  Se  `r < γ` (soglia), l'itemset viene eliminato.

IV. **H-confidence: Una Misura Migliorativa**
* A. **Problema:** La confidenza può essere alta anche con supporti molto diversi tra itemset, portando a regole poco significative.
* B. **Soluzione:** L'H-confidence considera la confidenza minima di tutte le possibili regole da un itemset.
* C. **Definizione:** $H = min\{c(X_1 \to X_2) | X_1 \subset X \wedge X_2 = X - X_1\}$, dove $c(X_1 \to X_2) = \frac{s(X_1 \cup X_2)}{s(X_1)}$.
* D. **Calcolo semplificato:** $hconf = \frac{s(X)}{max\{s(x_1), ..., s(x_n)\}}$
* E. **Vantaggi:** Elevata H-confidence implica una forte relazione tra tutti gli elementi; elimina i cross-support pattern; individua pattern con basso supporto ma elevata H-confidence.


I. **Cross Support e H-confidence**

   A. **Supporto Itemset:** Diminuisce all'aumentare del numero di elementi. Limitato dal supporto degli elementi individuali.
   B. **H-confidence (hconf):** Misura la relazione tra il supporto di un itemset e il supporto dei suoi elementi individuali.
      1. Formula:  $$hconf = \frac{s(X)}{max\{s(x_1), ..., s(x_n)\}}$$  dove  $s(X)$ è il supporto dell'itemset X e $s(x_i)$ è il supporto dell'elemento $x_i$.
      2. Limite superiore: $$hconf \le \frac{min\{s(x_1), ..., s(x_n)\}}{max\{s(x_1), ..., s(x_n)\}} = r$$
   C. **Hyperclique:** Itemset con bassa h-confidence (supporto significativamente inferiore al supporto degli elementi individuali).  Interessanti per pattern a basso supporto.
      1. **Hyperclique Chiusi:** Nessun superset immediato ha la stessa h-confidence.
      2. **Hyperclique Massimali:** Nessun superset immediato è un hyperclique.
   D. **Applicazione:** Utilizzo in sostituzione o congiunzione con il supporto per trovare itemset con supporti comparati.


II. **Gestione Attributi Categorici e Continui**

   A. **Estensione delle Regole Associative:**  Applicazione a attributi categorici e continui (es: $\{Gender = Male, Age ∈ [21, 30]\} → \{\text{No of hours Online\} ≥ 10}$).

   B. **Attributi Categorici:**
      1. **Trasformazione in Binari:** Creazione di attributi binari asimmetrici per ogni valore del dominio di un attributo categorico.
      2. **Problematiche:** Aumento del numero di elementi e potenziale generazione di molte regole, alcuni valori con basso supporto, pattern ridondanti a causa di distribuzioni non omogenee.
      3. **Soluzioni:**
         a. Aggregare valori a basso supporto.
         b. Eliminare attributi binari con elevato supporto che non forniscono informazioni utili.
         c. Utilizzo di hconf per gestire attributi con supporto disomogeneo.

   C. **Riepilogo Trasformazione in Binari:**
      1. Aumento del numero di elementi.
      2. Larghezza delle transazioni invariata.
      3. Produzione di più itemset frequenti (dimensione massima limitata al numero di attributi originali).
      4. Approccio per ridurre l'overhead: evitare la presenza di più attributi binari derivanti dallo stesso attributo categorico all'interno di una regola.



**Gestione di Attributi Continui in Regole Associative**

I. **Regole Associative Quantitative:**
* Definizione: Regole che includono attributi con valori continui.
* Caratteristica: Descrivono distribuzioni di valori, non categorie discrete.

II. **Discretizzazione:**
* Approccio: Trasforma attributi continui in attributi discreti (binari) tramite la suddivisione del dominio in intervalli.
* Tecniche di Discretizzazione:
	* Equal-width: Intervalli di uguale ampiezza.
	* Equal-depth: Intervalli con uguale numero di elementi.
	* Clustering:
* Problematiche:
	* Intervalli ampi: Riduzione della confidenza dei pattern, fusione di pattern diversi, perdita di pattern interessanti.
	* Intervalli stretti: Riduzione del supporto dei pattern, divisione di un pattern in più pattern, itemset con supporto inferiore alla soglia minima.
* Soluzione computazionalmente costosa: Provare tutte le possibili combinazioni di intervalli, partendo da una larghezza iniziale *k* e fondendo intervalli adiacenti all'aumentare di *k*.  Genera potenzialmente pattern ridondanti.

III. **Regole Ridondanti:**
* Problema: Un intervallo frequente implica la frequenza di tutti gli intervalli che lo contengono.  
* Soluzione per intervalli ampi: Utilizzo di un supporto massimo come soglia.
* Gestione di regole ridondanti con stessa confidenza: Eliminazione della regola più specifica (intervallo più piccolo). 

**Approccio Statistico all'Estrazione di Regole di Associazione con Attributo Target Continuo**

I. **Fasi dell'Algoritmo:**

   A. **Pre-processing:**
      1. Rimozione dell'attributo target dal dataset.
      2. Estrazione degli itemset frequenti dagli attributi rimanenti.
      3. Binarizzazione degli attributi continui (escluso il target).

   B. **Generazione delle Regole:**
      1. Creazione di regole con l'itemset frequente come antecedente (A) e l'attributo target come conseguente (B).
      2. Calcolo delle statistiche descrittive dell'attributo target per ogni itemset frequente (es. media μ).

   C. **Validazione delle Regole:**
      1. Confronto delle statistiche dell'attributo target per le transazioni che soddisfano la regola (μ) e quelle che non la soddisfano (μ').
      2. Applicazione del Z-test per valutare la significatività della differenza:
         $$Z = \frac{|μ' - μ| - ∆}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
      3. Accettazione della regola se Z > $Z_α$ (soglia di significatività basata sulla confidenza desiderata).


II. **Considerazioni:**

   A. **Inutilizzabilità della confidenza:**  La confidenza non è applicabile in quanto il conseguente è una variabile continua.
   B. **Significatività della regola:** Una regola è significativa solo se la differenza tra le statistiche dell'attributo target per le transazioni coperte e non coperte dalla regola è statisticamente significativa (supera la soglia definita dal Z-test).



**Regole Associative Senza Discretizzazione (minApriori)**

* **Applicazione a diversi tipi di dati:**
    * Dati a grafo
    * Dati sequenziali
    * Documenti (focus del testo)

* **Generazione di regole su documenti:**
    * Scopo: Identificare relazioni tra parole/gruppi di parole e la loro presenza/frequenza nei documenti.
    * Rappresentazione dati: Matrice documento-termine (occorrenze come valori numerici).
    * Evitare la binarizzazione: Perdita di informazione sulle occorrenze.
    * Normalizzazione:
        * Supporto per parola = 1
        * Somma dei pesi delle parole per documento = 1

* **Calcolo del supporto:**
    * Formula: $sup(C) = \sum_{i \in T} \min_{j \in C} D(i, j)$
        * C: insieme di parole
        * T: insieme di documenti (transazioni)
        * D(i, j): occorrenza della parola j nel documento i
    * Proprietà: Antimonotono (aggiungendo elementi, il supporto tende a diminuire).

* **Calcolo delle regole:** Basato sul supporto calcolato.


**Regole Associative Multi-Livello**

* **Gerarchie semantiche:**
    * Termini organizzati in gerarchie basate sulla semantica (es. cibo -> latte -> latte scremato).
    * Regole generate a diversi livelli di astrazione.

* **Vantaggi delle gerarchie:**
    * Regole a livelli bassi potrebbero non avere supporto sufficiente.
    * Regole a livelli troppo bassi potrebbero essere troppo specifiche.
    * Le gerarchie permettono di individuare associazioni più generali.  (es. latte e pane, indipendentemente dal tipo di latte o pane).

* **Supporto e confidenza:**
    * Variano a seconda del livello nella gerarchia.
    * Relazioni tra supporto e confidenza a diversi livelli:
        * `If X è genitore di X1 e X2, then σ(X) ≤ σ(X1) + σ(X2)`
        * `If σ(X1 ∪ Y1) ≥ minsup, and X è genitore di X1, Y è genitore di Y1 then σ(X ∪ Y1) ≥ minsup, σ(X1 ∪ Y) ≥ minsup σ(X ∪ Y) ≥ minsup`
        * `If conf(X1 ⇒ Y1) ≥ minconf, then conf(X1 ⇒ Y) ≥ minconf`


**Conseguenze dell'utilizzo di gerarchie di item nell'estrazione di pattern frequenti:**

* **Elevato supporto per item ad alti livelli:**
    * Supporto molto elevato per item nella parte superiore della gerarchia.
    * Possibilità di pattern cross-support (con bassa soglia di supporto).
    * Aumento del numero di associazioni ridondanti (es: Latte -> Pane; Latte scremato -> Pane).

* **Aumento della complessità computazionale:**
    * Aumento della dimensionalità dei dati.
    * Aumento del tempo di elaborazione.

* **Soluzione alternativa: Estrazione per livello:**
    * Generazione di pattern frequenti separatamente per ogni livello della gerarchia.
    * Inizio dall'estrazione al livello più alto.
    * Iterazione verso i livelli inferiori.

* **Conseguenze della soluzione alternativa:**
    * Aumento del costo di I/O (più scansioni dei dati).
    * Perdita di potenziali associazioni cross-livello.


