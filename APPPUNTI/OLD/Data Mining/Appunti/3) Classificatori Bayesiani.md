
In molte applicazioni, la relazione tra gli attributi di input e la classe target non è deterministica.
Questo introduce **incertezza** nelle previsioni del modello.
La teoria della probabilità offre un modo sistematico per quantificare e manipolare l'incertezza nei dati.

I modelli di classificazione che utilizzano la teoria della probabilità per rappresentare la relazione tra attributi e classe sono chiamati **modelli di classificazione probabilistici**.

### Teoria della Probabilità

* **Variabile casuale:** Una variabile a cui sono associate le probabilità di ciascun possibile risultato (valore assunto).
* **Probabilità di un evento:** Misura quanto è probabile che si verifichi l'evento.
* **Visione frequentista:** La probabilità si basa sulla frequenza relativa degli eventi.
* **Visione bayesiana:** La probabilità ha una visione più flessibile.
* **Probabilità congiunta:** La probabilità di osservare due eventi contemporaneamente.
 * $P(X=x_i, Y=y_j) = \frac{n_{ij}}{N}$ , dove $n_{ij}$ è la frequenza assoluta dell'evento X = xi & Y = yj su un numero totale N di osservazioni.
* **Marginalizzazione:** Somma delle probabilità congiunte rispetto a una delle variabili casuali.
 * $P(X=x_i) = \frac{n_{i}}{N}=\frac{\sum_{j=1}^k n_{ij}}{N}=\sum_{j=1}^K P(X=x_i, Y=y_j)$ 
 * $P(X=x_i)$ è la **probabilità marginale di X**.

### Probabilità Condizionata

* **Probabilità condizionata:** La probabilità di osservare la variabile casuale Y ogni volta che la variabile casuale X assume un particolare valore.
 * $P(Y|X) = \frac{P(X,Y)}{P(X)}$ 
 * $P(X|Y) = \frac{P(X,Y)}{P(Y)}$ 
* **Relazione tra probabilità congiunta e condizionata:**
 * $P(X, Y)= P(Y|X) \times P(X) = P(X|Y) \times P(Y)$ 

### Teorema di Bayes

* **Formula:** $P(Y|X) = \frac{P(Y,X)}{P(X)}=\frac{P(X|Y)P(Y)}{P(X)}$ 
* **Significato:** Fornisce una relazione tra le probabilità condizionali $P (Y I X) \text{ e } P(XIY$).

**Nota**
* La probabilità assume sempre valori compresi tra 0 e 1.
* La probabilità congiunta è simmetrica.

### Obiettivo della Classificazione

* Dato un record con attributi $X_1, X_2, ..., X_d$, l'obiettivo è prevedere la classe Y.
* Si vuole trovare il valore di Y che massimizza la **probabilità a posteriori**: $P(Y|X_1, X_2, ..., X_d)$.

## Teorema di Bayes

* Il teorema di Bayes permette di calcolare la probabilità a posteriori:

$$P(Y|X_1, X_2, ..., X_d) = \frac{P(X_1, X_2, ..., X_d|Y)P(Y)}{P(X_1, X_2, ..., X_d)}$$

* **Scegliere Y che massimizza $P(Y|X_1, X_2, ..., X_d)$ è equivalente a scegliere il valore di Y che massimizza $P(X_1, X_2, ..., X_d|Y)$**.
 * $P(X_1, X_2, ..., X_d|Y)$ è la **probabilità condizionata di classe** degli attributi data l'etichetta della classe.
 * Misura la probabilità di osservare gli attributi $X_1, X_2, ..., X_d$ dalla distribuzione delle osservazioni di Y.
 * Se il record con attributi $X_1, X_2, ..., X_d$ appartiene davvero alla classe Y, allora ci si aspetta che $P(X_1, X_2, ..., X_d|Y)$ sia alto.

### Termini del Teorema di Bayes

* **$P(Y)$: Probabilità a priori.**
 * Rappresenta il punto di vista Bayesiano.
 * Cattura le convinzioni preliminari sulla distribuzione delle etichette di classe, indipendentemente dai valori degli attributi osservati.
 * Ad esempio, potremmo avere una convinzione preliminare che la probabilità che una persona soffra di una malattia cardiaca sia pari a α, indipendentemente dai rapporti diagnostici.
* **$P(X_1, ..., X_d)$: Probabilità of evidence.**
 * Questo termine non dipende dall'etichetta della classe e quindi può essere trattato come una costante di normalizzazione nel calcolo delle probabilità a posteriori $P(Y|X_1, X_2, ..., X_d)$.

### Stima delle Probabilità

* La probabilità a priori $P(Y)$ può essere facilmente stimata dal training set calcolando la frazione di record appartenenti a ciascuna classe.
* Per calcolare $P(X_1, X_2, ..., X_d|Y)$ (probabilità condizionata di classe), un approccio è considerare la frazione di record del training set di una data classe per ogni combinazione di valori di attributo.
 * Questo approccio diventa computazionalmente proibitivo all'aumentare del numero di attributi, a causa dell'incremento esponenziale delle possibili combinazioni di valori.
 * Inoltre, con un training set di piccole dimensioni, si rischia di avere stime imprecise delle probabilità condizionate di classe.

## Assunzione di Naïve Bayes

* **Assunzione di indipendenza condizionale:** Questa assunzione afferma che gli attributi  sono **condizionatamente indipendenti** tra loro, dato il valore della classe .
* In questo modo, la probabilità condizionata di classe di tutti gli attributi può essere scomposta come un prodotto delle probabilità condizionate di classe di ogni singolo attributo $X_i$:

$$P(X_1, ..., X_d|Y ) = \prod_{i=1}^d P(X_i|Y )$$

### Classificatore Naïve Bayes

* Si basa sull'assunzione di indipendenza condizionale tra gli attributi.
* Calcola $P(X_i|Y_j)$ per tutti gli $X_i$ e $Y_j$ del training set.
* Classifica un nuovo oggetto come $Y_j$ se $P(Y_j) \prod_{i=1}^d P(X_i|Y_j)$ è massimo.

### Formule per il Calcolo delle Probabilità

* $P(Y_j) = \frac{n_j}{N}$
* $P(X_i|Y_j) = \frac{|X_{ij}|}{n_j}$

Dove:

* $N$ è il numero totale di record.
* $n_j$ è il numero di record appartenenti alla classe $Y_j$.
* $|X_{ij}|$ è il numero di record con valore $X_{ij}$ per l'attributo $X_i$ che appartengono alla classe $Y_j$.

**Nota:** Queste formule valgono se gli attributi sono categorici. Per attributi continui, si utilizzano metodi di stima della densità.

### Indipendenza Condizionale

* Date le variabili aleatorie X, Y, Z, si dice che X, Y sono **condizionatamente indipendenti** rispetto a Z se:

$$P(X|Y, Z) = P(X|Z)$$

* Da questa definizione deriva che:
$$P(X, Y |Z) = \frac{P(X, Y, Z)}{P(Z)}$$
$$= \frac{P(X, Y, Z)}{P(Y, Z)} \times \frac{P(Y, Z)}{P(Z)}$$
$$= P(X|Y, Z)P(Y, Z) = P(X|Z)P(Y |Z)$$

* Di conseguenza, come visto in precedenza:

$$P(X_1, ..., X_d|Y_j) = P(X_1|Y_j)P(X_2|Y_j)...P(X_d|Y_j)$$

### Applicazione al Naive Bayes

* Sotto l'ipotesi di indipendenza condizionale delle variabili X e Y rispetto a Z, la probabilità congiunta condizionata $P(X,Y|Z)$ può essere scomposta nel prodotto delle probabilità condizionate individuali $P(X|Y,Z)$ e $P(Y|Z).$
* Questo risultato è applicato al caso specifico del "*naive Bayes*", dove le variabili $X_{1},X_{2},\dots,X_{d}$ sono gli attributi e Y è la variabile di classe.
* Sotto l'assunzione di indipendenza condizionale degli attributi data la classe, la probabilità congiunta condizionata $P(X_{1},X_{2},\dots,X_{d}|Y)$ può essere scomposta nel prodotto delle probabilità condizionate individuali P(X_i|Y) per ogni attributo x_i.

### Stima della Probabilità per Attributi Continui

Quando si hanno attributi continui nell'algoritmo Naive Bayes, non è possibile stimare direttamente la probabilità per ogni singolo valore dell'attributo.

* Vengono proposte due strategie alternative:

1. **Discretizzazione:** Discretizzare gli attributi continui in intervalli, trasformandoli in attributi ordinali. Questo comporta però il rischio di errori di stima a seconda del numero di intervalli scelti.
2. **Distribuzione di probabilità:** Assumere che i valori degli attributi continui seguano una specifica distribuzione di probabilità, come la distribuzione Normale (Gaussiana).

Nel caso della distribuzione Normale, la probabilità condizionata dell'attributo $x_{i}$ dato la classe $Y=y_{j}$ (ovvero $P(X_{i}=x_{i}|Y=y_{j})$) può essere calcolata usando la funzione di densità della Normale, con i parametri media $\mu_{ij}$ e varianza $\sigma_{ij}^2$ stimati dai dati del training set appartenenti alla classe yj.

$$P(X_{i}=x_{i}|Y=y_{j})=\frac{1}{\sqrt{ 2 \pi\sigma_{ij} }}e^-\frac{(x_{i}-\mu_{ij})^2}{2\sigma^2_{ij}}$$

### Problemi con Probabilità Zero

Nell'algoritmo Naive Bayes, può capitare che la probabilità condizionata P(Xi|Y) per uno degli attributi Xi risulti essere zero.
* Ciò accade quando nel training set non si osservano alcune combinazioni di valori dell'attributo Xi e della classe Y.
* Questa situazione è più probabile quando il numero di record nel training set è piccolo e il numero di possibili valori dell'attributo è molto elevato.
* Avere una probabilità condizionata $P(X_{i}|Y)=0$ fa sì che l'intera probabilità condizionata della classe $P(X_{1},\dots X_{d}|Y)$ diventi zero, a causa del prodotto nelle formule di Naive Bayes.

### Soluzioni per Probabilità Zero

* Per ovviare a questo problema, invece di stimare le probabilità condizionate come semplici frazioni dai dati, si possono usare stime alternative come:

* **Originale:** $P(A_i|C) = \frac{N_{ic}}{N_c}$

* **Laplace:** $P(A_i|C) = \frac{N_{ic} + 1}{N_c + c}$

* **m-estimate:** $P(A_i|C) = \frac{N_{ic} + mp}{N_c + m}$

*Dove:*
* c = numero di classi
* p= probabilità a priori della classe C
* m = iperparametro che pesa p quando i dati sono scarsi
* $N_c$ = numero istanze della classe C
* $N_{ic}$ = numero istanze con valore $A_{i}$ nella classe C

## Reti Bayesiane

Le reti bayesiane sono in grado di catturare forme più generiche di indipendenza condizionale, usando rappresentazioni grafiche delle relazioni probabilistiche tra un insieme di variabili casuali.

* Si utilizza un grafo diretto aciclico (DAG), in cui:
 * **Nodi:** corrispondono alle variabili casuali.
 * **Archi:** corrispondono alle relazioni di dipendenza tra una coppia di variabili.

### Indipendenza Condizionale: La Condizione di Markov

* Una proprietà importante delle reti bayesiane è la loro capacità di rappresentare diverse forme di indipendenza condizionale tra variabili casuali.
* Viene utilizzata la **condizione di Markov:** un nodo è condizionatamente indipendente dai suoi non-discendenti, se i suoi genitori sono noti.
* La condizione di Markov locale aiuta a interpretare le relazioni genitore-figlio come rappresentazioni di probabilità condizionate.
* Poiché un nodo è condizionatamente indipendente dai non-discendenti dati i genitori, le ipotesi di indipendenza sono spesso di struttura sparsa.
* Le reti bayesiane possono esprimere una classe più ricca di indipendenza condizionale tra attributi ed etichette rispetto al Naïve Bayes e forniscono una struttura più generica, poiché la classe target può apparire ovunque nel grafo, non solo alla radice.

### Distribuzione della Probabilità Congiunta

* Grazie alla condizione di Markov è possibile calcolare la distribuzione di probabilità congiunta dell'insieme di variabili casuali coinvolte in una rete Bayesiana.
* Supponiamo una rete Bayesiana contenente $n$ nodi $\{X_1, ..., X_n\}$, dove ogni nodo è numerato in modo tale che $X_i$ è un antenato di $X_j$ solo se $i < j$. La probabilità congiunta $P(X_1, ..., X_n)$ può essere così espressa:

$$P(X_1, ..., X_n) = P(X_1)P(X_2|X_1)P(X_3|X_1, X_2)...P(X_n|X_1, ..., X_{n-1})$$

$$= \prod_{i=1}^{n} P(X_i|X_1, ..., X_{i-1}) = \prod_{i=1}^{n} P(X_i|Parents(X_i))$$

**Spiegazione:**

* La probabilità congiunta di tutte le variabili può essere scomposta in un prodotto di probabilità condizionali.
* Ogni probabilità condizionale $P(X_i|X_1, ..., X_{i-1})$ rappresenta la probabilità di $X_i$ dato il valore dei suoi genitori (antenati) nella rete.
* La seconda formulazione evidenzia che la probabilità condizionale di un nodo dipende solo dai suoi genitori diretti, non da tutti i suoi antenati.

* Questa scomposizione della probabilità congiunta è fondamentale per la semplificazione del calcolo delle probabilità in una rete Bayesiana. 

### Tabella di Probabilità Condizionata (CPT)

* Per rappresentare la probabilità congiunta P(X) in una rete bayesiana, è sufficiente rappresentare la probabilità di ogni nodo Xi in termini dei suoi genitori parents(Xi).
* Questo si ottiene tramite tabelle di probabilità condizionata (CPT):
 * Se Xi non ha genitori, la tabella contiene P(Xi);
 * Altrimenti contiene $P(X_{i}|\text{Parents}(X_{i}))$ per ogni combinazione di valori di $X_{i} \text{ e Parents}(X_{i})$.
* Per una variabile booleana con k genitori, la tabella ha 2(k+1) voci.
* Le CPT quantificano l'influenza dei genitori sul nodo corrente.

### Inference and Learning

La probabilità che una classe Y assuma un valore specifico y, dato dall'insieme di attributi osservati su un dato record x è dato dalla seguente:

La probabilità condizionale di $Y = y$ dato $x$ può essere espressa come:

$$P(Y = y|x) = \frac{P(y, x)}{P(x)}= \frac{P(y, x)}{\sum}$$

L'equazione precedente coinvolge probabilità marginali della forma $P(y, x)$. Queste possono essere calcolate effettuando la marginalizzazione delle variabili nascoste $H$ dalla probabilità congiunta come segue:

$$P(x, y) = \sum_{H} P(y, x, H)$$

Dove la quantità $P(y, x, H)$ può essere ottenuta utilizzando la fattorizzazione già descritta precedentemente.

**Spiegazione:**

* La probabilità condizionale $P(Y = y|x)$ è il rapporto tra la probabilità congiunta di $y$ e $x$ e la probabilità marginale di $x$.
* La probabilità marginale $P(x, y)$ può essere calcolata sommando la probabilità congiunta $P(y, x, H)$ su tutti i possibili valori della variabile nascosta $H$.
* La fattorizzazione precedentemente descritta si riferisce alla scomposizione della probabilità congiunta in un prodotto di probabilità condizionali, come visto nella rete Bayesiana.

Questa formulazione permette di calcolare la probabilità condizionale anche in presenza di variabili nascoste, sfruttando la fattorizzazione della probabilità congiunta. 
