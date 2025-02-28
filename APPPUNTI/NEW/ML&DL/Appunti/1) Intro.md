L'apprendimento automatico si basa sul principio di trasformare l'esperienza in competenza. Questo processo avviene attraverso l'analisi di dati e la creazione di modelli predittivi.

### Training Set

* **Definizione:** Un insieme di dati di esempio, rappresentato come $S = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$.
* **Elementi:**
 * $x_i$: Vettore numerico che rappresenta un esempio (input).
 * $y_i$: Etichetta associata all'esempio (output).

##### Esempio: Classificazione del Sapore della Frutta

* **Input ($x_i$):**
 * $x_{i1}$: Consistenza (0 = molle, 1 = duro)
 * $x_{i2}$: Colore (0 = nero, 1 = bianco)
* **Output ($y_i$):**
 * 0: Sapore cattivo
 * 1: Sapore buono

### Learner

* **Funzione:** Riceve in input il training set e restituisce un altro programma, chiamato **Predittore (h)**.
* **Fase di Training:** Il processo di apprendimento del Learner, che determina il Predittore.
* **Rappresentazione:** x -> Predittore -> y = h(x)

### Predittore

* **Obiettivo:** Classificare nuovi esempi (non presenti nel training set) in modo accurato.
* **Capacità di Generalizzazione:** La capacità del Predittore di estendere le conoscenze acquisite dal training set a nuovi dati. 

##### In sintesi:

L'apprendimento automatico si basa sull'addestramento di un modello (Learner) su un insieme di dati di esempio (Training Set) per creare un Predittore che può classificare nuovi esempi in modo accurato. La capacità di generalizzazione del Predittore è fondamentale per la sua efficacia. 

### Approcci all'Intelligenza Artificiale

* **Approcci Simbolici:**
 * I simboli manipolati hanno una corrispondenza 1-1 con oggetti del mondo reale.
 * Esempio: Logica (Prolog)
* **Approcci Sub-Simbolici:**
 * Non c'è una corrispondenza 1-1 tra simboli e oggetti del mondo reale.
 * Vengono manipolati vettori di numeri.
 * Esempio: Machine Learning

### Domande Importanti

* **Com'è fatto l'input?** La natura e la struttura dei dati di input sono fondamentali per l'apprendimento automatico.
* **Come possiamo automatizzare il processo di apprendimento?** L'obiettivo è sviluppare algoritmi che possano imparare da soli, senza intervento umano.
* **Come possiamo essere sicuri che il predittore (output) sia di buona qualità?** È necessario valutare la performance del modello e garantire che sia in grado di generalizzare a nuovi dati. 

## Framework dell'Apprendimento Statistico

### 1) Input del Learner

* **Dominio d'interesse (X):** Un sottoinsieme di $\mathbb{R}^d$ che rappresenta l'insieme di istanze da predire.
 * **d:** Numero di features (caratteristiche) di ogni istanza.
* **Insieme di etichette (Y):** In questo caso, un insieme binario {0, 1} per la classificazione binaria.
* **Training set (S):** Un insieme di esempi etichettati, rappresentato come $S = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$.

### 2) Output del Learner

* **Funzione h:** Il predittore o l'ipotesi, che mappa l'insieme di input X nell'insieme di output Y: $h: X \rightarrow Y$.

### 3) Modello di Generalizzazione dei Dati

* **Distribuzione di probabilità (D):** Descrive la probabilità di osservare un'istanza x: $D(x)$.
* **Funzione di etichettatura (f(x)):** Restituisce l'etichetta vera dell'istanza x.
* **Importanza:** La distribuzione di probabilità D è generalmente sconosciuta. Se fosse nota, potremmo costruire il predittore ottimale.

### 4) Misure di Successo

* **Valutazione della bontà del predittore:**
 * **Errore di generalizzazione (Loss):** $L_D(h) = P_{x \sim D}[h(x) \neq f(x)]$, ovvero la probabilità che il predittore h sbagli a classificare un'istanza x.
 * **Errore empirico (Rischio empirico):** $L_S(h) = \frac{|x \in S : h(x) \neq f(x)|}{|S|}$, ovvero la proporzione di errori del predittore h sul training set S.
 * **|S|:** Dimensione del training set.

**Nota:** Il framework dell'apprendimento statistico si basa sull'idea di addestrare un modello (Learner) su un insieme di dati di esempio (Training Set) per creare un Predittore che possa generalizzare a nuovi dati. Le misure di successo (errore di generalizzazione e errore empirico) sono utilizzate per valutare la performance del Predittore. 

## Minimizzazione del Rischio Empirico (ERM)

* **Algoritmo di apprendimento:** Cerca di trovare il predittore $h_S$ che minimizza l'errore empirico $L_S(h)$:
 * $h_S = arg \ min \ L_S(h)$

### Caso di Decision Boundary Uniforme:

* Se il decision boundary D è uniforme, il predittore ideale $h^*$ è definito come:
 $$h^*(x) = \begin{cases}
 y_i & \text{se } \exists x_i \in S: x_i = x \\
 0 & \text{altrimenti}
 \end{cases}$$

* In questo caso, l'errore empirico $L_S(h^*)$ è:
 * $L_S(h^*) = 0$ (se esiste un esempio nel training set con lo stesso input di x)
 * $L_S(h^*) = 1/2$ (se il numero di esempi positivi è uguale al numero di esempi negativi)
 * Un classificatore con un errore empirico di 1/2 è considerato casuale ed è a rischio di *overfitting* (il modello non riesce a generalizzare a nuovi dati).

## Errore di Ipotesi (H)

* **Vincolo induttivo:** Rappresenta la conoscenza a priori (prior knowledge) che gli esperti di dominio forniscono.
* **Definizione:** L'insieme H rappresenta l'insieme di tutte le possibili ipotesi (modelli) che il Learner può considerare.
* **ERM con vincolo di ipotesi:** L'algoritmo ERM con vincolo di ipotesi cerca di trovare il predittore $h_S$ che minimizza l'errore empirico $L_S(h)$ tra tutte le ipotesi in H:
 * $ERM_H(S) = h_S = arg \ min \ L_S(h), h \in H$
* **Esempio:** H potrebbe essere l'insieme di tutti i rettangoli contenuti in $[0, 1]^2$.

**Nota:** L'errore di ipotesi H limita lo spazio di ricerca del Learner, influenzando la capacità del modello di generalizzare a nuovi dati. 

## Teorema 

Se l'insieme di ipotesi H è finito, allora il predittore $h_S$ ottenuto tramite ERM (Minimizzazione del Rischio Empirico) non va in overfitting, oppure è probabilmente approssimativamente corretto.

## Assunzioni sul Training Set

* **Indipendenza e Identica Distribuzione (IID):** Gli esempi nel training set sono indipendenti e identicamente distribuiti. Ciò significa che conoscere un elemento del set non fornisce informazioni aggiuntive sugli altri elementi.
* **Assunzione di Realizzabilità:** Esiste un'ipotesi $h^*$ nell'insieme H che ha un rischio empirico nullo, ovvero $L_D(h^*) = 0$. In altre parole, esiste un modello perfetto che può classificare correttamente tutti gli esempi nel training set.

## Overfitting

* **Impossibilità di un rischio nullo:** È impossibile ottenere un rischio di generalizzazione $L_D(h) = 0$ in pratica.
* **Soglia di accuratezza (ε):** Definiamo una soglia di accuratezza ε per considerare un predittore "approssimativamente corretto":
 * $L_D(h) \leq \epsilon$
* **Probabilità di accuratezza:** Un predittore $h_S$ è considerato **probabilmente approssimativamente corretto (PAC)** se la probabilità che il suo rischio di generalizzazione sia inferiore a ε è maggiore di 1 - δ, dove δ è una soglia di confidenza:
 * $P_R[L_D(h_S) \leq \epsilon] \geq 1 - \delta$

##### In sintesi:

Il seguente teorema fornisce una garanzia per la generalizzazione dei modelli quando l'insieme di ipotesi è finito. Le assunzioni di IID e di realizzabilità sono importanti per garantire che il training set sia rappresentativo della distribuzione dei dati reali. Il concetto di overfitting si riferisce alla situazione in cui un modello si adatta troppo bene al training set e non riesce a generalizzare a nuovi dati. La probabilità di accuratezza (PAC) fornisce una misura della probabilità che un modello sia approssimativamente corretto. 

## Come deve essere fatta $h_S$?

##### Condizioni:

* **H finita:** L'insieme di ipotesi H è finito.
* **Realizzabilità:** Vale l'assunzione di realizzabilità, ovvero esiste un'ipotesi $h^*$ in H che ha un rischio empirico nullo ($L_D(h^*) = 0$).
* **S sufficientemente grande:** Il training set S contiene un numero sufficiente di esempi (m, legato ad altri parametri).

##### Conclusione:

Se le condizioni sopra elencate sono soddisfatte, allora $h_S$ è **probabilmente approssimativamente corretto (PAC)**.

##### Formulazione matematica:

* $P_r[{S:L_d(h_S) \leq \epsilon}] \geq 1- \delta$
* Complementare: $P_r[{S:L_d(h_S) \ge \epsilon}] \leq \delta$

##### Ipotesi cattiva (bad):

* $H_b$: Insieme di ipotesi in H che hanno un rischio di generalizzazione maggiore di
 $\epsilon$ ($h\in H : L_d(h_S) \ge \epsilon$).

##### Esempi forvianti (misleading):

* $M$: Insieme di training set S per i quali esiste un'ipotesi in $H_b$ che ha un rischio empirico nullo ($M= {S: ∃ h \in H_b, L_S(h)=0}$).

##### Relazione tra insiemi:

L'insieme di training set per i quali $L_d(h_S) \geq \epsilon$ è strettamente contenuto in M.

##### Conclusione finale:

La probabilità che $L_d(h_S) \geq \epsilon$ è minore o uguale alla probabilità di M:

* $pr[L_d(h_S) \geq \epsilon] \leq P_r(M)$

##### Dimostrazione:

$$M = \bigcup_{h \in H_B} \{S: L_S(h) = 0\}$$

$$Pr[\bigcup_{h \in H_B} \{S: L_S(h) = 0\}]=Pr[\bigcup_{h \in H_B} \{S: \forall _i h(x_i)=f(x_i)\}]$$

* Considerando che:
 * la probabilità dell'unione di due eventi è minore o uguale alla somma delle probabilità dei singoli eventi: $p(A \cup B) \leq P(A) + P(B)$ 
* Otteniamo:
$$\leq \sum_{h \in H_b} Pr[\{S: \forall_i h(x_i)=f(x_i)\}]$$

* Dove $H_b$ è l'ipotesi cattiva.
* Inoltre, se A e B sono indipendenti, allora $p(A \cap B) \leq P(A) P(B)$.
* Quindi:
$$= \sum_{h \in H_b} \prod_{i=1}^m [x_i: h(x_i) = f(x_i)]$$

* Visto che:
 * $Pr [\{x: h(x) = f(x) \}]$ è la probabilità di osservare un'istanza del dominio per cui il predittore non sbaglia.
 * $Pr[x:h(x)\neq f(x)]$ è la probabilità di errore.
 * $1-L_D(h)\leq 1- \epsilon$
$$\leq \sum_{h \in H_b} \prod_{i=1}^m (1-\epsilon) = \sum_{h \in H_b} (1-\epsilon)^m$$

* Utilizzando la disuguaglianza $1 - \epsilon \leq e^{-\epsilon}$ otteniamo:
$$\leq \sum_{h \in H_b} e^{-m \epsilon}$$

* Poiché $|H_b| \leq |H|$, abbiamo che $h \leq |H|e^{-m \epsilon} \leq \delta$. 
* Isolando m:
 * $m \leq \frac{1}{\epsilon} ln \frac{|H|}{\delta}$ 
 * Se m supera questa soglia, l'ipotesi è PAC.
 * Se H è infinita, la soglia è infinita.

## Sample Complexity

$$m_H(\epsilon, \delta)$$

* **Significato:** Indica la dimensione minima del training set necessaria per ottenere un'ipotesi **probabilmente approssimativamente corretta (PAC)**.
* **Formula:** $m_H(\epsilon, \delta) = \frac{1}{\epsilon}ln \frac{|H|}{\delta}$
* **PAC-Learnability:** Un concetto legato alla Sample Complexity. Una classe di ipotesi H è **PAC-Learnable** se esiste una funzione $m_H(\epsilon, \delta)$ tale che, con un training set di dimensione $m \geq m_H(\epsilon, \delta)$, l'algoritmo ERM produce un'ipotesi PAC.

##### Definizione:

Una classe di ipotesi H è **PAC-Learnable** se esiste una funzione $m_H(\epsilon, \delta)$ detta **Sample Complexity** tale che, se $m = |S| \geq m_H(\epsilon, \delta)$ sotto l'assunzione di realizzabilità, allora $h_S = ERM_H(S)$ è PAC, ovvero:
$$Pr[L_D(h_s) \leq \epsilon] \geq 1- \delta$$

### Teorema:

Se H è finita, allora è PAC-Learnable.

H={insieme dei rettangoli strettamente contenuti in $[0,1]^2 = (x_1,x_2,y_1,y_2) \in \mathbb{R}$ }

dove: $x_1,x_2,y_1,y_2$ rappresentano d
	- le x sono numeri float:
		- precisione singola: b= 32 bit
		- precisione doppia: b=64 bit
$$|H| \leq 2^b \times ... \times 2^b = 2^{bd}$$

### Discretizzazione

$$m \geq \frac{1}{\epsilon} ln \frac{|H|}{\delta} = \frac{1}{\epsilon} ln \frac{2^{bd}}{\delta}= \frac{bd \ ln2+ln(\frac{1}{\delta})}{\epsilon}$$

* **Significato:** La formula mostra come la Sample Complexity (m) dipende dal numero di parametri (d) e dalla precisione (b) utilizzata per rappresentare i numeri reali.
* **Dipendenza lineare:** La formula evidenzia che la Sample Complexity dipende linearmente dal numero di parametri (d). Questo significa che, all'aumentare del numero di parametri, aumenta anche la dimensione del training set necessaria per ottenere un'ipotesi PAC.

