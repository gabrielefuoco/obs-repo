
La teoria dei Giochi modella situazioni in cui agenti (umani o artificiali) interagiscono. Gli agenti possono *competere, collaborare o negoziare* per raggiungere i propri obiettivi.

### Tipi di giochi:

1. **Giochi strategici**:
- Situazioni di competizione tra giocatori con obiettivi da raggiungere.
- Esempio: competizione politica.
2. **Giochi estensivi**:
- Le azioni e le strategie considerano tutto ciò che accade nel tempo.
- Esempio: contrattazione, commercio.
3. **Giochi ripetitivi**:
- La stessa situazione si ripete più volte, influenzando le strategie.
- Esempio: dinamiche evolutive (predatore/preda).
4. **Giochi cooperativi**:
- I giocatori collaborano per ottenere vantaggi comuni.
- Esempio: teoria politica dei voti.

| Tipo di informazione | Descrizione |
| -------------------- | -------------------------------------------------------- |
| **Perfetta** | I giocatori conoscono tutte le azioni e decisioni prese. |
| **Imperfetta** | Non tutti i giocatori conoscono l'intera situazione. |
| **Completa** | I giocatori conoscono le preferenze degli avversari. |
| **Incompleta** | I giocatori non conoscono le preferenze degli avversari. |

Un ulteriore concetto è il **Mechanism design**: Progettare le regole del gioco per massimizzare i propri obiettivi. Esempio: aste.

## Giochi Coalizionali

Sono giochi cooperativi dove gli agenti (giocatori) cooperano per ottenere benefici comuni, detti **worth**, in base alla coalizione di cui fanno parte. Gli agenti possono essere persone, aziende, partiti politici, ecc.

**Obiettivo**: 

Calcolare una **solution concept**, ovvero come distribuire i benefici tra i giocatori in modo da soddisfare proprietà desiderabili (equità, razionalità).

**Struttura del Gioco**:

- **Grand Coalition**: insieme totale dei giocatori.
- **Worth Function (v(s))**: funzione che restituisce il beneficio di ogni sottoinsieme di giocatori.
- **Ricchezza totale**: il massimo beneficio ottenibile dalla grand coalition, denotato come **v(N)**.

Il gioco risponde a delle **Domande Fondamentali**:
- **Quale coalizione formare**?
- **Come distribuire la ricchezza tra i membri della coalizione**?

Si basa su due **Assunzioni**:
- **Transferable Utility (TU)**: la ricchezza può essere distribuita liberamente tra i membri (ad esempio, denaro).
- **Non-Transferable Utility (NTU)**: se la ricchezza non è distribuibile in modo frazionato (ad esempio, oggetti indivisibili), la worth function non sarà un numero, ma un insieme di oggetti.

**Formalmente**:

- Un gioco è una coppia **G = (N, v)** dove **v** è la worth function.
- **Outcome**: un assegnamento di valori ai giocatori, definito dall'imputation set **X(G)**, che contiene tutti i vettori possibili **x = (x₁, ..., xₙ)**.

Un'**imputazione** (soluzione), per essere considerata ammissibile, deve soddisfare le seguenti **proprietà:**
- **Efficiency**: la somma dei valori assegnati ai giocatori deve essere pari a **v(N)**.
- **Individual Rationality**: il valore di ogni giocatore deve essere almeno quanto otterrebbe da solo, ovvero $x_{i} ≥ v({i})$ per ogni giocatore $i$

**Solution Concepts**:

- **Fairness**: distribuzione equa della ricchezza.
- **Stability**: nessun giocatore ha incentivi a lasciare la coalizione.

##### Inoltre:

- Giocatori in una rete connessa formano una coalizione.
- La ricchezza della coalizione dipende dai costi degli archi nel grafo che li collega.
- **Eccesso**: misura l'insoddisfazione di una coalizione. Ad esempio, l'eccesso della coalizione $S$ è dato da $e(S, x) = v(S) -\sum_{i} x(S)$.

### Caratteristiche dei Giochi Coalizionali

1. **Giochi Superadditivi**:
- Se due coalizioni **S** e **T** (che non hanno membri in comune) si uniscono, il loro **worth** combinato è maggiore o uguale alla somma dei loro **worth** individuali.
- **Formula**: 
$$\forall S, T \subset N, \text{ se } S \cap T = \emptyset$, allora $v(S \cup T) \geq v(S) + v(T)$$
Significa che **conviene collaborare**, poiché la somma delle coalizioni è sempre maggiore o uguale alla somma delle singole worth. La **grand coalition** ha sempre il payoff massimo.

2. **Giochi Additivi**:
- Non c'è alcun vantaggio aggiuntivo nel formare una coalizione: la worth combinata di due coalizioni è esattamente la somma delle loro worth individuali.
- **Formula**: 
$$\forall S, T \subset N, \text{ se } S \cap T = \emptyset$, allora $v(S \cup T) = v(S) + v(T)$$
Qui, formare una coalizione non apporta alcun valore aggiunto rispetto alla somma dei singoli benefici.

3. **Giochi a Somma Costante**:
- La somma della **worth** di una coalizione **S** e quella dei giocatori fuori dalla coalizione $N \setminus S$ è sempre pari alla worth totale **v(N)**.
- Formula: 
$$\forall S \subset N$, allora $v(S) + v(N \setminus S) = v(N)$$
Questo implica che non importa come si distribuisce la ricchezza tra le coalizioni e i non membri, la somma sarà sempre costante.
* Dove $v(N \setminus S)$ è il valore che può ottenere la coalizione formata da tutti i giocatori che non sono in $S$.

4. **Giochi Convessi**:
- La ricchezza combinata di due coalizioni è maggiore o uguale alla somma delle loro worth individuali meno la worth della loro intersezione.
- Formula: 
$$\forall S, T \subset N, \\ v(S \cup T) \geq v(S) + v(T) - v(S \cap T)$$
Questo tipo di gioco incentiva la collaborazione, perché formare coalizioni conviene più che agire singolarmente o in piccoli gruppi.
* Estende il concetto di superadditività anche a coalizioni non disgiunte, mediante la presenza dell'intersezione nella formula.

5. **Giochi Semplici**:
- La worth è binaria, ovvero può essere solo 0 o 1. Un esempio tipico di questi giochi sono i sistemi di voto, dove una coalizione "vince" (worth = 1) o "perde" (worth = 0).
- Formula: 
$$\forall S \subset N, v(S) \in \{0, 1\}$$
Quando un gioco semplice è anche a somma costante, si parla di **proper simple games**, dove la somma delle worth delle coalizioni e dei non membri è costante.
## Esempio

Questo esempio riguarda la distribuzione della **ricchezza** (o "worth") in un gioco coalizionale tra tre giocatori A, B, C. 
### Dati iniziali

- Abbiamo tre giocatori: A\), B\), e C\).
- La **worth** di ogni singolo giocatore (da solo) è **0**:
- v(A) = 0\)
- v(B) = 0\)
- v(C) = 0\)

- La **worth** delle coalizioni formate da due giocatori è la seguente:
- v(\{A, B\}) = 20\)
- v(\{A, C\}) = 30\)
- v(\{B, C\}) = 40\)

- La **worth** della coalizione **grand coalition** (tutti e tre i giocatori insieme) è $v(\{A, B, C\}) = 42$
### Obiettivo: Distribuire la ricchezza

Il nostro obiettivo è distribuire la **worth totale** v(\{A, B, C\}) = 42\) tra i tre giocatori A\), B\), e C\), rispettando alcune condizioni.
### Imputazione iniziale: x = [4, 14, 24]\)

In questo esempio, viene data una **imputazione** (una possibile distribuzione della ricchezza): x = [4, 14, 24]), cioè:
- A) riceve **4**,
- B) riceve **14**,
- C) riceve **24**.
#### Condizioni da rispettare:

1. **Efficienza**:
- La somma delle ricchezze assegnate deve essere uguale alla **worth totale**. In questo caso:
\[4 + 14 + 24 = 42\]
La condizione di **efficienza** è soddisfatta, perché la somma è 42.

2. **Razionale individualmente**:
- Ogni giocatore deve ricevere almeno quanto otterrebbe se giocasse da solo, cioè il suo valore individuale $v(\{i\}).$ Poiché v(A) = v(B) = v(C) = 0\), tutti i giocatori ricevono più di **0**, quindi la condizione di **razionalità individuale** è soddisfatta.

### Problema: Insoddisfazione delle coalizioni

Nonostante l'imputazione sia efficiente e razionalmente valida, le **coalizioni** non sono soddisfatte. Ad esempio:
- La coalizione \{A, B\}\) riceve 4 + 14 = 18\), ma la loro worth è 20\), quindi sono insoddisfatti.
- La coalizione \{A, C\}\) riceve 4 + 24 = 28\), ma la loro worth è 30\), quindi sono insoddisfatti.
- La coalizione \{B, C\}\) riceve 14 + 24 = 38\), ma la loro worth è 40\), quindi sono insoddisfatti.

### Tentativo di risolvere con un sistema lineare

Per cercare una distribuzione equa, possiamo impostare un **sistema di disequazioni** basato sulle coalizioni:
1. $x_A + x_B ≥20$ (la coalizione \{A, B\}\) deve ricevere almeno 20)
2. $x_B + x_C ≥ 40$\) (la coalizione \{B, C\}\) deve ricevere almeno 40)
3. $x_A + x_C ≥ 30$\) (la coalizione \{A, C\}\) deve ricevere almeno 30)
4. $x_A + x_B + x_C = 42$ (la somma totale deve essere uguale a 42)
### Contraddizione

Semplificando queste disequazioni, otteniamo:
$$x_A + x_B + x_C ≥ 45$$
Questa disequazione va in **conflitto** con l'equazione ($x_A + x_B + x_C = 42$\), perché non è possibile avere contemporaneamente una somma che sia uguale a 42 e maggiore o uguale a 45.
### Conclusione

Non esiste una **soluzione** che soddisfi contemporaneamente tutte le coalizioni e rispetti la condizione di efficienza.
### Caso alternativo: x = [5, 15, 25]\)

Se invece la **worth totale** fosse 45 (cioè $x_A + x_B + x_C = 45$\)), un'imputazione come x = [5, 15, 25]renderebbe tutti contenti, poiché:
- \{A, B\}\) riceve 5 + 15 = 20\), che soddisfa la loro worth.
- \{A, C\}\) riceve 5 + 25 = 30\), che soddisfa la loro worth.
- \{B, C\}\) riceve 15 + 25 = 40\), che soddisfa la loro worth.
Tuttavia, questo non è possibile nel contesto del problema, poiché la ricchezza totale è 42, non 45.

## Core di un Gioco Coalizionale

Il **core** è un concetto fondamentale nella teoria dei giochi coalizionali e rappresenta l'insieme delle distribuzioni (o imputazioni) della **ricchezza** tra i giocatori che sono **stabili**, ovvero tali che nessuna coalizione abbia incentivi a "rompere" l'accordo e formare una coalizione separata.

#### Definizione Formale

Dato un gioco coalizionale $G = (N, v)$ , dove:
- N è l'insieme dei giocatori,
- v(S) è la **worth** di una coalizione $S ⊆ N$ , ovvero il valore che quella coalizione può ottenere collaborando
Il **core** è l'insieme delle imputazioni $x \in X(G)$ tali che:
$$\left\{ x \in X(G):\ \ 
\forall S \subseteq N, \quad v(S) \leq \sum_{i \in S}x_{i} \right\}

$$
Ovvero, il valore assegnato a una coalizione S (la somma delle ricchezze dei giocatori nella coalizione S) non deve **superare** la somma delle utilità allocate ai giocatori di **S**.

#### Stabilità del Core

Il core è importante perché garantisce la **stabilità** della coalizione. Se scegliamo una distribuzione che appartiene al core, allora:
- Nessuna coalizione ha un incentivo a staccarsi e formare una coalizione separata, perché ogni coalizione S ottiene **almeno** quanto otterrebbe formando una coalizione indipendente.
- Ogni giocatore riceve almeno il proprio contributo individuale.

Tuttavia, il core può essere **vuoto**, e, anche se non è vuoto, può contenere molte possibili soluzioni.

### Proprietà delle Soluzioni nel Core

#### 1. Simmetry Axiom (Assioma di Simmetria)

Se due giocatori sono **interscambiabili**, cioè hanno lo stesso ruolo e contribuiscono allo stesso modo alla coalizione, allora dovrebbero ricevere la stessa **worth**. Formalmente:
$$
\forall S \subseteq N,\ \text{ tale che } i, j \notin S, \quad v(S \cup \{i\}) = v(S \cup \{j\})
$$
Questo significa che, se la worth di una coalizione non cambia scambiando i giocatori i e j, allora questi due giocatori dovrebbero ricevere la stessa ricompensa.

#### 2. Dummy Players (Giocatori Nulli)

Un **giocatore nullo** è un giocatore il cui contributo marginale a qualsiasi coalizione è sempre uguale al valore che otterrebbe da solo. Formalmente, per ogni $S \subseteq N$ tale che $i \notin S$ :
$$
v(S \cup \{i\}) - v(S) = v(\{i\})
$$
Questo significa che, se il contributo di un giocatore i a una coalizione S è pari a ciò che otterrebbe giocando da solo, quel giocatore dovrebbe ricevere esattamente quella quantità.

#### 3. Additivity (Additività)

Se abbiamo due giochi $G_{1}=(N, v_{1}) \ \ e \ \  G_{2}=(N, v_{2})$ con la stessa serie di giocatori, possiamo combinare i due giochi sommandone le worth function:
$$
(v_1 + v_2)(S) = v_1(S) + v_2(S) \quad \forall S \subseteq N
$$
Una soluzione $\Psi$ è **additiva** se la worth assegnata ai giocatori nella somma dei giochi è uguale alla somma delle worth assegnate nei giochi separati:

$$
\Psi_i(N, v_1 + v_2) = \Psi_i(N, v_1) + \Psi_i(N, v_2)
$$
In altre parole, un'imputazione è definita additiva se la soluzione per due giochi combinati è la somma delle soluzioni dei due giochi separati.

## Shapley Value

Lo **Shapley Value** è una soluzione importante nella teoria dei giochi coalizionali, che assegna a ogni giocatore una quota della ricchezza totale, basata sui suoi **contributi marginali** a tutte le possibili coalizioni. Viene considerato una soluzione **meritocratica**, poiché si basa su quanto un giocatore contribuisce a una coalizione rispetto a quanto la coalizione ottiene senza di lui.
- Lo Shapley Value è l'unica **pre-imputazione** (una distribuzione che non deve necessariamente rispettare la razionalità individuale) che soddisfa le 3 proprietà elencate prima.
- Lo Shapley Value esiste **sempre** in qualsiasi gioco, anche se non sempre rispetta le condizioni di razionalità individuale.
#### Caratteristiche chiave:

- **Equità**: Assicura che ogni giocatore riceva una quota proporzionale al suo contributo.
- **Stabilità**: È una soluzione che può essere sempre trovata.
- **Meritocrazia**: Si basa sui contributi effettivi che un giocatore fornisce alle coalizioni.

#### Formula dello Shapley Value

Per calcolare lo Shapley Value per un giocatore i, si tiene conto del contributo marginale che i apporta a tutte le possibili coalizioni S ⊆ N (dove i non fa parte di S).

La formula è la seguente:

$$
\phi_i(N, v) = \frac{1}{|N|!} \sum_{S \subseteq N \setminus \{i\}} |S|! \cdot (|N| - |S| - 1)! \cdot [v(S \cup \{i\}) - v(S)]
$$

#### Interpretazione

Lo shapley value puo essere interpretato come una **media pesata** dei contributi marginali del giocatore $i$ in ogni possibile coalizione $s$ che non contiene $i$. In altre parole, si calcola il contributo di $i$ a ogni coalizione e si fa una media considerando **tutti i possibili ordinamenti** in cui i puo essere aggiunto alla coalizione.

#### Esempio

Immaginiamo tre giocatori A, B, e C. Se A, B, e C formano coalizioni diverse, lo Shapley Value di un giocatore è la media dei suoi contributi marginali calcolati su tutte le coalizioni possibili. Per esempio, se aggiungere A a una coalizione S aumenta la worth di S , il contributo marginale di A in quella coalizione sarà $v(S ∪ \{A\}) - v(S)$ 

## Nucleolo in un gioco coalizionale

Il **nucleolo** è un concetto di soluzione per i giochi coalizionali, che si concentra sulla riduzione dell'insoddisfazione massima tra i giocatori. L'idea alla base del nucleolo è quella di minimizzare l'insoddisfazione, o **eccesso** (excess), delle coalizioni in modo gerarchico: prima si riduce l'insoddisfazione massima, poi quella successiva e così via, fino ad ottenere una distribuzione delle risorse più equa e stabile possibile.

#### Definizione

- **Eccesso e(x, S)**: L'eccesso di una coalizione S rispetto a una imputazione x è la differenza tra la worth v(S) della coalizione S e la somma dei valori assegnati ai giocatori in S dalla distribuzione x :
$$ 
e(x, S) = v(S) - \sum_{i \in S} x_i
$$
Questo misura quanto la coalizione S si sente insoddisfatta rispetto alla distribuzione x .

- **Vettore degli Eccessi** $\Theta(x)$ : È un vettore che contiene gli eccessi $e(x, S)$ per tutte le coalizioni $S \subseteq N$ , ordinati in modo **non crescente** (dal più grande al più piccolo). Questo vettore permette di confrontare le imputazioni in base all'insoddisfazione che creano nelle coalizioni.

#### Definizione del Nucleolo

Il **nucleolo** è l'imputazione x che minimizza il vettore degli eccessi $( \Theta(x) )$. In altre parole, il nucleolo è la soluzione in cui la massima insoddisfazione è minimizzata, e in caso di parità su questa, si minimizza l'insoddisfazione successiva, e così via.

Formalmente, il **nucleolo** N(G) di un gioco G = (N, v) è l'insieme delle imputazioni x tali che:

$$
N(G) = \{x \in X(G): \ \nexists y \ \text{ tale che } \ \Theta(y) \succ \Theta(x)\}
$$

Dove $\Theta(y) \succ \Theta(x)$ significa che il vettore degli eccessi di y ha meno insoddisfazione rispetto a quello di x

#### Proprietà del Nucleolo

- **Unicità**: Il nucleolo è costituito da un unico vettore x, a meno che i valori non siano interi. Questo significa che esiste una sola distribuzione delle risorse che minimizza l'insoddisfazione.

#### Relazione con l’ε-core

Il concetto di **ε-core** è una generalizzazione più debole del core, dove si permette una certa insoddisfazione, regolata dal parametro $\epsilon$. L'**ε-core** è l'insieme di imputazioni x tali che:

$$\sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \subseteq N
$$
All'aumentare di $\epsilon$ , l'**ε-core** si espande e aumenta la probabilità di trovare una soluzione.
#### Iterazione verso il Nucleolo

Il processo per trovare il nucleolo può essere visto come una procedura iterativa. 
* Si parte risolvendo il problema di minimizzare $\epsilon$ in un sistema lineare, che descrive il **least core** (la versione più forte dell'ε-core). 
* Una volta ottenuto un valore ottimale $( \epsilon_1 )$, si identifica un insieme di **coalizioni critiche** $S_i$ , che sono quelle coalizioni per cui l'eccesso è esattamente pari a $\epsilon_1$ .
* Il passo successivo è ripetere il processo considerando solo le coalizioni non critiche, minimizzando di nuovo *ε*, e così via, finché non si converge al **nucleolo**.

Il problema lineare al passo 1 è:

$$
\min \epsilon
$$
$$
\text{soggetto a: } \sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \subseteq N
$$
$$
x(N) = v(N)
$$

Calcoliamo il valore ottimo, che chiamiamo $\epsilon_1$. È possibile che esistano delle coalizioni che soddisfano i vincoli di eguaglianza, le cosiddette *coalizioni critiche* $S_i$, definite come segue:
$$x^*(S_i) = v(S_i) - \epsilon_1 \quad$$
dove $x^*(S_i)$ è la soluzione ottima per la coalizione i .

A questo punto, definendo con F₁ l'insieme delle coalizioni critiche ottenute al passo 1 (quello che abbiamo appena esaminato), possiamo rendere ancora più forte il least core iterando il procedimento. Questo significa risolvere il seguente problema lineare:

$$
\begin{array}{rl}
\text{minimize} & \epsilon \\
\text{subject to} & \sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \in 2^N - F_1 \\
& x(S) = v(S) - \epsilon_1, \quad \forall S \in F_1 \\
& x(N) = v(N)
\end{array} 

$$
Questo procedimento si ripete, restringendo sempre di più l'insieme delle coalizioni critiche, fino a ottenere la soluzione finale, che rappresenta il nucleolo.

### Contested Garment Rule

La **Contested Garment Rule** è un criterio per dividere un'eredità tra due creditori con debiti $d_1 \ e \ d_2 \ \ (\text{dove} \ \ (d_1 \leq d_2))$ e un totale da dividere $e$ con $(0 \leq e \leq d)$:

1. Il creditore 1 può concedere al creditore 2 un importo massimo di $\max(e - d_1, 0) .$
2. Il creditore 2 può concedere al creditore 1 un importo massimo di $\max(e - d_2, 0) .$
3. L'importo rimanente, cioè $( e - \max(e - d_1, 0) - \max(e - d_2, 0) )$, viene diviso equamente tra i due creditori.

Questa regola combina linearmente il criterio di divisione eguale dei guadagni e delle perdite. È stato dimostrato che esiste una sola divisione dell'eredità che segue questa regola e che tale divisione corrisponde esattamente al nucleolo del gioco associato, dove ogni coalizione pretende il massimo tra 0 e ciò che le è concesso da chi non fa parte della coalizione.

## Aste e Mechanism Design

Si occupano di stabilire regole per le aste in modo da incentivare comportamenti strategici da parte degli agenti. Esistono due principali categorie di aste:

1. **Single good**: offerte su un unico oggetto.
2. **Multiple goods**: offerte su insiemi di oggetti, tipiche delle *combinatorial auctions*.

Ci concentriamo sulle aste *single good*, in particolare:

- **English auction**: i partecipanti fanno offerte crescenti e il bene viene assegnato al maggiore offerente. L'obiettivo è vendere al prezzo più alto possibile. L'utilità di un partecipante i è $u_i = v_i - S_i$ se vince, altrimenti è 0, dove $v_i$ è il valore segreto che attribuisce all'oggetto e $S_i$ l'offerta fatta.

- **Japanese auction**: simile a quella inglese, ma con il gestore che fissa progressivamente le offerte. I giocatori scelgono se uscire o continuare fino a quando rimane un solo vincitore.

- **Dutch auction**: un'asta al ribasso in cui si parte da un prezzo alto e si scende. Vince chi fa la prima offerta.

Il **Mechanism Design** mira a creare regole che inducano gli agenti a seguire strategie ottimali, garantendo un *mechanism design truthful*, ovvero rendendo conveniente rivelare le proprie vere preferenze o valori.

## Sealed-Bid Auctions (Aste a busta chiusa)

In questo tipo di asta, le offerte sono **segrete** e il vincitore è colui che offre la cifra più alta.
Gli agenti devono basarsi su:
- Il valore che attribuiscono al bene.
- Il numero di partecipanti.
- Le caratteristiche dei partecipanti (budget, tendenze nelle offerte).
- Potrebbero avere informazioni sulle distribuzioni di probabilità che descrivono come gli altri agenti valutano il bene.

##### Distribuzioni IPV (Independent Private Values)

- Ogni partecipante ha un valore attribuito al bene indipendente da quello degli altri.
- Ciascun giocatore conosce la distribuzione di probabilità da cui vengono estratti i valori degli altri partecipanti, ma non conosce i valori esatti.
Dunque, nelle aste a busta chiusa, la strategia di un partecipante può essere influenzata dalla distribuzione di probabilità delle valutazioni altrui, rendendo complessa la scelta di una strategia ottimale.
## Second-Price Auctions (Aste al secondo prezzo)

Ogni partecipante fa la propria offerta e vince chi ha offerto di più, ma paga il secondo prezzo più alto, non la propria offerta.
- **Mechanism Design truthful**: l'asta al secondo prezzo incoraggia i partecipanti a fare offerte corrispondenti al reale valore che attribuiscono al bene. Questa strategia è ottimale indipendentemente dalle strategie degli altri.

##### CASO 1: Gli altri agenti offrono meno del valore $v_i$ che attribuisco al bene

- Se offro $s_i < v_i$, rischio di perdere e ottenere utilità $u_i = 0$.
- Se offro $s_i = v_i$, vinco e la mia utilità è $u_i = v_i - s_{\text{max}}$ dove $s_{\text{max}}$ è l’offerta più alta tra gli altri agenti).
- Se offro $s_i > v_i$, vinco, ma la mia utilità resta $u_i = v_i - s_{\text{max}}$, identica al caso precedente.

**Conclusione**: Offrire $s_i = v_i$ è la scelta ottimale. Offrire più o meno del valore che attribuisco al bene non cambia l'utilità, quindi mi conviene essere veritiero.

**CASO 2: Altri agenti offrono più del valore $v_i$ 

- Se $s_{\text{max}} > v_i$ :
- Se offro $s_i < v_i$ o $s_i = v_i$, perdo e guadagno $u_i = 0$ .
- Se offro $s_i > v_i$ , potrei vincere ma la mia utilità sarebbe negativa $u_i = v_i - s_{\text{max}} < 0$ .

**Conclusione**: Offrire più del valore che attribuisco al bene mi può portare a un guadagno negativo. Poiché non posso prevedere esattamente come si comporteranno gli altri, la strategia migliore resta offrire $s_i = v_i$, cioè il valore esatto che attribuisco al bene.

- Il meccanismo dell’asta giapponese si comporta in modo simile all'asta al secondo prezzo: anch’esso è **truthful** e incoraggia i partecipanti a fare offerte veritiere.
## First-Price Auctions: Aste al Primo Prezzo

Nelle **aste al primo prezzo**, i partecipanti presentano offerte segrete e chi offre di più vince, ma deve pagare l'importo offerto 

#### Punti Chiave:

1. **Offerte a busta chiusa**: I partecipanti non sanno l'offerta degli altri.
2. **Valore stimato del bene**: Ogni giocatore ha un proprio valore $v_i$ , che rappresenta quanto è disposto a pagare per l'oggetto.
3. **Strategia ottimale**: I partecipanti devono considerare le offerte degli altri e ragionare su come massimizzare il proprio guadagno.
#### Esempio con 2 Giocatori:

- **Assunzioni**:
- Giocatore 2 offre metà del valore che attribuisce al bene: $s_2 = \frac{1}{2} v_2$ .
- I valori sono distribuiti uniformemente tra 0 e 1.
- **Domanda**: Qual è la miglior strategia per il giocatore 1?

### Calcolo del Valore Atteso dell'Utilità

1. **Utilità del Giocatore 1** $u_1$ : Guadagno che ottiene se vince, meno l'offerta fatta $s_1$ .
$$ 
E[u_1] = \int_0^{2s_1} u_1 dv_2
$$
Questo perché se l'offerta del giocatore 2 $s_2 = \frac{1}{2} v_2$ è inferiore a $2s_1$ , il giocatore 1 vince.

2. **Condizioni al contorno**:
- Se $v_2 > 2s_1$ , il giocatore 2 offre più di $s_1$ , quindi l'utilità del giocatore 1 è zero (secondo integrale).

3. **Calcolo dell'integrale**:
$$
E[u_1] = \int_0^{2s_1} (v_1 - s_1) dv_2 = 2v_1s_1 - 2s_1^2
$$

### Massimizzazione dell'Utilità

Per trovare l'**offerta ottimale** $s_1$ , si calcola la derivata dell'utilità attesa rispetto a $s_1$ :
$$
\frac{d}{ds_1} E[u_1] = 2v_1 - 4s_1
$$
Imponendo che la derivata sia uguale a zero:
$$
2v_1 - 4s_1 = 0 \quad \Rightarrow \quad s_1 = \frac{1}{2} v_1
$$
**Conclusione**: La strategia ottimale per il giocatore 1 è offrire metà del proprio valore $v_1$, proprio come fa il giocatore 2.

### Equilibrio

Se entrambi i giocatori offrono metà del proprio valore, nessuno ha incentivo a cambiare strategia. Questo è un **equilibrio di Nash**.

### Caso Generale con $n$ Giocatori

Esiste un **teorema generale** che afferma che in un'asta al primo prezzo con $n$ agenti **risk-neutral** (neutri al rischio) e distribuzioni uniformi, l'unico **equilibrio simmetrico** ha la forma:
$$
\left( \frac{n-1}{n} v_1, \frac{n-1}{n} v_2, \dots, \frac{n-1}{n} v_n \right)
$$
Questo significa che ogni giocatore offre una frazione $\frac{n-1}{n}$ del proprio valore.

### Riassunto

- In un'asta al primo prezzo, la strategia ottimale per ciascun giocatore è offrire una frazione del proprio valore.
- In un contesto con 2 giocatori, questa frazione è la metà del valore stimato.
- Per $n$ giocatori, la frazione ottimale è $\frac{n-1}{n}$.

## Giochi Strategici

- **Definizione**: Giochi in cui ogni giocatore agisce per il proprio interesse individuale, senza considerare coalizioni o accordi tra giocatori.
- **Obiettivo**: Massimizzare il proprio profitto, spesso in presenza di obiettivi contrastanti tra i giocatori.

#### Dilemma del Prigioniero

- **Descrizione**: Due prigionieri devono decidere se confessare un crimine o non confessare. La loro decisione influenza la pena che riceveranno.
- Il Dilemma del Prigioniero è un esempio classico dove entrambi i giocatori finiscono in una situazione subottimale (confessando) anche se avrebbero ottenuto un risultato migliore cooperando (non confessando).
Consideriamo una tabella dei payoff 

| **Prigioniero 2** \ **Prigioniero 1** | **Non Confessa** | **Confessa** |
| ------------------------------------- | ---------------- | ------------ |
| **Non Confessa** | (-1, -1) | (-4, 0.4) |
| **Confessa** | (0.4, -4) | (-0.3, -0.3) |
#### Spiegazione dei Payoff:

- **(Non Confessa, Non Confessa)**: Entrambi ottengono (-1, -1). Entrambi ricevono una pena minore perché hanno collaborato e non hanno confessato.
- **(Non Confessa, Confessa)**: Il prigioniero che non confessa riceve -4, mentre quello che confessa riceve 0.4. Il prigioniero che confessa beneficia di una riduzione della pena, mentre l'altro subisce una pena più grave.
- **(Confessa, Non Confessa)**: Il prigioniero che confessa riceve 0.4, mentre quello che non confessa riceve -4. Il prigioniero che confessa ottiene una ricompensa, mentre l'altro riceve una pena pesante.
- **(Confessa, Confessa)**: Entrambi ricevono (-0.3, -0.3). Entrambi confessano, quindi ricevono una pena minore rispetto a se uno solo avesse confessato.

Se il **Prigioniero 1** decide di non confessare e il **Prigioniero 2** confessa, i payoff saranno:
- **Prigioniero 1**: -4
- **Prigioniero 2**: 0.4
I Prigioniero 1 subisce una pena più severa (-4) mentre il Prigioniero 2 riceve una pena ridotta (0.4).

- **Equilibrio di Nash**:
- **Situazione**: Entrambi i prigionieri confessano.
- **Motivazione**: Anche se non confessare sarebbe più vantaggioso se l’altro non confessa, confessare diventa la scelta migliore se si sospetta che l’altro confessi.
- **Risultato**: L’equilibrio di Nash si verifica quando entrambi i giocatori scelgono di confessare, poiché nessuno ha incentivo a deviare dalla propria scelta data la scelta dell’altro.

# Equilibrio di Nash

#### Definizione Formale di Gioco Strategico

1. **Giocatori**: Un insieme di $N$ giocatori.
2. **Azioni/Strategie**:
	- Per ogni giocatore $i \in N$, esiste un insieme di azioni ammissibili $S_i$.
	- Le azioni di ciascun giocatore sono indicate con $s_i$.
3. **Profili di Azione**:
	- L'insieme dei profili di azione è $S = \times_{j \in N} \ S_j = S_1 \times \cdots \times S_N$.
		- $S$ rappresenta il prodotto cartesiano delle azioni ammissibili.
	- Un **profilo di azione** è una combinazione delle azioni scelte da tutti i giocatori, rappresentato come una $N$-upla di azioni ammissibili $(s_1, ..., s_N)$.
4. **Utilità**:
	- Per ogni giocatore $i \in N$, esiste una funzione di utilità $u_i : S \rightarrow \mathbb{R}$ che assegna un valore reale a ciascun profilo di azione.

#### Definizione di Equilibrio di Nash

Un profilo di azione $S^*$ è un equilibrio di Nash se:
$$
u_i(S^*) \geq u_i(s_i, S^*_{-i}) \quad \forall i \in N, \quad \forall s_i \in S_i
$$
dove $S^*_{-i}$ rappresenta le scelte degli altri giocatori. In altre parole, dato un profilo di azione $S^*$, nessun giocatore può migliorare la propria utilità cambiando unilateralmente la propria azione, dato che le azioni degli altri giocatori rimangono costanti.

==Per ogni giocatore i, il valore reale del profilo d'azione S* è maggiore o uguale del valore reale del profilo d'azione che rappresenta le scelte degli altri giocatori. ==

## Esempio: Gioco dei Bach e Stravinsky

Due persone devono decidere a quale concerto andare (Bach o Stravinsky). Le preferenze e i payoff sono i seguenti:

| **Giocatore 2** \ **Giocatore 1** | **Bach** | **Stravinsky** |
|----------------------------------|----------|----------------|
| **Bach** | (2.1, 2.1) | (0, 0) |
| **Stravinsky** | (0, 0) | (1, 1) |

- **Payoff**:
- Se entrambi vanno a **Bach**, ottengono (2.1, 2.1).
- Se entrambi vanno a **Stravinsky**, ottengono (1, 1).
- Se uno va a **Bach** e l'altro a **Stravinsky**, entrambi ricevono (0, 0).

**Equilibri di Nash**:

- (Bach, Bach) è un equilibrio di Nash: Se entrambi vanno a Bach, nessuno ha incentivo a cambiare, dato che preferiscono stare insieme piuttosto che andare separatamente.
- (Stravinsky, Stravinsky) è un altro equilibrio di Nash per lo stesso motivo.

## Funzione di Best Response

La funzione di best response per un giocatore $i$ dato un profilo di azioni degli altri giocatori $s_{-i}$ è:
$$
B_i(s_{-i}) = \{s_i \in S_i : u_i(s_i, s_{-i}) \geq u_i(s'_i, s_{-i}), \forall s'_i \in S_i\}
$$
Rappresenta l'insieme delle strategie $s_i \in S_{i}$ tali che l'utilità $u_i$ ottenuta, dato il profilo di strategie $s_{-i}$ degli altri giocatori, è maggiore o uguale all'utilità ottenuta con qualsiasi altra strategia alternativa $s'_i \in S_i$.

Un profilo di azione $s^*$ è un equilibrio di Nash se e solo se ogni azione $s^*_i$ è una best response alle azioni degli altri giocatori, ovvero:
$$
s^*_i \in B_i(s^*_{-i}) \quad \forall i \in N
$$

### Utilità Attesa:

L’utilità attesa di una lotteria è calcolata come:
$$
U(p) = \sum_{z \in Z} p(z) \cdot v(z)
$$
 L'utilità della lotteria $U(p)$ è calcolata come la somma ponderata dei valori $v(z)$ dei premi, con i pesi dati dalle probabilità. 

- La funzione di utilità di Von Neumann-Morgenstern $v(\cdot)$ rappresenta le preferenze sui premi di una lotteria.
 - Questa funzione rappresenta le preferenze dei giocatori in termini di utilità attesa e soddisfa gli assiomi di indipendenza e continuità, garantendo valori unici e consistenti per rappresentare le preferenze tra diverse lotterie.
### Strategie Miste

Una **strategia mista** è una distribuzione di probabilità sulle azioni che un giocatore può scegliere. In altre parole, ogni giocatore sceglie un’azione non in modo deterministico, ma con una certa probabilità.
Questo approccio garantisce sempre l’esistenza di un equilibrio di Nash, anche in giochi complessi con strategie non deterministiche.

**Motivazione**:

- In molti giochi, le scelte possono non portare a risultati certi (sono stocastiche). Invece di optare per una scelta deterministica,i giocatori possono decidere di mescolare le loro scelte per massimizzare l’utilità attesa, che è una misura dell'aspettativa dei risultati di tutte le azioni possibili.

**Riassumendo**: Una strategia mista è quando un giocatore sceglie le sue azioni assegnando *probabilità* a ciascuna di esse.

### Utilità Attesa di una Strategia Mista:

L'obiettivo è calcolare quanto guadagno (o utilità) ci si aspetta da una strategia mista.
- La **utilità attesa** di una strategia mista $\sigma$ per un giocatore $i$ è:
$$
U_i(\sigma) = \sum_{s_i \in S_i} \sigma_i(s_i) \cdot U_i(e(s_i), \sigma_{-i})
$$
Dove:
- $\sigma_i(s_i)$ è la probabilità con cui il giocatore $i$ sceglie l'azione $s_i$.
- $U_i(e(s_i), \sigma_{-i})$: guadagno del giocatore $i$ quando:
 - Gioca **solo** $s_i$ (con probabilità 1) 
 - Gli altri giocatori seguono la loro strategia mista $\sigma_{-i}$.

Rappresenta la somma dei prodotti tra le probabilità con cui il giocatore $i$ sceglie l'azione $s_i$ e il guadagno quando gioca solo l'azione $s_i$ e gli altri giocatori scelgono la loro strategia mista.

### Equilibrio di Nash con Strategie Miste:

- Un profilo di strategie miste $\sigma^*$ è un equilibrio di Nash se:
$$
U_i(\sigma^*) \geq U_i(\sigma'_i, \sigma^*_{-i}) \quad \forall i \in N, \quad \forall \sigma'_i
$$
Dove $\sigma'_i$ è qualsiasi strategia mista alternativa per il giocatore $i$.

# Teorema di Nash:

Il **Teorema di Nash** afferma che in ogni *gioco non cooperativo* con un numero finito di giocatori e un numero finito di strategie pure, esiste almeno un equilibrio di Nash in strategie miste. Questo equilibrio ha due proprietà fondamentali:
* Garantisce l'esistenza di un equilibrio anche quando non esiste un equilibrio in strategie pure. 
* In questo equilibrio, tutte le azioni che hanno una probabilità positiva nella strategia mista di un giocatore sono le migliori risposte alle strategie degli altri giocatori.

#### Definizione di Supporto di una Strategia Mista

Il **supporto** di una strategia mista $\sigma_i$ per il giocatore $i$ è l'insieme delle azioni che hanno probabilità $\sigma$ positiva nella strategia:
$$
\text{supp}(\sigma_i) = \{ s_i \in S_i : \sigma_i(s_i) > 0 \}
$$
In altre parole, ==il supporto è l'insieme delle azioni che il giocatore $i$ sceglie con una probabilità non nulla.==

#### Definizione Formale **Teorema di Nash**

Esiste sempre un equilibrio di Nash con strategie miste. 
Un profilo di strategie miste $\sigma^* = (\sigma^*_i)_{i \in N}$ è un equilibrio di Nash se e solo se:

- Per ogni giocatore $i \in N$, tutte le strategie pure nel supporto di $\sigma^*_i$ sono **best response** alle strategie miste degli altri giocatori $\sigma^*_{-i}$.

### Prima Parte: Dimostrazione $\Rightarrow$

Se $\sigma^*$ è un equilibrio di Nash, allora tutte le azioni nel supporto di $\sigma^*$ sono best response.

1. **Assunzione per Assurdo**:
- Supponiamo che $\sigma^*$ non sia un equilibrio di Nash. Allora, esiste almeno un giocatore $i$ per il quale una delle sue azioni $s_i$ nel supporto di $\sigma^*_i$ non è una **best response** rispetto alle strategie degli avversari $\sigma^*_{-i}$.

2. **Conseguenza**:
- Ciò significa che esiste una strategia alternativa $s_j \in B(\sigma^*_{-i})$ che offre una migliore utilità per il giocatore $i$ rispetto a $s_i$.

3. **Aggiustamento delle Probabilità**:
- Se $s_j$ appartiene già al supporto, si può aumentare la sua probabilità e ridurre quella di $s_i$ per migliorare l'utilità del giocatore $i$.
- Se $s_j$ non appartiene al supporto, la probabilità di $s_j$ può essere introdotta riducendo quella di $s_i$.
- Il giocatore *i* potrebbe aumentare la sua utilità spostando la probabilità da $s_i$ a $s_j$.

4. **Contraddizione**:
- In entrambi i casi, l'utilità del giocatore *i* aumenta, quindi otteniamo che:
$$
U_i(\sigma'_i, \sigma^*_{-i}) > U_i(\sigma^*, \sigma^*_{-i})
$$
- Questo contraddice l'assunzione che $\sigma^*$ sia un equilibrio di Nash, perché $\sigma'_i$ sarebbe una strategia migliore.

### Seconda Parte: Dimostrazione $\Leftarrow$

1. **Assunzione Iniziale**:
- Supponiamo che $\sigma^*$ non sia un equilibrio misto di Nash. Esiste quindi una strategia $\sigma'_i$ per il giocatore $i$ tale che:
$$
U_i(\sigma'_i, \sigma^*_{-i}) > U_i(\sigma^*)
$$

2. **Conseguenza**:
- Questo implica che esiste una strategia nel supporto di $\sigma'_i$ che dà un'utilità maggiore rispetto ad almeno una delle azioni nel supporto di $\sigma^*_i$ in risposta alle strategie degli avversari $\sigma^*_{-i}$.

3. **Contraddizione**:
- Se non tutte le azioni nel supporto di $\sigma^*_i$ sono best response a $\sigma^*_{-i}$, ciò contraddice la definizione di equilibrio di Nash.

### Corollario 1:

Per ogni giocatore $i$ e per tutte le azioni $s'_i$ e $s''_i$ nel supporto di $\sigma^*_i$, vale che:
$$
U_i(s'_i, \sigma^*_{-i}) = U_i(s''_i, \sigma^*_{-i})
$$
Questo significa che tutte le azioni nel supporto di una strategia mista in equilibrio danno la stessa utilità attesa.

### Corollario 2:

Tutte le azioni nel supporto di $\sigma^*_i$ sono best response rispetto alle strategie $\sigma^*_{-i}$, ossia:
$$
\text{supp}(\sigma^*_i) \subseteq B(\sigma^*_{-i})
$$
Dove $B(σ^*_{-i})$ è l'insieme delle best response alle strategie degli altri giocatori.

## Esempio di Equilibrio di Nash Misto: Bach e Stravinsky

- **Scenario**: Due giocatori devono scegliere tra andare a un concerto di Bach o di Stravinsky. Le preferenze sono diverse, ma entrambi preferiscono essere insieme.

- **Probabilità**:
- Il giocatore 2 sceglie Bach con probabilità $\sigma_2(B)$ e Stravinsky con $1 - \sigma_2(B)$.

- **Calcolo per il Giocatore 1**:
- Utilità se gioca **Bach**: $U_1(B, \sigma_2) = 2\sigma_2(B)$
- Utilità se gioca **Stravinsky**: $U_1(S, \sigma_2) = 1 - \sigma_2(B)$

- **Condizione per Giocare Bach**:
- Il giocatore 1 preferisce Bach se $2\sigma_2(B) > 1 - \sigma_2(B)$, cioè quando $\sigma_2(B) > \frac{1}{3}$.

- **Strategia del Giocatore 1**:
- $\sigma_1(B) = 1$ se $\sigma_2(B) > \frac{1}{3}$
- $\sigma_1(B) = 0$ se $\sigma_2(B) < \frac{1}{3}$
- $\sigma_1(B) \in [0, 1]$ se $\sigma_2(B) = \frac{1}{3}$

- **Strategia del Giocatore 2**: 
- Simile per il giocatore 2, che preferisce Bach se $\sigma_1(B) > \frac{2}{3}$.

- **Equilibrio di Nash Misto**: 
- L'unico equilibrio non puro si verifica quando $\sigma_1(B) = \frac{2}{3}$ e $\sigma_2(B) = \frac{1}{3}$.
---
### Giochi Strategici:

- **Modalità di gioco**: I giocatori scelgono simultaneamente.
- **Informazione**:
- **Informazione completa**: Conosci le preferenze e utilità dell'avversario.
- **Informazione incompleta**: Non sai cosa sceglierà l'avversario e viceversa.

## Giochi in forma estesa

![[8) Teoria dei giochi 2024-12-17 10.59.47.excalidraw]]
I giochi in forma estesa rappresentano situazioni in cui i giocatori prendono decisioni in momenti diversi, considerando cosa è successo in passato. Questa rappresentazione si basa su un **game tree,** una struttura che mostra i nodi di decisione (dove un giocatore sceglie un'azione) e i nodi terminali (dove si determinano i risultati e le utilità).
- **Modalità di gioco**: Giocati in sequenza, tenendo conto degli eventi passati.
- **Informazione e Memoria**:
	- **Informazione perfetta**: Conosci tutto (dove ti trovi, cosa è successo, ecc.).
	- **Informazione imperfetta**: Non hai informazioni complete. I nodi indistinguibili per il giocatore sono raggruppati in un **information set**, e il giocatore deve scegliere una strategia basata su questo insieme, non sul singolo nodo.
	- **Memoria perfetta**: Ricordi tutto ciò che è successo.
	- **Memoria imperfetta**: Hai una memoria parziale degli eventi.
### Struttura:

1. **Nodi di decisione**: Partizionati per ciascun giocatore. Ogni giocatore sceglie un'azione da un insieme di azioni disponibili, rappresentate dagli archi tra i nodi.
2. **Nodi terminali**: Qui si indicano le utilità per ciascun giocatore, determinate dalle azioni intraprese.
3. **Strategia**: È una funzione che, per ogni nodo in cui il giocatore si trova, restituisce un'azione da eseguire. Le strategie possono essere deterministiche o randomizzate.

### Subgame Perfect Equilibrium (SPE):

È un concetto fondamentale nei giochi estensivi. Un equilibrio di Nash è definito per l'intero gioco, ma l'SPE rafforza questo concetto:
- **Sottogiochi**: Parti del gioco che possono essere viste come giochi autonomi.
- **SPE**: Un equilibrio che non solo è un equilibrio di Nash per il gioco completo, ma anche per ogni sottogioco. Questo assicura che le strategie siano razionali non solo nell'insieme, ma in ogni possibile sottosituazione.

### Algoritmo Minimax:

La dimostrazione del teorema che ogni gioco estensivo a informazione perfetta ammette un Subgame Perfect Equilibrium di strategie pure si basa sul principio del Minimax, un algoritmo che cerca di minimizzare le perdite massime in situazioni di conflitto, tipicamente usato in giochi competitivi a due giocatori.

# Strategie nei Giochi

### 1. Strategie Pure

Una strategia pura consiste nella selezione di una singola azione da un insieme di azioni disponibili, senza alcun elemento di casualità. Il giocatore sceglie una mossa specifica per ogni situazione possibile.

##### Caratteristiche:

* Deterministica: la scelta dell'azione è fissa e prevedibile.
* Percorso definito: ogni strategia pura corrisponde ad un percorso univoco nel gioco.

**Esempio:** Nel gioco "carta-sasso-forbice", giocare sempre "sasso" è una strategia pura.

### 2. Strategie Non Pure

Una strategia non pura indica una scelta di azioni che varia a seconda del contesto del gioco, ma *senza* l'introduzione di elementi probabilistici. Si tratta di strategie condizionali, spesso utilizzate in giochi sequenziali o estesi.

##### Caratteristiche:

* Strategia condizionale: la scelta dell'azione dipende dallo stato del gioco.
* Assenza di probabilità: a differenza delle strategie miste, non vengono utilizzate probabilità per selezionare le azioni.

### 3. Strategie Miste

Una strategia mista assegna una probabilità ad ogni azione disponibile, introducendo così un elemento di casualità nella scelta del giocatore.

##### Caratteristiche:

* Probabilistica: la scelta dell'azione è governata da una distribuzione di probabilità.
* Utile in situazioni di conflitto: particolarmente utile quando non esistono strategie pure dominanti o equilibri di Nash in strategie pure.
* Equilibri di Nash: permette l'esistenza di equilibri di Nash anche in giochi dove le strategie pure non portano a soluzioni stabili.

**Esempio:** Nel gioco "carta-sasso-forbice", scegliere ogni mossa con probabilità 1/3 (33% "carta", 33% "sasso", 33% "forbice") è una strategia mista.

# Equilibrio di Nash e Strategie Miste

## 1. Equilibrio di Nash in un Gioco Strategico

Un equilibrio di Nash è una configurazione in cui nessun giocatore può migliorare il proprio payoff cambiando unilateralmente strategia.

* **Definizione formale:** $(\sigma_i^* \in S_i)$ è un equilibrio per il giocatore $i$ se $u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*)$ per ogni $\sigma_i \in S_i$, dove $\sigma_{-i}^*$ rappresenta le strategie degli altri giocatori.

* **Caratteristiche:** Può essere in strategie pure (una singola azione scelta) o in strategie miste (una distribuzione di probabilità su più azioni).

## 2. Best Response

La *best response* è la strategia ottimale per un giocatore, dato il profilo delle strategie degli altri giocatori.

* **Definizione formale:** $\text{BR}_i(\sigma_{-i}) = \{\sigma_i \in S_i : u_i(\sigma_i, \sigma_{-i}) \geq u_i(\sigma_i', \sigma_{-i}), \forall \sigma_i' \in S_i\}$.

* **Relazione con l'Equilibrio di Nash:** Un equilibrio di Nash è raggiunto quando tutti i giocatori stanno giocando una loro *best response* contemporaneamente.

## 3. Strategia Mista e il suo Equilibrio

Una strategia mista è una distribuzione di probabilità sulle strategie pure di un giocatore.

* **Esempio:** Un giocatore può scegliere la strategia A con probabilità 0.6 e B con probabilità 0.4.

* **Equilibrio in Strategia Mista:** Ogni giocatore assegna probabilità alle strategie in modo che la strategia scelta sia una *best response* contro le strategie miste degli altri. Questo permette di superare la limitazione dei giochi in strategie pure dove non sempre esiste un equilibrio.

## 4. Supporto di una Strategia Mista

Il supporto di una strategia mista è l'insieme delle strategie pure che ricevono probabilità positiva nella strategia mista.

* **Esempio:** Se una strategia mista assegna probabilità 0.5 ad A, 0.5 a B, e 0 a C, il supporto è {A, B}.

* **Implicazione:** Per un giocatore, ogni strategia nel supporto deve garantire lo stesso payoff atteso; altrimenti, si potrebbe riassegnare probabilità per migliorare il payoff.

## 5. Teorema di Nash

Il Teorema di Nash afferma che ogni gioco finito (con un numero finito di giocatori e strategie) ha almeno un equilibrio di Nash in strategie miste.

* **Dimostrazione:** Si basa su tecniche di topologia (teorema del punto fisso di Kakutani).

* **Significato Pratico:** Anche nei giochi senza equilibrio in strategie pure, si può sempre trovare un equilibrio considerando strategie miste. Questo assicura che ogni gioco "ben definito" abbia almeno un punto di stabilità.

## Connessioni

1. **Equilibrio di Nash** e **Best Response:** L'equilibrio di Nash si raggiunge quando tutti i giocatori giocano una loro *best response*.

2. **Strategia Mista** e **Supporto:** Le strategie miste ampliano le possibilità per raggiungere un equilibrio, utilizzando un supporto costituito da strategie pure.

3. **Teorema di Nash** e **Equilibrio in Strategia Mista:** Il teorema di Nash garantisce l'esistenza di un equilibrio in strategie miste per ogni gioco finito.

4. **Supporto** e **Best Response:** Nel supporto di una strategia mista, ogni strategia deve essere una *best response*, altrimenti il giocatore non utilizzerebbe quella strategia.
