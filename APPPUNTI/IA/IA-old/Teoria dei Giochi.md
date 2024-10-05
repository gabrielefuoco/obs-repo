Modella situazioni in cui agenti (umani o artificiali) interagiscono. Gli agenti possono competere, collaborare o negoziare per raggiungere i propri obiettivi.

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

### Informazione nei giochi:
- **Informazione perfetta**: I giocatori conoscono tutte le azioni e decisioni prese.
- **Informazione imperfetta**: Non tutti i giocatori conoscono l'intera situazione.
- **Informazione completa**: I giocatori conoscono le preferenze degli avversari.
- **Informazione incompleta**: I giocatori non conoscono le preferenze degli avversari.

Un ulteriore concetto è il **Mechanism design**: Progettare le regole del gioco per massimizzare i propri obiettivi. Esempio: aste.

### Giochi Coalizionali
   Sono giochi cooperativi dove gli agenti (giocatori) cooperano per ottenere benefici comuni, detti **worth**, in base alla coalizione di cui fanno parte. Gli agenti possono essere persone, aziende, partiti politici, ecc.

2. **Obiettivo**:  
   Calcolare una **solution concept**, ovvero come distribuire i benefici tra i giocatori in modo da soddisfare proprietà desiderabili (equità, razionalità).

3. **Struttura del Gioco**:
   - **Grand Coalition**: insieme totale dei giocatori.
   - **Worth Function (v(s))**: funzione che restituisce il beneficio di ogni sottoinsieme di giocatori.
   - **Ricchezza totale**: il massimo beneficio ottenibile dalla grand coalition, denotato come **v(N)**.

4. Il gioco risponde a delle **Domande Fondamentali**:
   - **Quale coalizione formare**?
   - **Come distribuire la ricchezza tra i membri della coalizione**?

5. Si basa su due **Assunzioni**:
   - **Transferable Utility (TU)**: la ricchezza può essere distribuita liberamente tra i membri (ad esempio, denaro).
   - **Non-Transferable Utility (NTU)**: se la ricchezza non è distribuibile in modo frazionato (ad esempio, oggetti indivisibili), la worth function non sarà un numero, ma un insieme di oggetti.

6. **Formalmente**:
   - Un gioco è una coppia **G = (N, v)** dove **v** è la worth function.
   - **Outcome**: un assegnamento di valori ai giocatori, definito dall'imputation set **X(G)**, che contiene tutti i vettori possibili **x = (x₁, ..., xₙ)**.

7. Un'imputazione, per essere considerata ammissibile, deve soddisfare le seguenti **Proprietà di una soluzione**:
   - **Efficiency**: la somma dei valori assegnati ai giocatori deve essere pari a **v(N)**.
   - **Individual Rationality**: il valore di ogni giocatore deve essere almeno quanto otterrebbe da solo, ovvero **xᵢ ≥ v({i})** per ogni giocatore **i**.

8. **Solution Concepts**:
   - **Fairness**: distribuzione equa della ricchezza.
   - **Stability**: nessun giocatore ha incentivi a lasciare la coalizione.

9. **Esempio**:
   - Giocatori in una rete connessa formano una coalizione.
   - La ricchezza della coalizione dipende dai costi degli archi nel grafo che li collega.
   - **Eccesso**: misura l'insoddisfazione di una coalizione. Ad esempio, l'eccesso della coalizione **S** è dato da **e(S, x) = v(S) - x(S)**.

### Caratteristiche dei Giochi Coalizionali

1. **Giochi Superadditivi**:
   - Se due coalizioni **S** e **T** (che non hanno membri in comune) si uniscono, il loro **worth** combinato è maggiore o uguale alla somma dei loro **worth** individuali.
   - Formula:  
     ∀S, T ⊂ N, se S ∧ T = ∅, allora v(S ∨ T) ≥ v(S) + v(T)  
     Significa che conviene collaborare, poiché la somma delle coalizioni è sempre maggiore o uguale alla somma delle singole worth. La **grand coalition** ha sempre il payoff massimo.

2. **Giochi Additivi**:
   - Non c'è alcun vantaggio aggiuntivo nel formare una coalizione: la worth combinata di due coalizioni è esattamente la somma delle loro worth individuali.
   - Formula:  
     ∀S, T ⊂ N, se S ∧ T = ∅, allora v(S ∨ T) = v(S) + v(T)  
     Qui, formare una coalizione non apporta alcun valore aggiunto rispetto alla somma dei singoli benefici.

3. **Giochi a Somma Costante**:
   - La somma della **worth** di una coalizione **S** e quella dei giocatori fuori dalla coalizione **N \ S** è sempre pari alla worth totale **v(N)**.
   - Formula:  
     ∀S ⊂ N, allora v(S) + v(N\S) = v(N)  
     Questo implica che non importa come si distribuisce la ricchezza tra le coalizioni e i non membri, la somma sarà sempre costante.

4. **Giochi Convessi**:
   - La ricchezza combinata di due coalizioni è maggiore o uguale alla somma delle loro worth individuali meno la worth della loro intersezione.
   - Formula:  
     ∀S, T ⊂ N, v(S ∨ T) ≥ v(S) + v(T) − v(S ∧ T)  
     Questo tipo di gioco incentiva la collaborazione, perché formare coalizioni conviene più che agire singolarmente o in piccoli gruppi.

5. **Giochi Semplici**:
   - La worth è binaria, ovvero può essere solo 0 o 1. Un esempio tipico di questi giochi sono i sistemi di voto, dove una coalizione "vince" (worth = 1) o "perde" (worth = 0).
   - Formula:  
     ∀S ⊂ N, v(S) ∈ {0, 1}  
     Quando un gioco semplice è anche a somma costante, si parla di **proper simple games**, dove la somma delle worth delle coalizioni e dei non membri è costante.

## Esempio
Questo esempio riguarda la distribuzione della **ricchezza** (o "worth") in un gioco coalizionale tra tre giocatori \(A\), \(B\) e \(C\). Cerchiamo di spiegarlo passo per passo:

### Dati iniziali
- Abbiamo tre giocatori: \(A\), \(B\), e \(C\).
- La **worth** di ogni singolo giocatore (da solo) è **0**:
  - \(v(A) = 0\)
  - \(v(B) = 0\)
  - \(v(C) = 0\)

- La **worth** delle coalizioni formate da due giocatori è la seguente:
  - \(v(\{A, B\}) = 20\)
  - \(v(\{A, C\}) = 30\)
  - \(v(\{B, C\}) = 40\)

- La **worth** della coalizione **grand coalition** (tutti e tre i giocatori insieme) è \(v(\{A, B, C\}) = 42\).

### Obiettivo: Distribuire la ricchezza
Il nostro obiettivo è distribuire la **worth totale** \(v(\{A, B, C\}) = 42\) tra i tre giocatori \(A\), \(B\), e \(C\), rispettando alcune condizioni.

### Imputazione iniziale: \(x = [4, 14, 24]\)
In questo esempio, viene data una **imputazione** (una possibile distribuzione della ricchezza): \(x = [4, 14, 24]\), cioè:
- \(A\) riceve **4**,
- \(B\) riceve **14**,
- \(C\) riceve **24**.

#### Condizioni da rispettare:
1. **Efficienza**:
   - La somma delle ricchezze assegnate deve essere uguale alla **worth totale**. In questo caso:
     \[4 + 14 + 24 = 42\]
     La condizione di **efficienza** è soddisfatta, perché la somma è 42.

2. **Razionale individualmente**:
   - Ogni giocatore deve ricevere almeno quanto otterrebbe se giocasse da solo, cioè il suo valore individuale \(v(\{i\})\). Poiché \(v(A) = v(B) = v(C) = 0\), tutti i giocatori ricevono più di **0**, quindi la condizione di **razionalità individuale** è soddisfatta.

### Problema: Insoddisfazione delle coalizioni
Nonostante l'imputazione sia efficiente e razionalmente valida, le **coalizioni** non sono soddisfatte. Ad esempio:
- La coalizione \(\{A, B\}\) riceve \(4 + 14 = 18\), ma la loro worth è \(20\), quindi sono insoddisfatti.
- La coalizione \(\{A, C\}\) riceve \(4 + 24 = 28\), ma la loro worth è \(30\), quindi sono insoddisfatti.
- La coalizione \(\{B, C\}\) riceve \(14 + 24 = 38\), ma la loro worth è \(40\), quindi sono insoddisfatti.

### Tentativo di risolvere con un sistema lineare
Per cercare una distribuzione equa, possiamo impostare un **sistema di disequazioni** basato sulle coalizioni:

1. (x_A + x_B ≥20\) (la coalizione \(\{A, B\}\) deve ricevere almeno 20)
2. (x_B + x_C ≥ 40\) (la coalizione \(\{B, C\}\) deve ricevere almeno 40)
3. (x_A + x_C ≥ 30\) (la coalizione \(\{A, C\}\) deve ricevere almeno 30)
4. (x_A + x_B + x_C = 42\) (la somma totale deve essere uguale a 42)

### Contraddizione
Semplificando queste disequazioni, otteniamo:

- (x_A + x_B + x_C ≥ 45\)
  
Questa disequazione va in **conflitto** con l'equazione (x_A + x_B + x_C = 42\), perché non è possibile avere contemporaneamente una somma che sia uguale a 42 e maggiore o uguale a 45.

### Conclusione
Non esiste una **soluzione** che soddisfi contemporaneamente tutte le coalizioni e rispetti la condizione di efficienza.

### Caso alternativo: \(x = [5, 15, 25]\)
Se invece la **worth totale** fosse \(45\) (cioè \(x_A + x_B + x_C = 45\)), un'imputazione come \(x = [5, 15, 25]\) renderebbe tutti contenti, poiché:
- \(\{A, B\}\) riceve \(5 + 15 = 20\), che soddisfa la loro worth.
- \(\{A, C\}\) riceve \(5 + 25 = 30\), che soddisfa la loro worth.
- \(\{B, C\}\) riceve \(15 + 25 = 40\), che soddisfa la loro worth.

Tuttavia, questo non è possibile nel contesto del problema, poiché la ricchezza totale è 42, non 45.

## Core di un Gioco Coalizionale
Il **core** è un concetto fondamentale nella teoria dei giochi coalizionali e rappresenta l'insieme delle distribuzioni (o imputazioni) della **ricchezza** tra i giocatori che sono **stabili**, ovvero tali che nessuna coalizione abbia incentivi a "rompere" l'accordo e formare una coalizione separata.

#### Definizione Formale
Dato un gioco coalizionale \( G = (N, v) \), dove:
- ( N \) è l'insieme dei giocatori,
- v(S) \ è la **worth** di una coalizione \ S \⊆ N \, ovvero il valore che quella coalizione può ottenere collaborando,
il **core** è l'insieme delle imputazioni \ x \∈ X(G) tali che:
$$[
x(S) \geq v(S) \quad \forall S \subseteq N
]$$
Ovvero, il valore assegnato a una coalizione \S \ (la somma delle ricchezze dei giocatori nella coalizione \S \) deve essere **almeno** pari alla worth  v(S) \ di quella coalizione.

Formalmente, per ogni imputazione \ x \∈ X(G)\ (dove \ X(G) \ è l'insieme delle imputazioni possibili):
$$[
x(S) = \sum_{i \in S} x_i
]$$
e il core è l'insieme di tutte le imputazioni \( x \) tali che:
$$[
v(S) \leq x(S) \quad \forall S \subseteq N \quad \text{e} \quad x(N) = v(N)
]$$
dove l'ultima condizione \( x(N) = v(N) \) garantisce che tutta la ricchezza \ v(N) \ della **grand coalition** (l'insieme di tutti i giocatori) sia distribuita tra i giocatori.

#### Stabilità del Core
Il core è importante perché garantisce la **stabilità** della coalizione. Se scegliamo una distribuzione che appartiene al core, allora:
- Nessuna coalizione ha un incentivo a staccarsi e formare una coalizione separata, perché ogni coalizione \( S \) ottiene **almeno** quanto otterrebbe formando una coalizione indipendente.
- Ogni giocatore riceve almeno il proprio contributo individuale.

Tuttavia, il core può essere **vuoto**, e, anche se non è vuoto, può contenere molte possibili soluzioni.

### Proprietà delle Soluzioni nel Core

#### 1. **Simmetry Axiom (Assioma di Simmetria)**
Se due giocatori sono **interscambiabili**, cioè hanno lo stesso ruolo e contribuiscono allo stesso modo alla coalizione, allora dovrebbero ricevere la stessa **worth**. Formalmente:
$$[
\forall S \subseteq N, \text{ tale che } i, j \notin S, \quad v(S \cup \{i\}) = v(S \cup \{j\})
]$$
Questo significa che, se la worth di una coalizione non cambia scambiando i giocatori \( i \) e \( j \), allora questi due giocatori dovrebbero ricevere la stessa ricompensa.

#### 2. **Dummy Players (Giocatori Nulli)**
Un **giocatore nullo** è un giocatore il cui contributo marginale a qualsiasi coalizione è sempre uguale al valore che otterrebbe da solo. Formalmente, per ogni \( S \subseteq N \) tale che \( i \notin S \):
$$[
v(S \cup \{i\}) - v(S) = v(\{i\})
]$$
Questo significa che, se il contributo di un giocatore \( i \) a una coalizione \( S \) è pari a ciò che otterrebbe giocando da solo, quel giocatore dovrebbe ricevere esattamente quella quantità.

#### 3. **Additivity (Additività)**
Se abbiamo due giochi \( G₁ = (N, v₁ \) e \( G₂ = (N, v₂) \) con la stessa serie di giocatori \N , possiamo combinare i due giochi sommandone le worth function:
$$[
(v_1 + v_2)(S) = v_1(S) + v_2(S) \quad \forall S \subseteq N
]$$
Una soluzione \( \Psi \) è **additiva** se la worth assegnata ai giocatori nella somma dei giochi è uguale alla somma delle worth assegnate nei giochi separati:
$$[
\Psi_i(N, v_1 + v_2) = \Psi_i(N, v_1) + \Psi_i(N, v_2)
]$$
In altre parole, la soluzione per due giochi combinati è la somma delle soluzioni dei due giochi separati.

### Shapley Value
 Lo **Shapley Value** è una soluzione importante nella teoria dei giochi coalizionali, che assegna a ogni giocatore una quota della ricchezza totale, basata sui suoi **contributi marginali** a tutte le possibili coalizioni. Viene considerato una soluzione "meritocratica", poiché si basa su quanto un giocatore contribuisce a una coalizione rispetto a quanto la coalizione ottiene senza di lui.
- Lo Shapley Value è l'unica **pre-imputazione** (una distribuzione che non deve necessariamente rispettare la razionalità individuale) che soddisfa le 3 proprietà elencate prima.
- Lo Shapley Value esiste **sempre** in qualsiasi gioco, anche se non sempre rispetta le condizioni di razionalità individuale.
#### Caratteristiche chiave:
- **Equità**: Assicura che ogni giocatore riceva una quota proporzionale al suo contributo.
- **Stabilità**: È una soluzione che può essere sempre trovata.
- **Meritocrazia**: Si basa sui contributi effettivi che un giocatore fornisce alle coalizioni.

#### Formula dello Shapley Value
Per calcolare lo Shapley Value per un giocatore \ i \, si tiene conto del contributo marginale che  i apporta a tutte le possibili coalizioni \( S \⊆ N \) (dove \i \ non fa parte di  S ).

La formula è la seguente:

$$[
\phi_i(N, v) = \frac{1}{|N|!} \sum_{S \subseteq N \setminus \{i\}} |S|! \cdot (|N| - |S| - 1)! \cdot [v(S \cup \{i\}) - v(S)]
]$$

Dove:
- \( N \ è l'insieme di tutti i giocatori.
- \( v(S) \ è la worth (il valore) della coalizione \( S \).
- \( |N|! \ è il numero di permutazioni dei giocatori.
- \( |S|! \ è il numero di modi per ordinare i giocatori in \( S \).
- \( (|N| - |S| - 1)! \ è il numero di modi per ordinare i giocatori rimanenti dopo aver aggiunto \i alla coalizione \S .

#### Interpretazione
Lo Shapley Value può essere interpretato come una **media pesata** dei contributi marginali del giocatore \( i \) in ogni possibile coalizione \( S \) che non contiene \( i \). In altre parole, si calcola il contributo di \( i \) a ogni coalizione, misurato come la differenza tra la worth della coalizione con \( i \) e la worth della coalizione senza \( i \), e poi si fa una media considerando **tutti i possibili ordinamenti** in cui \( i \) può essere aggiunto alla coalizione.

#### Esempio
Immaginiamo tre giocatori \( A \), \( B \), e \( C \). Se \( A \), \( B \), e \( C \) formano coalizioni diverse, lo Shapley Value di un giocatore è la media dei suoi contributi marginali calcolati su tutte le coalizioni possibili. Per esempio, se aggiungere \( A \) a una coalizione \( S \) aumenta la worth di \( S \), il contributo marginale di \( A \) in quella coalizione sarà  *v(S \∪ \{A\}) - v(S)* 


## Nucleolo in un gioco coalizionale

Il **nucleolo** è un concetto di soluzione per i giochi coalizionali, che si concentra sulla riduzione dell'insoddisfazione massima tra i giocatori. L'idea alla base del nucleolo è quella di minimizzare l'insoddisfazione, o **eccesso** (excess), delle coalizioni in modo gerarchico: prima si riduce l'insoddisfazione massima, poi quella successiva e così via, fino ad ottenere una distribuzione delle risorse più equa e stabile possibile.

#### Definizione di **Eccesso** e **Vettore degli Eccessi**
- **Eccesso** e(x, S): L'eccesso di una coalizione  S  rispetto a una imputazione  x è la differenza tra la worth  v(S)  della coalizione S  e la somma dei valori assegnati ai giocatori in \S  dalla distribuzione \x :
  
 $$ 
  e(x, S) = v(S) - \sum_{i \in S} x_i
  $$

  Questo misura quanto la coalizione \( S \) si sente insoddisfatta rispetto alla distribuzione \( x \).

- **Vettore degli Eccessi**  θ(x) : È un vettore che contiene gli eccessi e(x, S)  per tutte le coalizioni S $\subseteq$ N \, ordinati in modo **non crescente** (dal più grande al più piccolo). Questo vettore permette di confrontare le imputazioni in base all'insoddisfazione che creano nelle coalizioni.

#### Definizione del **Nucleolo**
Il **nucleolo** è l'imputazione  x \) che minimizza lessicograficamente il vettore degli eccessi $( \Theta(x) )$. In altre parole, il nucleolo è la soluzione in cui la massima insoddisfazione è minimizzata, e in caso di parità su questa, si minimizza l'insoddisfazione successiva, e così via.

Formalmente, il **nucleolo**  N(G) di un gioco G = (N, v)  è l'insieme delle imputazioni x tali che:

$$
N(G) = \{x \in X(G) : \nexists y \text{ tale che } \Theta(y) \succ \Theta(x)\}
$$

Dove $\Theta(y) \succ \Theta(x)$ significa che il vettore degli eccessi di  y  è preferibile (cioè ha meno insoddisfazione) rispetto a quello di x .

#### Proprietà del Nucleolo
- **Unicità**: Il nucleolo è costituito da un unico vettore \( x \), a meno che i valori non siano interi. Questo significa che esiste una sola distribuzione delle risorse che minimizza lessicograficamente l'insoddisfazione.

#### Relazione con l’**ε-core**
Il concetto di **ε-core** è una generalizzazione più debole del core, dove si permette una certa insoddisfazione, regolata dal parametro $\epsilon$.  L'**ε-core** è l'insieme di imputazioni x  tali che:

$$\sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \subseteq N
$$
All'aumentare di $\epsilon$ \, l'**ε-core** si espande e aumenta la probabilità di trovare una soluzione.

#### Iterazione verso il Nucleolo
Il processo per trovare il nucleolo può essere visto come una procedura iterativa. Si parte risolvendo il problema di minimizzare  $\epsilon$  in un sistema lineare, che descrive il **least core** (la versione più forte dell'ε-core). Una volta ottenuto un valore ottimale $( \epsilon_1 )$, si identifica un insieme di **coalizioni critiche** \( $S_i$ \), che sono quelle coalizioni per cui l'eccesso è esattamente pari a \( $\epsilon_1$ \).

Il passo successivo è ripetere il processo considerando solo le coalizioni non critiche, minimizzando di nuovo  $\epsilon$ , e così via, finché non si converge al **nucleolo**.

Il problema lineare al passo \( 1 \) è:

$$
\min \epsilon
$$
$$
\text{soggetto a: } \sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \subseteq N
$$
$$
x(N) = v(N)
$$


Calcoliamo il valore ottimo, che chiamiamo  $\epsilon_1$ . È possibile che esistano delle coalizioni che soddisfano i vincoli di eguaglianza, le cosiddette *coalizioni critiche* $S_i$, definite come segue:

$x^*(S_i) = v(S_i) - \epsilon_1 \quad$

dove $x^*(S_i)$ è la soluzione ottima per la coalizione i .

A questo punto, definendo con F₁ l'insieme delle coalizioni critiche ottenute al passo 1 (quello che abbiamo appena esaminato), possiamo rendere ancora più forte il least core iterando il procedimento. Questo significa risolvere il seguente problema lineare:

$$
\begin{array}{rl}
\text{minimize} & \epsilon \\
\text{subject to} & \sum_{i \in S} x_i \geq v(S) - \epsilon, \quad \forall S \in 2^N - F_1 \\
                  & x(S) = v(S) - \epsilon_1, \quad \forall S \in F_1 \\
                  & x(N) = v(N)
\end{array} \quad (8.24)

$$
Questo procedimento si ripete, restringendo sempre di più l'insieme delle coalizioni critiche, fino a ottenere la soluzione finale, che rappresenta il nucleolo.

---

La **Contested Garment Rule** è un criterio per dividere un'eredità tra due creditori con debiti d₁ e $d_2 (dove (d_1 \leq d_2))$ e un totale da dividere \(e\) con $(0 \leq e \leq d)$:

1. Il creditore 1 può concedere al creditore 2 un importo massimo di $\max(e - d_1, 0) .$
2. Il creditore 2 può concedere al creditore 1 un importo massimo di $\max(e - d_2, 0) .$
3. L'importo rimanente, cioè $( e - \max(e - d_1, 0) - \max(e - d_2, 0) )$, viene diviso equamente tra i due creditori.

Questa regola combina linearmente il criterio di divisione eguale dei guadagni e delle perdite. È stato dimostrato che esiste una sola divisione dell'eredità che segue questa regola e che tale divisione corrisponde esattamente al nucleolo del gioco associato, dove ogni coalizione pretende il massimo tra 0 e ciò che le è concesso da chi non fa parte della coalizione.

---

## **Aste e Mechanism Design** 
Si occupano di stabilire regole per le aste in modo da incentivare comportamenti strategici da parte degli agenti. Esistono due principali categorie di aste:

1. **Single good**: offerte su un unico oggetto.
2. **Multiple goods**: offerte su insiemi di oggetti, tipiche delle *combinatorial auctions*.

Ci concentriamo sulle aste *single good*, in particolare:

- **English auction**: i partecipanti fanno offerte crescenti e il bene viene assegnato al maggiore offerente. L'obiettivo è vendere al prezzo più alto possibile. L'utilità di un partecipante  i è $u_i = v_i - S_i$ se vince, altrimenti è 0, dove $v_i$ è il valore segreto che attribuisce all'oggetto e $\( S_i \)$ l'offerta fatta.
  
- **Japanese auction**: simile a quella inglese, ma con il gestore che fissa progressivamente le offerte. I giocatori scelgono se uscire o continuare fino a quando rimane un solo vincitore.
  
- **Dutch auction**: un'asta al ribasso in cui si parte da un prezzo alto e si scende. Vince chi fa la prima offerta.

Il **Mechanism Design** mira a creare regole che inducano gli agenti a seguire strategie ottimali, garantendo un *mechanism design truthful*, ovvero rendendo conveniente rivelare le proprie vere preferenze o valori.

### **Sealed-Bid Auctions (Aste a busta chiusa)**
In questo tipo di asta, le offerte sono segrete e il vincitore è colui che offre la cifra più alta.
Gli agenti devono basarsi su:
  - Il valore che attribuiscono al bene.
  - Il numero di partecipanti.
  - Le caratteristiche dei partecipanti (budget, tendenze nelle offerte).
- Potrebbero avere informazioni sulle distribuzioni di probabilità che descrivono come gli altri agenti valutano il bene.

**Distribuzioni IPV (Independent Private Values)**
- Ogni partecipante ha un valore attribuito al bene indipendente da quello degli altri.
- Ciascun giocatore conosce la distribuzione di probabilità da cui vengono estratti i valori degli altri partecipanti, ma non conosce i valori esatti.
Dunque, nelle aste a busta chiusa, la strategia di un partecipante può essere influenzata dalla distribuzione di probabilità delle valutazioni altrui, rendendo complessa la scelta di una strategia ottimale.
---
### **Second-Price Auctions (Aste al secondo prezzo)**
 Ogni partecipante fa la propria offerta e vince chi ha offerto di più, ma paga il secondo prezzo più alto, non la propria offerta.
- **Mechanism Design truthful**: l'asta al secondo prezzo incoraggia i partecipanti a fare offerte corrispondenti al reale valore che attribuiscono al bene. Questa strategia è ottimale indipendentemente dalle strategie degli altri.

**CASO 1: Gli altri agenti offrono meno del valore $v_i$ che attribuisco al bene**
- Se offro$s_i < v_i$, rischio di perdere e ottenere utilità $u_i = 0$.
- Se offro $s_i = v_i$, vinco e la mia utilità è $u_i = v_i - s_{\text{max}}$ dove $s_{\text{max}}$ è l’offerta più alta tra gli altri agenti).
- Se offro $s_i > v_i$, vinco, ma la mia utilità resta $u_i = v_i - s_{\text{max}}$, identica al caso precedente.

**Conclusione**: Offrire  $s_i = v_i$ è la scelta ottimale. Offrire più o meno del valore che attribuisco al bene non cambia l'utilità, quindi mi conviene essere veritiero.

---

**CASO 2: Altri agenti offrono più del valore \( v_i \)**

- Se \( $s_{\text{max}} > v_i$ \):
  - Se offro  $s_i < v_i$   o  $s_i = v_i$, perdo e guadagno \( $u_i = 0$ \).
  - Se offro \( $s_i > v_i$ \), potrei vincere ma la mia utilità sarebbe negativa \( $u_i = v_i - s_{\text{max}} < 0$ \).

**Conclusione**: Offrire più del valore che attribuisco al bene mi può portare a un guadagno negativo. Poiché non posso prevedere esattamente come si comporteranno gli altri, la strategia migliore resta offrire $s_i = v_i$, cioè il valore esatto che attribuisco al bene.

- Il meccanismo dell’asta giapponese si comporta in modo simile all'asta al secondo prezzo: anch’esso è **truthful** e incoraggia i partecipanti a fare offerte veritiere.

### First-Price Auctions: Aste al Primo Prezzo

Nelle **aste al primo prezzo**, i partecipanti presentano offerte segrete e chi offre di più vince, ma deve pagare l'importo offerto (non il secondo prezzo, come nelle aste di Vickrey). 

#### Punti Chiave:
1. **Offerte a busta chiusa**: I partecipanti non sanno l'offerta degli altri.
2. **Valore stimato del bene**: Ogni giocatore ha un proprio valore \( v_i \), che rappresenta quanto è disposto a pagare per l'oggetto.
3. **Strategia ottimale**: I partecipanti devono considerare le offerte degli altri e ragionare su come massimizzare il proprio guadagno.
   
#### Esempio con 2 Giocatori:
- **Assunzioni**:
  - Giocatore 2 offre metà del valore che attribuisce al bene: \( $s_2 = \frac{1}{2} v_2$ \).
  - I valori sono distribuiti uniformemente tra 0 e 1.
  
- **Domanda**: Qual è la miglior strategia per il giocatore 1?
  
### Calcolo del Valore Atteso dell'Utilità

1. **Utilità del Giocatore 1** \( $u_1$ \): Guadagno che ottiene se vince, meno l'offerta fatta \( $s_1$ \).
$$   
   E[u_1] = \int_0^{2s_1} u_1 dv_2
   $$
   Questo perché se l'offerta del giocatore 2 \( $s_2 = \frac{1}{2} v_2$ \) è inferiore a \( $2s_1$ \), il giocatore 1 vince.

2. **Condizioni al contorno**:
   - Se \( $v_2 > 2s_1$ \), il giocatore 2 offre più di \( $s_1$ \), quindi l'utilità del giocatore 1 è zero (secondo integrale).
  
3. **Calcolo dell'integrale**:
   $$
   E[u_1] = \int_0^{2s_1} (v_1 - s_1) dv_2 = 2v_1s_1 - 2s_1^2
   $$

### Massimizzazione dell'Utilità
Per trovare l'**offerta ottimale** \( $s_1$ \), si calcola la derivata dell'utilità attesa rispetto a \( $s_1$ \):
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
Esiste un **teorema generale** che afferma che in un'asta al primo prezzo con $ n$ agenti **risk-neutral** (neutri al rischio) e distribuzioni uniformi, l'unico **equilibrio simmetrico** ha la forma:
$$
\left( \frac{n-1}{n} v_1, \frac{n-1}{n} v_2, \dots, \frac{n-1}{n} v_n \right)
$$
Questo significa che ogni giocatore offre una frazione $\frac{n-1}{n}$ del proprio valore.

### Riassunto
- In un'asta al primo prezzo, la strategia ottimale per ciascun giocatore è offrire una frazione del proprio valore.
- In un contesto con 2 giocatori, questa frazione è la metà del valore stimato.
- Per $n$ giocatori, la frazione ottimale è $\frac{n-1}{n}$.
---
### Giochi Strategici e Dilemma del Prigioniero

#### Giochi Strategici
- **Definizione**: Giochi in cui ogni giocatore agisce per il proprio interesse individuale, senza considerare coalizioni o accordi tra giocatori.
- **Obiettivo**: Massimizzare il proprio profitto, spesso in presenza di obiettivi contrastanti tra i giocatori.

#### Dilemma del Prigioniero
- **Descrizione**: Due prigionieri devono decidere se confessare un crimine o non confessare. La loro decisione influenza la pena che riceveranno.
- Il Dilemma del Prigioniero è un esempio classico dove entrambi i giocatori finiscono in una situazione subottimale (confessando) anche se avrebbero ottenuto un risultato migliore cooperando (non confessando).
Consideriamo una tabella dei payoff 

| **Prigioniero 2** \ **Prigioniero 1** | **Non Confessa** | **Confessa** |
| ------------------------------------- | ---------------- | ------------ |
| **Non Confessa**                      | (-1, -1)         | (-4, 0.4)    |
| **Confessa**                          | (0.4, -4)        | (-0.3, -0.3) |
#### Spiegazione dei Payoff:
- **(Non Confessa, Non Confessa)**: Entrambi ottengono (-1, -1). Entrambi ricevono una pena minore perché hanno collaborato e non hanno confessato.
- **(Non Confessa, Confessa)**: Il prigioniero che non confessa riceve -4, mentre quello che confessa riceve 0.4. Il prigioniero che confessa beneficia di una riduzione della pena, mentre l'altro subisce una pena più grave.
- **(Confessa, Non Confessa)**: Il prigioniero che confessa riceve 0.4, mentre quello che non confessa riceve -4. Il prigioniero che confessa ottiene una ricompensa, mentre l'altro riceve una pena pesante.
- **(Confessa, Confessa)**: Entrambi ricevono (-0.3, -0.3). Entrambi confessano, quindi ricevono una pena minore rispetto a se uno solo avesse confessato.

Se il **Prigioniero 1** decide di non confessare e il **Prigioniero 2** confessa, i payoff saranno:
- **Prigioniero 1**: -4
- **Prigioniero 2**: 0.4

Questo significa che se il Prigioniero 2 confessa mentre il Prigioniero 1 non lo fa, il Prigioniero 1 subisce una pena più severa (-4) mentre il Prigioniero 2 riceve una pena ridotta (0.4).

- **Equilibrio di Nash**:
  - **Situazione**: Entrambi i prigionieri confessano.
  - **Motivazione**: Anche se non confessare sarebbe più vantaggioso se l’altro non confessa, confessare diventa la scelta migliore se si sospetta che l’altro confessi.
  - **Risultato**: L’equilibrio di Nash si verifica quando entrambi i giocatori scelgono di confessare, poiché nessuno ha incentivo a deviare dalla propria scelta data la scelta dell’altro.

### Equilibrio di Nash

#### Definizione Formale di Gioco Strategico
1. **Giocatori**: Un insieme di $N$ giocatori.
2. **Azioni/Strategie**:
   - Per ogni giocatore $i \in N$, esiste un insieme di azioni ammissibili $S_i$.
   - Le azioni di ciascun giocatore sono indicate con $s_i$.
3. **Profili di Azione**:
   - L'insieme dei profili di azione è $S = \times_{j \in N} S_j = S_1 \times \cdots \times S_N$.
   - Un profilo di azione è una $N$-upla di azioni ammissibili $(s_1, ..., s_N)\).
4. **Utilità**:
   - Per ogni giocatore $i \in N$, esiste una funzione di utilità $u_i : S \rightarrow \mathbb{R}$ che assegna un valore reale a ciascun profilo di azione.

#### Definizione di Equilibrio di Nash
Un profilo di azione $S^*$ è un equilibrio di Nash se:
$$
u_i(S^*) \geq u_i(s_i, S^*_{-i}) \quad \forall i \in N, \quad \forall s_i \in S_i
$$
dove $S^*_{-i}$ rappresenta le scelte degli altri giocatori. In altre parole, dato un profilo di azione $S^*$, nessun giocatore può migliorare la propria utilità cambiando unilateralmente la propria azione, dato che le azioni degli altri giocatori rimangono costanti.

#### Esempio: Gioco dei Bach e Stravinsky
Due persone devono decidere a quale concerto andare (Bach o Stravinsky). Le preferenze e i payoff sono i seguenti:

| **Giocatore 2** \ **Giocatore 1** | **Bach** | **Stravinsky** |
|----------------------------------|----------|----------------|
| **Bach**                         | (2.1, 2.1) | (0, 0)         |
| **Stravinsky**                   | (0, 0)    | (1, 1)         |

- **Payoff**:
  - Se entrambi vanno a **Bach**, ottengono (2.1, 2.1).
  - Se entrambi vanno a **Stravinsky**, ottengono (1, 1).
  - Se uno va a **Bach** e l'altro a **Stravinsky**, entrambi ricevono (0, 0).

**Equilibri di Nash**:
- (Bach, Bach) è un equilibrio di Nash: Se entrambi vanno a Bach, nessuno ha incentivo a cambiare, dato che preferiscono stare insieme piuttosto che andare separatamente.
- (Stravinsky, Stravinsky) è un altro equilibrio di Nash per lo stesso motivo.

#### Funzione di Best Response
La funzione di best response per un giocatore $i$ dato un profilo di azioni degli altri giocatori $s_{-i}$ è:
$$
B_i(s_{-i}) = \{s_i \in S_i : u_i(s_i, s_{-i}) \geq u_i(s'_i, s_{-i}), \forall s'_i \in S_i\}
$$
Un profilo di azione $s^*$ è un equilibrio di Nash se e solo se ogni azione $s^*_i$ è una best response alle azioni degli altri giocatori, ovvero:
$$
s^*_i \in B_i(s^*_{-i}) \quad \forall i \in N
$$
