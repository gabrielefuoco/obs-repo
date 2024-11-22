### Capitolo 1

### Calcolo Proposizionale

Nel calcolo proposizionale scriviamo delle formule logiche: una formula/asserzione il cui
valore di verità può essere vero o falso. Per scrivere le formule si utilizzano dei simboli:

consideriamo il seguente esempio.

P = "piove", R = "fa freddo"

N = "ci sono nuvole", V = "c'è vento"

Ne = "nevica", C = "ci si copre"

Piove e fa molto freddo

Fa freddo, ma non piove

Se ci sono nuvole e non c'è vento, allora piove

Piove solo se ci sono nuvole e non c'è vento

Nevica, ma non fa freddo: se ci si copre

Se ci si copre, allora fa freddo o nevica

Introduciamo qualche concetto:

*   Gli identificatori sono gli elementi base del nostro alfabeto (ad es: P="piove").
*   Un atomo è un identificatore, un atomo predefinito (T/F) oppure può essere una
    proposizione, che contiene altre proposizioni/atomi combinati mediante degli operatori.
*   Gli operatori possono essere ∧, ∨, ⇒, ⇐, ¬, =, ≠. Per limitare l'uso delle
    parentesi, si assume che i vari operatori abbiano una determinata precedenza:

| operatore | livello di precedenza (crescente) |

Si noti che un'interpretazione è un assegnamento di valori di verità ai predicati.

---

**CALCOLO PROPOSIZIONALE**

Si noti che l'equivalenza (=) è differente dall'uguaglianza (=): l'uguaglianza, infatti, ci dice che il valore associato ai due termini è uguale. L'equivalenza può essere identificata anche come (⇔). Il significato può essere definito utilizzando le tabelle di verità: infatti, ad ogni operatore associamo una tabella di verità. Ad ogni operatore associamo una tabella di verità.

**Cosa possiamo dire sulle formule?**

*   Una formula è una **tautologia** se, per qualunque valore assegniamo alle variabili, il risultato è sempre vero.
*   Una formula è una **contraddizione** se, per qualunque valore venga assegnato alle variabili, è sempre falsa.
*   Quindi, $P$ è una tautologia se e solo se $\neg P$ è una contraddizione.

**Possiamo dire che:**

*   $P$ implica tautologicamente $Q$ se $P \Rightarrow Q$ è una tautologia.
*   $P$ è tautologicamente equivalente a $Q$ se $P \Leftrightarrow Q$ è una tautologia.

**Per dimostrare che una formula è una tautologia possiamo:**

*   Usare le **tabelle di verità**: consideriamo $2^n$ casi dove $n$ è il numero di variabili proposizionali. Se esistono righe sia false che vere allora la formula è soddisfacibile.

| $P$ | $Q$ | $R$ | $(P \wedge Q) \Rightarrow R$ |
| --- | --- | --- | --- |
| T | T | T | T |
| T | T | F | F |
| T | F | T | T |
| T | F | F | T |
| F | T | T | T |
| F | T | F | T |
| F | F | T | T |
| F | F | F | T |

**Si può costruire una dimostrazione usando le leggi della logica dimostrate e le regole di inferenza.**

**Chiaramente, per mostrare che una formula non è una tautologia è sufficiente individuare una sostituzione che rende la formula falsa.**

---

**CALCOLO PROPOSIZIONALE**

A questo punto, passiamo in rassegna le varie leggi:

### Leggi di Equivalenza

*   **Riflessività**
    *   $p=p$
    *   $(p \equiv q) \equiv (q \equiv p)$
    *   $(p \equiv q) \equiv (p \equiv (q \equiv q))$
    *   $(p \equiv T) \equiv (p \equiv (p \equiv p))$
    *   $(p \equiv q) \wedge (q \equiv r) \equiv (p \equiv r)$
*   **Simmetria**
    *   $(p \equiv q) \equiv (q \equiv p)$
*   **Associatività**
    *   $(p \equiv (q \equiv r)) \equiv ((p \equiv q) \equiv r)$
*   **Transitività**
    *   $(p \equiv q) \wedge (q \equiv r) \equiv (p \equiv r)$

### Leggi Congiunzione e Disgiunzione

*   **Commutatività**
    *   $p \vee q \equiv q \vee p$
    *   $p \wedge q \equiv q \wedge p$
*   **Associatività**
    *   $(p \vee q) \vee r \equiv p \vee (q \vee r)$
    *   $(p \wedge q) \wedge r \equiv p \wedge (q \wedge r)$
*   **Distributività**
    *   $p \vee (q \wedge r) \equiv (p \vee q) \wedge (p \vee r)$
    *   $p \wedge (q \vee r) \equiv (p \wedge q) \vee (p \wedge r)$
*   **Idempotenza**
    *   $p \vee p \equiv p$
    *   $p \wedge p \equiv p$
*   **Unità**
    *   $p \vee F \equiv p$
    *   $p \wedge T \equiv p$
*   **Zero**
    *   $p \vee T \equiv T$
    *   $p \wedge F \equiv F$
*   **Doppia Negazione**
    *   $\neg \neg p \equiv p$
*   **Terzo Escluso**
    *   $p \vee \neg p \equiv T$
*   **Contraddizione**
    *   $p \wedge \neg p \equiv F$

### Leggi della Negazione

*   **Doppia Negazione**
    *   $\neg \neg p \equiv p$
*   **Terzo Escluso**
    *   $p \vee \neg p \equiv T$
*   **Contraddizione**
    *   $p \wedge \neg p \equiv F$
*   **De Morgan**
    *   $\neg (p \wedge q) \equiv \neg p \vee \neg q$
    *   $\neg (p \vee q) \equiv \neg p \wedge \neg q$

### Leggi di Eliminazione

*   **Eliminazione**
    *   $(p \Rightarrow q) \equiv \neg p \vee q$
*   **Eliminazione Bis**
    *   $(p \Leftrightarrow q) \equiv (p \Rightarrow q) \wedge (q \Rightarrow p)$

---

### CALCOLO PROPOSIZIONALE

Altre leggi utili, che possono essere ricavate usando le altre, sono le seguenti:

$$
\begin{aligned}
P \vee (\neg A \wedge B) & = (P \vee \neg A) \wedge (P \vee B) \\
P \wedge (\neg A \vee B) & = (P \wedge \neg A) \vee (P \wedge B) \\
P \wedge (A \vee P) & = P \\
P \vee (A \wedge P) & = P \\
P \wedge Q & = P \\
P \vee Q & = P \vee Q \\
P \wedge Q & = Q \wedge P \\
P \vee Q & = Q \vee P \\
P \wedge (P \vee Q) & = P \\
P \vee (P \wedge Q) & = P \\
(P = Q) & = (P \wedge Q) \vee (\neg P \wedge \neg Q) \\
P \wedge Q & = (P \wedge Q) \vee (P \wedge \neg Q) \\
P \vee Q & = (P \vee Q) \vee (P \vee \neg Q) \\
(P = Q) & = (P \wedge Q) \vee (\neg P \wedge \neg Q) \\
P \wedge Q & = P \wedge Q \\
P \vee Q & = P \vee Q \\
\end{aligned}
$$

### 1.1 Dimostrazione di Tautologie - Regole di Inferenza

#### 1.1.1 Principio di Sostituzione

Il principio di sostituzione esprime una proprietà fondamentale dell'uguaglianza: se sappiamo che $A = B$, allora il valore di una espressione $C'$ in cui compare $A$ non cambia se $A$ è sostituito con $B$. In altri termini:

$$
A = B \Rightarrow C = C\{A/B\}
$$

dove $A = B$ è una legge e $C = C\{A/B\}$ è l'uguaglianza da essa giustificata, grazie al principio di sostituzione.

Nel calcolo proposizionale, esprime una proprietà dell'equivalenza:

$$
A = B \Rightarrow C = C\{A/B\} \tag{1.2}
$$

Una tautologia è detta legge se descrive una proprietà di uno o più connettivi logici o se essa è usata come una giustificazione nelle dimostrazioni.

---

**CALCOLO PROPOSIZIONALE**

**1.1.2 Dimostrazioni di Equivalenze Tautologiche**

Il nostro obiettivo solitamente è individuare equivalenze tautologiche. $P_{1} = P_{2}$. Nella pratica individuiamo delle equivalenze successive: $P_{1} = P_{2} = \cdots = P_{n}$. Quindi, data una equivalenza che vogliamo dimostrare essere tautologica, siccome essa potrebbe avere una tabella di verità enorme, possiamo effettuare delle trasformazioni.

$$
\begin{aligned}
P_{1} & = \{ \text{giustificazione} \} \\
& = P_{2} \\
& = \{ \text{giustificazione} \} \\
& \vdots \\
& = P_{n}
\end{aligned}
$$

Dove ogni passo, supponendo che $P = Q$ sia una legge, ha la seguente forma:

$$
\begin{aligned}
P & = \{ \text{giustificazione} \} \\
& = Q
\end{aligned}
$$

Ciascun passo, siccome vale il principio di sostituzione, è corretto. Si noti che il calcolo proposizionale è decidibile: data una formula $\delta$ (ossia sempre stabilire se è una tautologia, contraddizione o soddisfacibile). Inoltre, l'insieme $\{ \neg, \wedge \}$ è funzionalmente completo: il posiamo scrivere qualsiasi formula, applicando le opportune trasformazioni... si usano anche gli altri per questioni di semplicità ed efficienza nel calcolo.

**1.1.3 Dimostrazioni con Implicazioni**

Se una formula è del tipo $A \Rightarrow B$, possiamo utilizzare catene di equivalenze/implicazioni $A = \cdots \Rightarrow \cdots = \cdots \Rightarrow B$.

Possiamo introdurre altre regole importanti, usate come giustificazioni in prove di implicazioni:

*   Transitività dell'implicazione: $((A \Rightarrow B) \wedge (B \Rightarrow C)) \Rightarrow (A \Rightarrow C)$
*   Modus Ponens: $(P \Rightarrow Q) \wedge P \Rightarrow Q$
*   Tollendo Ponens: $(P \vee Q) \wedge \neg P \Rightarrow Q$

Semplificazione A: $P \wedge Q \Rightarrow P$

Dm Tollendo? (° Introduzione $\vee$): $P \Rightarrow P \vee Q$

$(P \wedge Q) \Rightarrow (P \vee Q)$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$

$(P \wedge Q) \Rightarrow P$

$(P \wedge Q) \Rightarrow Q$

$(P \vee Q) \Rightarrow Q$



---

