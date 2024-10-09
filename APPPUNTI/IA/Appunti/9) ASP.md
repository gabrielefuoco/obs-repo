
| Termine                                 | Spiegazione                                                                                                                            |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Logica Classica (Monotona)**          | Un sistema logico in cui ogni nuova informazione aumenta la conoscenza senza contraddire le informazioni già acquisite.                |
| **Logica Umana (Non-Monotona)**         | Un sistema logico che riflette il ragionamento umano, dove nuove informazioni possono contraddire o integrare la conoscenza esistente. |
| **Commonsense Programming**             | Un approccio alla programmazione che mira a formalizzare il ragionamento umano, tipicamente non monotono.                              |
| **Default Logic**                       | Una teoria che gestisce le eccezioni nel ragionamento non monotono.                                                                    |
| **Circumscription**                     | Una teoria che minimizza ciò che è considerato vero nel ragionamento non monotono.                                                     |
| **Epistemic Reasoning**                 | Un tipo di ragionamento che riguarda la conoscenza e il ragionamento su ciò che è noto e ignoto.                                       |
| **Closed World Assumption (CWA)**       | Un'assunzione che tutto ciò che non è esplicitamente conosciuto è falso o inesistente.                                                 |
| **Logica del Primo Ordine**             | Uno strumento per specificare problemi in modo dichiarativo, rappresentando fatti, regole e relazioni.                                 |
| **Potenza Espressiva**                  | La classe di problemi che un linguaggio può rappresentare.                                                                             |
| **Complessità**                         | La difficoltà computazionale di risolvere un problema.                                                                                 |
| **Datalog**                             | Un linguaggio di programmazione logico che utilizza regole per derivare nuova conoscenza da fatti.                                     |
| **Termini**                             | Elementi di base in Datalog, che possono essere costanti o variabili.                                                                  |
| **Predicati**                           | Funzioni che rappresentano relazioni tra termini.                                                                                      |
| **Atomo**                               | Un'istanza di un predicato.                                                                                                            |
| **Negazione**                           | Un operatore logico che nega ciò che segue.                                                                                            |
| **Letterale**                           | Un atomo o il suo negato.                                                                                                              |
| **Regole di Datalog**                   | Strutture che definiscono nuove relazioni tra predicati.                                                                               |
| **Testa**                               | L'atomo che viene definito in una regola.                                                                                              |
| **Corpo**                               | La condizione che deve essere soddisfatta per rendere vera la testa di una regola.                                                     |
| **EDB (Extensional Database)**          | Predicati definiti esplicitamente nei fatti o nei corpi delle regole.                                                                  |
| **IDB (Intensional Database)**          | Predicati definiti intenzionalmente attraverso regole.                                                                                 |
| **Ricorsione**                          | Un meccanismo che permette di definire relazioni ricorsive in Datalog.                                                                 |
| **Fatti**                               | Assiomi che sono veri per ipotesi, senza condizioni.                                                                                   |
| **Interpretazione**                     | Un insieme di atomi ground che definisce il significato degli atomi in un programma.                                                   |
| **Modello**                             | Un'interpretazione che soddisfa tutte le regole di un programma.                                                                       |
| **Modello Minimale**                    | Un modello che non ha sottoinsiemi che siano anch'essi modelli.                                                                        |
| **Answer Set**                          | Un modello stabile di un programma di logica disgiuntiva.                                                                              |
| **Disjunctive Logic Programming (DLP)** | Un'estensione della programmazione logica che permette di gestire disgiunzioni nella testa delle regole.                               |
| **Integrity Constraints**               | Regole senza testa che impongono vincoli su modelli validi.                                                                            |
| **Weak Constraints**                    | Vincoli che si preferisce non violare, ma la loro violazione non rende il modello non valido.                                          |
| **Aggregate Function**                  | Funzioni che permettono di eseguire operazioni come somma, conteggio, minimo, massimo, ecc., su insiemi di dati.                       |
| **Safety**                              | Una proprietà delle regole che garantisce che le variabili siano correttamente vincolate.                                              |
| **Negazione Stratificata**              | Un metodo per gestire la negazione in programmi ricorsivi.                                                                             |
| **Grafo di Dipendenze**                 | Un grafo che rappresenta le dipendenze tra predicati.                                                                                  |
| **Programma Ground**                    | Un programma in cui tutte le variabili sono state sostituite con costanti.                                                             |
| **Programma Stratificato**              | Un programma in cui non ci sono cicli nel grafo delle dipendenze che coinvolgono la negazione.                                         |
| **Brave-Reasoning**                     | Un tipo di ragionamento che considera vero un atomo se è vero in almeno un Answer Set.                                                 |
| **Cautious-Reasoning**                  | Un tipo di ragionamento che considera vero un atomo se è vero in tutti gli Answer Set.                                                 |
| **Vertex Cover**                        | Un insieme di nodi in un grafo che copre tutti gli archi.                                                                              |
| **Dominating Set**                      | Un insieme di nodi in un grafo che domina tutti gli altri nodi.                                                                        |
| **Semantica di DLP**                    | La semantica che definisce il significato di un programma di logica disgiuntiva.                                                       |
| **Riduzione di un Programma**           | Un processo che elimina le regole con letterali negativi falsi e rimuove i letterali negativi dai corpi delle regole.                  |
| **Teoremi di DLP**                      | Teoremi che descrivono le proprietà degli Answer Set e dei programmi di logica disgiuntiva.                                            |
| **Programma Positivo**                  | Un programma che non contiene negazioni.                                                                                               |
| **Condizione di Supporto**              | Una condizione che deve essere soddisfatta per un atomo per essere supportato in un Answer Set.                                        |
| **Programma con Aggregati**             | Un programma che utilizza funzioni di aggregazione per eseguire operazioni su insiemi di dati.                                         |
| **Semantica con Aggregati**             | La semantica che definisce il significato di un programma con aggregati.                                                               |
| **Programma Ridotto con Aggregati**     | Un programma ridotto che elimina le regole con letterali aggregati falsi e rimuove i letterali aggregati dai corpi delle regole.       |

### Logica Classica vs Logica Non-Monotona
- **Logica classica (monotona):**
  - Ogni nuova informazione **aumenta** la conoscenza.
  - Nuove informazioni non contraddicono quelle già conosciute.

- **Logica umana (non-monotona):**
  - Nuove informazioni possono **contraddire** o **integrare** ciò che già conosciamo.
  - Si basa sul ragionamento di senso comune.
  
#### Commonsense Programming
- **Commonsense Programming:** 
  - Lo scopo è formalizzare il ragionamento umano, che è tipicamente non monotono.
  - Esempi di teorie che affrontano il ragionamento non monotono:
    - **Default Logic**: Gestione delle eccezioni.
    - **Circumscription**: Minimizza ciò che è considerato vero.
    - **Epistemic Reasoning**: Riguarda la conoscenza e il ragionamento su ciò che è noto e ignoto.

#### Closed World Assumption (CWA)
- **CWA (Assunzione di mondo chiuso):**
  - Si assume che tutto ciò che non è esplicitamente conosciuto sia **falso** o **inesistente**.
  - Utilizzata per minimizzare la conoscenza, affermando che solo ciò che è noto è rilevante.

#### Logica del Primo Ordine
- **Logica del primo ordine:**
  - Strumento per **specificare problemi** in maniera dichiarativa, rappresentando fatti, regole e relazioni.
  
#### Potenza Espressiva vs Complessità
- **Potenza espressiva ($Σ^p_2$):**
  - Descrive la **classe di problemi** che un linguaggio può rappresentare.
  - Indica se un problema può essere espresso o rappresentato in un determinato linguaggio.
  
- **Complessità ($Σ^p_2$)**:
  - Descrive la **difficoltà** computazionale di risolvere un problema, utilizzando una macchina deterministica con l'ausilio di un "oracolo".

#### Linguaggi e Problemi
- **Linguaggio $L ∈ Σ^p_2$**:
  - Esiste un programma logico (PL) che risolve i problemi rappresentati da L.
  - Il programma prende come input una **istanza** (w) e fornisce una risposta "yes" se w appartiene al problema rappresentato da L.

## Datalog
- **Termini:** 
  - Possono essere **costanti** o **variabili**.
  - **Costanti:** oggetti esistenti nel mondo, iniziano con una lettera minuscola o sono racchiuse tra virgolette. I numeri sono costanti, ma vanno gestiti con l'**ipotesi di mondo chiuso**.
  - **Variabili:** iniziano con una lettera maiuscola.

- **Predicati:** 
  - Forma generale: `pred(t1, ..., tn)` dove `t1, ..., tn` sono termini.
  - L'**arità** di un predicato è il numero di argomenti.

- **Atomo:** 
  - Istanza di un predicato.

- **Negazione:** 
  - Espressa con la parola chiave `not`, nega ciò che la segue.

- **Letterale:** 
  - Un atomo o il suo negato.

### Regole di Datalog
- Un **programma Datalog** è un insieme di regole.
  - **Struttura di una regola:**
    - `a :- b1, ..., bk, not bk+1, ..., not bm`
    - **Testa**: `a` (un atomo).
    - **Corpo**: una congiunzione di atomi e negazioni (`b1, ..., not bm`).
    - Significato: "la testa `a` è vera se il corpo è vero".

### Definizioni:
- **H(r):** La testa della regola `r`.
- **B(r):** L'insieme di tutti i letterali nel corpo di `r`.
- **B+(r):** L'insieme dei letterali positivi (senza negazione) nel corpo.
- **B−(r):** L'insieme dei letterali negati nel corpo.

### Varianti di Datalog
- **Datalog classico (positivo):** 
  - La testa contiene **un solo atomo**.
  - Tutti i letterali nel corpo sono **positivi**.
  - Non gestisce logica **non monotona**.

- **Datalog disgiuntivo:** 
  - Nella testa possono esserci una **disgiunzione di atomi** (`a1 ∨ ... ∨ an`).

### Ricorsione in Datalog
- **Ricorsione diretta:** la testa di una regola richiama il suo stesso atomo nel corpo.
- **Ricorsione indiretta:** una regola richiama altre regole che a loro volta richiamano la prima.
- La ricorsione è fondamentale per l'**espressività** di Datalog.

### Esempio di Regola
```prolog
britishProduct(X) :- product(X,Y,P), company(P,"UK",SP).
```
Questa regola deriva nuova conoscenza, definendo cosa sia un prodotto britannico.

### Fatti
- I **fatti** sono assiomi che sono veri per ipotesi, senza un corpo (ovvero senza condizioni).
- Esempio di fatto:
```prolog
parent(eugenio, peppe).
parent(mario, ciccio).
```

### Predicati in Datalog
- **EDB (Extensional Database):**
  - Predicati definiti esplicitamente nei fatti o nei corpi delle regole.
  - Possono essere visti come l'**input** del programma.

- **IDB (Intensional Database):**
  - Predicati definiti intenzionalmente attraverso regole.
  - Rappresentano la **conoscenza derivata** o l'**output**.

### Differenze tra Datalog e SQL
- **SQL** è meno potente nella gestione della ricorsione rispetto a Datalog.
  - In SQL, è complesso esprimere query ricorsive come trovare città raggiungibili da un aeroporto con voli diretti e indiretti.
  
  - **In Datalog:** 
    ```prolog
    reaches(lamezia,B) :- connected(lamezia,B).
    reaches(lamezia,C) :- reaches(lamezia,B), connected(B,C).
    ```

### Differenze tra Datalog e Prolog
- **Ordine**: In **Datalog** l'ordine delle regole non è importante, tutto è basato sulla **logica**.

## Interpretazione in un Programma di Logica
- **Interpretazione (I):**
  - Un insieme di **atomi ground**(senza variabili).
  - Definisce il **significato** degli atomi nel programma: 
    - Gli atomi in `I` sono **veri**.
    - Gli atomi non in `I` sono **falsi**.
    - Un **letterale negativo** (`not(a)`) è vero se l'atomo `a` non appartiene a `I` (`a ∉ I`).

### Modelli in Logica
- **Modello (M):**
  - Una **interpretazione** `I` è un **modello** di un programma se, per ogni regola del programma:
    - La **testa** della regola è **vera** rispetto a `I`, se il **corpo** della regola è vero rispetto a `I`.
  
### Esempio di Programma:
```prolog
a :- b, c.
c :- d.
d.
```

#### Interpretazioni Possibili:
- **I = {c, d}:**
  - Gli atomi `c` e `d` sono veri, gli altri sono falsi.
- **I1 = {b, c, d}:**
  - Non è un modello: il corpo della regola `a :- b, c` è vero, ma la testa `a` non è vera in `I1`.
- **I2 = {a, b, c, d}:**
  - È un modello: tutte le regole sono soddisfatte.
- **I3 = {c, d}:**
  - È un modello: la regola `a :- b, c` non è soddisfatta nel corpo, quindi la testa può essere vera o falsa.

### Modelli Minimali
- Un modello è **minimale** se non esiste un suo **sottoinsieme** che sia anch'esso un modello.

#### Esempio:
```prolog
a :- b.
b :- not(c).
```

#### Interpretazioni:
- **I1 = {a}:**
  - Non è un modello: `b` deve essere vero, ma non lo è in `I1`.
  - **I2 = {c}:**
  - È un modello.
- **I3 = {a, b}:**
  - È un modello.
**Modelli minimali:**
  - **I2** e **I3** sono modelli minimali poiché non esiste un sottoinsieme di essi che sia un modello.

## Differenza tra Logica Monotona e Non Monotona
- **Logica Monotona:**
  - Nuove informazioni **non influenzano** i modelli esistenti.
  - **Regole:** La negazione può essere interpretata come una disgiunzione.
    - Esempio: `b :- not(c)` equivale a `b ∨ c :- true`.
  - Modelli **fissi**: anche con nuova conoscenza, i modelli non cambiano.
  
- **Logica Non Monotona:**
  - Nuove informazioni possono **modificare** i modelli.
  - **Default negation:** `b :- not(c)` significa che `b` è vero se non è possibile dimostrare `c`.
  - I modelli possono diventare **non validi** con nuova conoscenza (es. se `c` diventa falso).

### Esempio di Programma con Logica Non Monotona:
```
a :- b.
a ∨ b :- .
:- not(a).
```
- La regola con testa vuota (`:- not(a)`) è un **vincolo**: il corpo non può verificarsi.
- **Interpretazioni:**
  - `I1 = {a}` è un **modello minimale**.
  - `I2 = {a, b}` è un modello, ma **non minimale** (si cerca il modello minimale per l'Answer Set).

### Programmi Ground
- Un **programma ground** contiene tutte le possibili istanziazioni delle regole con fatti.
- Esempio:
```
p(x) :- q(x), not(r(x)).
q(a).
q(b).
r(a).
```
- **Programma ground risultante:**
```
p(a) :- q(a), not(r(a)).
p(b) :- q(b), not(r(b)).
q(a).
q(b).
r(a).
```
- L'unico **Answer Set** è `I = {p(b), q(a), q(b), r(a)}`.

### Concetto di Safety
- Una regola è **safe** se:
  1. **Ogni variabile nella testa** appare in un letterale positivo del corpo.
  2. **Ogni variabile in un letterale negato** appare nel corpo.
  3. **Ogni variabile in un operatore di confronto** appare in un letterale positivo.

- Esempi di regole **non safe:**
```
s(Y) :- b(Y), not r(X).
s(X) :- not r(X).
s(Y) :- b(Y), X < Y.
```
  - In queste regole, ci sono variabili nella testa o nei letterali negati che non appaiono nel corpo.

### Negazione Stratificata e Grafi di Dipendenze
- La **negazione in una ricorsione** può creare problemi.
- Si risolve con la **negazione stratificata**:
  - Si costruisce un **grafo di dipendenze** dove ogni arco rappresenta una relazione tra predicati.
  - Se c'è un **ciclo con negazione**, potrebbe esserci un problema.
  - I **programmi stratificati** (senza cicli con negazione) sono **più semplici** da risolvere.

### Esempio di Programma Stratificato:
```
reach(X) :- source(X).
reach(X) :- reach(Y), arc(Y, X).
noReach(X) :- target(X), not reach(X).
```
- **Programmi stratificati** sono valutati in **tempo polinomiale**.

### Semantica di Programmi con Disgiunzione e Negazione
- **Teorema:** 
  - Se il programma è **positivo** (senza negazioni), esiste un **unico modello minimale**.
  - Se c'è **negazione o disgiunzione**, la situazione diventa più complessa e non necessariamente esiste un modello unico.

## Disjunctive Logic Programming (DLP)

- In **DLP**, le regole possono avere **disgiunzioni** nella testa, permettendo di modellare situazioni in cui possono verificarsi più alternative.
  
  - **Esempio di regola con disgiunzione:**
    ```
    mother(P, S) | father(P, S) :- parent(P, S).
    ```

- Un **programma** è un insieme finito di regole e constraint (regole con testa vuota).

---
### Forma di una regola in DLP

Una regola generica in DLP ha la seguente forma:
```
a1 | a2 | ... | an :- b1, ..., bk, not(bk+1), ..., not(bm).
```
- **Testa (Head):** `a1 | a2 | ... | an` rappresenta una **disgiunzione** di atomi.
  - Almeno uno tra gli `ai` deve essere vero se il **corpo** è vero.
  
- **Corpo (Body):** `b1, ..., bk, not(bk+1), ..., not(bm)` sono **letterali positivi** (senza negazione) e **letterali negati** (con negazione default `not`).
  - Se tutti i letterali nel corpo sono veri, la regola impone che almeno un elemento nella testa sia vero.

**Esempio:**
```
isInterestedInDLP(john) | isCurious(john) :- attendsDLP(john).
attendsDLP(john).
```
- Ci sono due modelli minimali: 
  - Uno in cui `john` è interessato (`isInterestedInDLP(john)`).
  - Uno in cui `john` è curioso (`isCurious(john)`).

---
### Tipologie di Derivazioni in DLP

1. **Brave-Reasoning:**
   - Se un atomo è vero in **almeno un Answer Set**.
   - Formalmente:
     ```prolog
     LP |= brave a  se ∃ un Answer Set S : a ∈ S.
     ```

2. **Cautious-Reasoning:**
   - Se un atomo è vero in **tutti gli Answer Set**.
   - Formalmente:
     ```
     LP |= cautious a se ∀ Answer Set S, a ∈ S.
     ```

---
### Integrity Constraints

- Le **regole senza testa** sono utilizzate per scartare modelli che violano determinati vincoli.
  - **Esempio:**
    ```
    :- edge(x, y), not(inCover(x)), not(inCover(y)).
    ```

---
### Esempio: Problema del Vertex Cover
- Determinare un **vertex cover** minimale (insieme di nodi che coprono tutti gli archi di un grafo).

**Programma:**
```
inCover(x) | outCover(x) :- node(x).
node(x) :- edge(x, _).
node(y) :- edge(_, y).
:- edge(x, y), not(inCover(x)), not(inCover(y)).
```

- **Interpretazione:**
  - Ogni nodo del grafo è **incluso** o **escluso** dal cover.
  - Il vincolo (`:-`) assicura che ogni arco del grafo sia coperto da almeno un nodo.

---
### Esempio: Problema del Dominating Set

- **Obiettivo:** Trovare un insieme di nodi che dominano tutti gli altri nodi nel grafo (ogni nodo è dominato o ha un nodo vicino che lo domina).

**Programma:**
```
inDS(x) | outDS(x) :- node(x).
dominated(x) :- edge(x, y), inDS(y).
dominated(x) :- inDS(x).
:- node(x), not(dominated(x)).
```

- **Interpretazione:**
  - Ogni nodo è **dominato** da un nodo nel dominating set o appartiene al dominating set stesso.

---
### Weak Constraints

- **Vincoli deboli** sono vincoli che si preferisce **non violare**, ma la loro violazione non rende il modello non valido. Tuttavia, ogni violazione impone una penalità.

  - **Sintassi:**
    ```prolog
    :-~ inDS(x), [1@1, x].
    ```
  - Questo vincolo preferisce che `inDS(x)` non sia vero per nessun `x`. 
  - La violazione di questo vincolo comporta una **penalità di 1**.

- **Livello delle penalità:** I vincoli deboli possono avere livelli (`@i`), e si cerca di minimizzare la somma dei pesi dei vincoli violati, a partire dal livello 1.

---
### Esempio di Weak Constraint
```prolog
:-~ inDS(x), [1@1, x].
```
- Penalizza la presenza di `inDS(x)`.
- Gli **Answer Set** validi sono quelli che minimizzano le penalità associate ai vincoli violati.

---
### Complessità

- I problemi in DLP (come il **vertex cover** o il **dominating set**) sono **NP-completi**.
  - La ricerca di un modello minimale in DLP è computazionalmente equivalente a risolvere problemi NP-completi.
---
### Semantica per la Disjunctive Logic Programming (DLP)
La semantica di DLP si basa sulla ricerca di **Answer Set**, chiamati anche **modelli stabili**. 

---
#### Answer Set per Programmi Positivi
- Per i programmi **positivi** (senza negazione), un **Answer Set** è un **modello minimale** del programma, simile alla semantica della logica classica.
  - Un modello minimale è un insieme di atomi veri che non può essere ridotto ulteriormente senza violare le regole del programma.

---
#### Negazione e Complicazioni
Quando viene introdotta la **negazione** nel programma, il calcolo degli **Answer Set** diventa più complesso. 

---
#### Riduzione di un Programma P rispetto a una Interpretazione I
Dato un programma P e una interpretazione I, la **riduzione** del programma positivo $P^I$ viene ottenuta seguendo questi passi:

1. **Eliminare le regole** che contengono un **letterale negativo** che è **falso** rispetto a I.
2. **Rimuovere i letterali negativi** dai corpi delle altre regole.

---
#### Definizione di Answer Set
Un **Answer Set** di un programma P è una interpretazione $I$ tale che $I$ è un **modello minimale** del programma ridotto $(P^I)$.

---
#### Esempio di Calcolo di Answer Set
Consideriamo il seguente programma:
```prolog
a :- d, not(b).
b :- not(d).
d.
```

1. **Interpretazione iniziale I = {a, d}**:
   - \(b\) è falso in I, quindi possiamo eliminare il letterale negativo `not(b)` nella prima regola.
   - \(d\) è vero, quindi possiamo eliminare la seconda regola `b :- not(d)`.

2. **Riduzione del programma**:
   ```prolog
   a :- d.
   d.
   ```

3. **Risultato**: L'interpretazione I = {a, d} è un **Answer Set**.

## Teoremi Importanti per la DLP

1. **Modello Minimo Unico per Programmi Positivi**
   - Un programma di **Datalog positivo senza disgiunzione** ha sempre un **unico modello minimale**.

2. **Inclusione dei Fatti negli Answer Set**
   - Se S è un Answer Set di P, allora i **fatti** del programma devono essere **inclusi** in S.

3. **Unicità dell'Answer Set per un Programma con Fatti**
   - Se l'insieme dei fatti è un Answer Set per un programma P, allora è **l'unico Answer Set** per quel programma.

4. **Condizione di Supporto per un Atomo**
   - Un atomo \(a \in I\) è **supportato** in I se esiste una regola \(r \in P\) tale che:
     - Il corpo di \(r\) è vero rispetto a I.
     - L'unico atomo vero rispetto a I nella testa di \(r\) è \(a\).
     - In altre parole, **head(r) = a**.

5. **Condizione Necessaria e Sufficiente di Supporto**
   - I è un Answer Set per P solo se **ogni atomo** \(a \in I\) è **supportato** in I.
     - Questa condizione diventa **necessaria e sufficiente** se non ci sono **cicli** nel grafo delle dipendenze.

---
### Grafo delle Dipendenze
- Il **grafo delle dipendenze** tra predicati permette di identificare cicli di dipendenze che potrebbero complicare la verifica delle condizioni di supporto.
  - Se ci sono cicli che coinvolgono la **negazione**, il programma può diventare più difficile da risolvere.
---
### Aggregate Function in DLP (Disjunctive Logic Programming)

Le **funzioni di aggregazione** sono state aggiunte ai sistemi di **Answer Set Programming (ASP)** per permettere di eseguire operazioni come somma, conteggio, minimo, massimo, ecc., su insiemi di dati. Queste funzioni sono particolarmente utili per gestire insiemi di valori e calcoli complessi, che in SQL sono facili da implementare ma in DLP richiedono maggiore attenzione.

---
#### Sintassi di Base per l'Aggregazione

- **Aggregazione di un insieme**: `{Variabili : CongiunzioneDiAtomi}`  
  Esempio:
  ```prolog
  {EmpId : emp(EmpId, Sex, Skill, Salary)}
  ```
  Questo espressione restituisce l'insieme degli **ID degli impiegati**.
  
  - Se aggiungessimo una condizione come `emp(EmpId, male, Skill, Salary)`, otterremmo l'insieme degli ID degli **impiegati maschili**.

---
#### Funzioni di Aggregazione

Le **funzioni di aggregazione** sono indicate con `f{}` dove `f` rappresenta una funzione specifica. Alcune delle funzioni di aggregazione più comuni includono:
- **#count**: Conteggio
- **#sum**: Somma
- **#min**: Minimo
- **#max**: Massimo

---
##### Esempio: Conteggio con Aggregazione
Se voglio contare il numero di impiegati maschili, posso utilizzare:
```prolog
5 < #count{EmpId : emp(EmpId, male, Skill, Salary)} ≤ 10
```
Questo vincolo richiede che il numero di impiegati maschili sia compreso tra 5 e 10.

---
#### Esempio di Aggregazione con #sum
Supponiamo di avere una base di dati con le seguenti istanze:
```prolog
emp(1, male, s1, 1000).
emp(2, female, s3, 1000).
emp(3, female, s2, 2000).
emp(4, male, s3, 1500).
```

Se vogliamo calcolare la **somma dei salari** degli impiegati, possiamo scrivere:
```prolog
sum(S) :- S = #sum{Y : emp(Id, , , Y)}.
```

- **Risultato**: La somma ottenuta sarebbe **4500** (anziché 5500) perché uno dei salari `1000` viene considerato duplicato e ignorato.

---
#### Gestione dei Duplicati
Per evitare la rimozione dei duplicati durante l'aggregazione, possiamo includere anche l'ID degli impiegati nell'espressione di aggregazione. In questo modo, ogni occorrenza viene trattata come distinta:
```prolog
sum(S) :- S = #sum{Y, Id : emp(Id, , , Y)}.
```
- **Risultato**: Includendo l'ID, la somma sarà correttamente **5500**.

---
### Semantica in Presenza di Aggregati

Quando utilizziamo **aggregati**, la semantica degli **Answer Set** si estende rispetto alla semantica base.

#### Programma Ridotto con Aggregati
Per verificare se un'interpretazione I è un **Answer Set** in presenza di aggregati, costruiamo un **programma ridotto** seguendo questi passi:

1. **Eliminazione delle regole**:
   - Rimuovere tutte le regole che contengono un **letterale negativo falso** o un **letterale aggregato falso**.
   
2. **Rimozione di letterali aggregati**:
   - Rimuovere i letterali aggregati e negativi dalle regole rimanenti.

3. Se l'interpretazione **I** è un **modello minimale** del programma ridotto, allora I è un **Answer Set**.


