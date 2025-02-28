
Scala è un linguaggio di programmazione moderno che combina la programmazione orientata agli oggetti con la programmazione funzionale. Offre una sintassi concisa e leggibile, un forte sistema di tipi statici e prestazioni elevate.

### Confronto con Java

#### Definizione di una Classe

* **Java:**

```java
public class Person {
public final String name;
public final int age;

Person(String name, int age) {
this. name = name;
this. age = age;
}
}
```

* **Scala:**

```scala
class Person(val name: String, val age: Int) {}
```

Scala offre una sintassi più concisa e intuitiva per la definizione di classi. I campi `name` e `age` sono dichiarati come `val`, il che significa che sono immutabili.

#### Utilizzo di una Classe

* **Java:**

```java
import java. util. ArrayList;

Person[] people;
Person[] minors;
Person[] adults;

{
ArrayList<Person> minorsList = new ArrayList<Person>();
ArrayList<Person> adultsList = new ArrayList<Person>();

for (int i = 0; i < people. length; i++) {
(people[i]. age < 18 ? minorsList : adultsList)
. add(people[i]);
}

minors = minorsList. toArray(people);
adults = adultsList. toArray(people);
}
```

* **Scala:**

```scala
val people: Array[Person]
val (minors, adults) = people partition (_. age < 18)
```

Scala offre una sintassi più concisa e funzionale per l'utilizzo di classi. La funzione `partition` divide l'array `people` in due sotto-array, `minors` e `adults`, in base alla condizione `_.age < 18`.

### Elementi di Scala

* **Chiamata di metodo infissa:** Scala consente l'utilizzo di metodi in forma infissa, offrendo una sintassi più leggibile. Ad esempio, `people partition (_.age < 18)` è più leggibile di `people.partition(_.age < 18)`.
* **Valore funzione:** Le funzioni in Scala sono trattate come valori, facilitando la loro manipolazione e passaggio tra contesti. Ad esempio, la funzione `_.age < 18` è passata come argomento alla funzione `partition`.

### Sintesi

Scala è un linguaggio di programmazione moderno che offre una sintassi concisa e leggibile, un forte sistema di tipi statici e prestazioni elevate. La sua natura orientata agli oggetti e funzionale lo rende un linguaggio potente e versatile per lo sviluppo di software. 

---
### L'Essenza di Scala

Scala è stato progettato con due ipotesi chiave:

1. **Scalabilità:** Un linguaggio di programmazione generico dovrebbe essere scalabile, ovvero i suoi concetti dovrebbero essere applicabili sia a piccole che a grandi parti di codice.
2. **Unificazione:** La scalabilità può essere raggiunta unificando e generalizzando i concetti della programmazione funzionale (FP) e della programmazione orientata agli oggetti (OOP).

### Unificazione di FP e OOP

Scala combina i punti di forza di FP e OOP per creare un linguaggio potente e versatile:

* **Programmazione orientata agli oggetti (OOP):**
 * **Adattabilità e estensione:** Facilita l'adattamento e l'estensione di sistemi complessi attraverso sottotipi, ereditarietà e configurazioni dinamiche.
 * **Astrazione parziale:** Le classi fungono da astrazioni parziali, consentendo di modellare sistemi complessi in modo graduale.
* **Programmazione funzionale (FP):**
 * **Composizione:** Facilita la costruzione di soluzioni complesse partendo da parti semplici tramite funzioni di ordine superiore, tipi algebrici e pattern matching.
 * **Polimorfismo parametrico:** Consente di scrivere codice generico che può essere applicato a diversi tipi di dati.

### Scala: Un Linguaggio Unificato

Scala è un linguaggio orientato agli oggetti e funzionale, completamente interoperabile con Java. Offre:

* **Un modello a oggetti uniforme:** Consente di utilizzare i concetti OOP in modo coerente e intuitivo.
* **Pattern matching e funzioni di ordine superiore:** Fornisce strumenti potenti per la manipolazione di dati e la creazione di codice conciso.
* **Nuovi modi per astrarre e comporre i programmi:** Consente di creare codice più leggibile, manutenibile e riutilizzabile.

### Interoperabilità con Java

Scala è completamente interoperabile con Java. I programmi Scala possono interagire con le librerie Java senza problemi, utilizzando:

* **Chiamate a metodi**
* **Accesso a campi**
* **Ereditarietà di classi**
* **Implementazione di interfacce**

I programmi Scala vengono compilati in bytecode JVM, il che garantisce la compatibilità con l'ecosistema Java.

### Sintassi di Scala

La sintassi di Scala assomiglia a quella di Java, ma presenta alcune differenze:

* **`object` invece di membri statici:** In Scala, gli oggetti sono utilizzati per rappresentare membri statici.
* **`Array[String]` invece di `String[]`:** Scala utilizza una notazione più esplicita per i tipi di dati.
* **Indicizzazione degli array:** Gli array sono indicizzati con `args(i)` invece di `args[i]`.

### Scala: Un Linguaggio Funzionale

Scala supporta uno stile di programmazione funzionale, che consente di scrivere codice più conciso e leggibile.

* **Funzioni di ordine superiore:** Le funzioni possono essere passate come argomenti ad altre funzioni e restituite come risultati.
* **Chiusure:** Le funzioni possono accedere a variabili locali del contesto in cui sono definite.

### Scala: Un Linguaggio Conciso

La sintassi di Scala è leggera e concisa, grazie a:

* **Inference del punto e virgola:** Il compilatore Scala può inferire automaticamente il punto e virgola alla fine di ogni riga.
* **Inference dei tipi:** Il compilatore Scala può inferire automaticamente il tipo di dati delle variabili.
* **Classi leggere:** Le classi in Scala sono più concise rispetto a Java.
* **API estensibili:** Le API di Scala sono progettate per essere estensibili e personalizzabili.
* **Chiusure come astrazioni di controllo:** Le chiusure possono essere utilizzate per creare astrazioni di controllo più flessibili.

## Scala: Un Linguaggio Preciso

Scala offre un sistema di tipi statici che aiuta a prevenire errori durante la compilazione. Questo sistema di tipi statici contribuisce a rendere il codice Scala più sicuro e affidabile.

##### Caratteristiche chiave:

* **Specificazione dei tipi:** È possibile specificare il tipo di dati delle variabili e delle funzioni. Questo aiuta a garantire che il codice sia utilizzato correttamente e che non si verifichino errori di tipo durante la compilazione.
* **Mix-in trait:** I trait sono come interfacce che possono essere utilizzate per aggiungere funzionalità alle classi senza ereditarietà. Questo consente di creare codice più modulare e riutilizzabile.
* **Valori predefiniti:** È possibile fornire valori predefiniti per le variabili. Questo è utile per gestire situazioni in cui una variabile potrebbe non essere inizializzata esplicitamente.

##### Esempio:

```scala
import scala.collection.mutable._

val capital = new HashMap[String, String] with SynchronizedMap[String, String] {
  override def default(key: String) = "? "
}

capital += ("US" -> "Washington",
            "France" -> "Paris",
            "Japan" -> "Tokyo")

assert(capital("Russia") == "? ")
```

##### Spiegazione:

* **`import scala.collection.mutable._`**: Importa le collezioni mutabili dalla libreria standard di Scala.
* **`val capital = new HashMap[String, String] with SynchronizedMap[String, String]`**: Crea una nuova mappa mutabile di tipo `HashMap` che implementa anche il trait `SynchronizedMap`. Questo rende la mappa thread-safe.
* **`override def default(key: String) = "? "`**: Definisce un valore predefinito per la mappa. Se si cerca una chiave che non esiste, la mappa restituirà "? ".
* **`capital += ("US" -> "Washington", "France" -> "Paris", "Japan" -> "Tokyo")`**: Aggiunge elementi alla mappa.
* **`assert(capital("Russia") == "? ")`**: Verifica che la mappa restituisca il valore predefinito "? " per la chiave "Russia", che non è presente nella mappa.

##### In sintesi:

Il codice dimostra come Scala consente di specificare il tipo di dati delle collezioni, di utilizzare i trait per aggiungere funzionalità e di fornire valori predefiniti per le variabili. Questo approccio aiuta a rendere il codice più sicuro, affidabile e manutenibile. 

---
## Grande o piccolo? Il dilemma del design dei linguaggi

Ogni linguaggio di programmazione deve affrontare un dilemma fondamentale: essere **grande** o **piccolo**.
### Grande è buono

Un linguaggio **grande** offre una vasta gamma di funzionalità e costrutti, rendendolo **espressivo** e **facile da usare** per compiti complessi. 
### Piccolo è buono

Un linguaggio **piccolo** si concentra su un set di funzionalità essenziale, risultando **elegante** e **facile da apprendere**.

### Il dilemma

La sfida sta nel trovare un equilibrio tra queste due tendenze. Un linguaggio troppo grande può diventare complesso e difficile da imparare, mentre un linguaggio troppo piccolo potrebbe non essere abbastanza potente per compiti complessi.
### L'approccio di Scala

Scala affronta questo dilemma concentrandosi sulle **capacità di astrazione e composizione** piuttosto che sui costrutti di base del linguaggio. Questo approccio consente a Scala di essere **potente** e **espressivo** pur rimanendo **relativamente semplice** da imparare.

**In sintesi:** Scala cerca di essere sia grande che piccolo, offrendo un set di funzionalità di base relativamente piccolo ma potente, che può essere esteso e combinato in modi complessi attraverso l'astrazione e la composizione.

| **Scala aggiunge** | **Scala rimuove** |
|------------------------------------------|----------------------------------------------|
| + un sistema a oggetti puro | - membri statici |
| + sovraccarico degli operatori | - trattamento speciale dei tipi primitivi |
| + chiusure come astrazioni di controllo | - break, continue |
| + composizione mixin con trait | - trattamento speciale delle interfacce |
| + membri di tipo astratto | - wildcard |
| + pattern matching | |

---

## L'estensibilità di Scala: un esempio con i numeri complessi

Guy Steele ha proposto un benchmark per valutare l'estensibilità di un linguaggio di programmazione: la possibilità di aggiungere un nuovo tipo di dato, come i numeri complessi, e farlo funzionare come se fosse un tipo nativo.

### Il benchmark di Steele

Il benchmark di Steele si basa sulla seguente domanda:

> **"Puoi aggiungere un tipo di numeri complessi alla libreria e farlo funzionare come se fosse un tipo di numero nativo?"**

Questa domanda evidenzia la capacità di un linguaggio di supportare l'estensione delle sue funzionalità di base con nuovi tipi di dati, senza compromettere la coerenza e l'integrazione con il resto del linguaggio.

### Esempi di estensione

Oltre ai numeri complessi, altri esempi di tipi di dati che potrebbero essere aggiunti a un linguaggio includono:

* **BigInt:** per rappresentare numeri interi di grandi dimensioni.
* **Decimal:** per rappresentare numeri decimali con precisione arbitraria.
* **Intervals:** per rappresentare intervalli di valori.
* **Polynomials:** per rappresentare polinomi.

### Estensibilità in Scala

Scala dimostra un'elevata estensibilità, come mostrato nell'esempio dei numeri complessi:

```scala
scala> import Complex._
import Complex._

scala> val x = 1 + 1 * i
x: Complex = 1.0 + 1.0 * i

scala> val y = x * i
y: Complex = -1.0 + 1.0 * i

scala> val z = y + 1
z: Complex = 0.0 + 1.0 * i
```

In questo esempio, `Complex` è un tipo di dato definito dall'utente che rappresenta i numeri complessi. Scala consente di utilizzare `Complex` come se fosse un tipo nativo, supportando operazioni come l'addizione, la moltiplicazione e l'assegnazione.

---

#### Implementazione dei numeri complessi

```scala
object Complex {
  val i = new Complex(0, 1)
  implicit def double2complex(x: Double): Complex = new Complex(x, 0)
  ...
}

class Complex(val re: Double, val im: Double) {
  def + (that: Complex): Complex = new Complex(this.re + that.re, this.im + that.im)
  def - (that: Complex): Complex = new Complex(this.re - that.re, this.im - that.im)
  def * (that: Complex): Complex = new Complex(this.re * that.re - this.im * that.im, 
                                                this.re * that.im + this.im * that.re)
  def / (that: Complex): Complex = {
    val denom = that.re * that.re + that.im * that.im
    new Complex((this.re * that.re + this.im * that.im) / denom, 
                (this.im * that.re - this.re * that.im) / denom)
  }

  override def toString = re + (if (im < 0) "-" + (-im) else "+" + im) + "*i"
  ...
}
```

- **`+`** è un identificatore; può essere utilizzato come nome di metodo.

---

Le operazioni infisse sono chiamate metodi:
```scala
a + b è lo stesso di a.+(b)
```

- **Oggetti** sostituiscono i membri statici della classe.
- **Parametri di classe** invece di campi + costruttore esplicito.
---

## Il design di Scala: un linguaggio ibrido

Scala, abbreviazione di *Scalable Language*, è un linguaggio di programmazione ibrido che combina caratteristiche dei linguaggi orientati agli oggetti e dei linguaggi funzionali.

### Caratteristiche principali

* **Ibrido:** Scala integra senza soluzione di continuità le caratteristiche dei linguaggi orientati agli oggetti e dei linguaggi funzionali.
* **Compilato per la JVM:** Scala viene compilato per funzionare sulla Java Virtual Machine (JVM), offrendo compatibilità con Java e accesso alle librerie Java.
* **Puramente orientato agli oggetti:** Ogni valore in Scala è un oggetto.
* **Funzionale:** Ogni funzione è un valore e ogni valore è un oggetto, quindi le funzioni sono oggetti.
* **Tipi statici:** Scala è un linguaggio a tipi statici, ma l'inferenza di tipo riduce la necessità di dichiarazioni di tipo esplicite.

### Vantaggi di Scala

* **Produttività:** Scala consente di scrivere codice conciso e leggibile, aumentando la produttività degli sviluppatori.
* **Scalabilità:** Scala è progettato per gestire applicazioni complesse e ad alta scalabilità.
* **Affidabilità:** Scala offre un sistema di tipi statico che aiuta a prevenire errori durante la compilazione.
* **Interoperabilità con Java:** Scala può utilizzare le classi Java e interagire con codice Java esistente.

### Differenze tra Scala e Java

Scala presenta diverse caratteristiche che lo differenziano da Java:

* **Tutti i tipi sono oggetti:** In Scala, ogni tipo è un oggetto, mentre in Java solo le classi sono oggetti.
* **Inferenza di tipo:** Scala deduce automaticamente i tipi delle variabili, riducendo la necessità di dichiarazioni di tipo esplicite.
* **Funzioni annidate:** Scala supporta le funzioni annidate, che possono essere definite all'interno di altre funzioni.
* **Le funzioni sono oggetti:** In Scala, le funzioni sono oggetti di prima classe, possono essere passate come argomenti, restituite da funzioni e assegnate a variabili.
* **Supporto per DSL:** Scala supporta la creazione di linguaggi specifici del dominio (DSL), che semplificano la scrittura di codice per compiti specifici.
* **Traits:** Scala offre i *traits*, che sono come interfacce ma possono contenere implementazioni di metodi.
* **Chiusure:** Scala supporta le chiusure, che sono funzioni che possono accedere a variabili locali del contesto in cui sono definite.
* **Concorrenza:** Scala offre un modello di concorrenza ispirato a Erlang, che semplifica la scrittura di codice concorrente.

### Le funzioni come oggetti

In Scala, le funzioni sono oggetti di prima classe. Il tipo di una funzione `S => T` è equivalente a `scala.Function1[S, T]`, dove `Function1` è un *trait* che definisce il metodo `apply`:

```scala
trait Function1[-S, +T] {
  def apply(x: S): T
}
```

Ad esempio, la funzione successore anonima `(x: Int) => x + 1` può essere espansa come:

```scala
new Function1[Int, Int] {
  def apply(x: Int): Int = x + 1
}
```

Questo dimostra che le funzioni in Scala sono oggetti che implementano il *trait* `Function1`.

---

#### Scala cheat sheet (1): Definizioni

##### Definizioni di metodo in Scala:

```scala
def fun(x: Int): Int = {
  result
}
```
```scala
def fun = result
```

##### Definizioni di variabili in Scala:

```scala
var x: Int = expression
val x: String = expression
```

##### Definizioni di metodo in Java:

```java
int fun(int x) {
  return result;
}
```
*(nessun metodo senza parametri)*

##### Definizioni di variabili in Java:

```java
int x = expression;
final String x = expression;
```

### Scala cheat sheet (2): Espressioni

##### Chiamate ai metodi in Scala:

```scala
obj.meth(arg)
```
o 
```scala
obj meth arg
```

##### Espressioni di scelta in Scala:

```scala
if (cond) expr1 else expr2
```
```scala
expr match {
  case pat1 => expr1
  ...
  case patn => exprn
}
```

##### Chiamata ai metodi in Java:

```java
obj.meth(arg)
// (nessun sovraccarico degli operatori)
```

##### Espressioni di scelta e stati in Java:

```java
cond ? expr1 : expr2; // espressione
if (cond) return expr1; // istruzione
else return expr2;
```
```java
switch (expr) {
  case pat1: return expr1;
  ...
  case patn: return exprn; 
} // solo istruzione
```

### Scala cheat sheet (3): Oggetti e Classi

##### Classe e Oggetto in Scala:

```scala
class Sample(x: Int) {
  def instMeth(y: Int) = x + y
}

object Sample {
  def staticMeth(x: Int, y: Int) = x * y
}
```

##### Classe Java con statico:

```java
class Sample {
  final int x;

  Sample(int x) { 
    this.x = x; 
  }

  int instMeth(int y) {
    return x + y;
  }

  static int staticMeth(int x, int y) {
    return x * y;
  }
}
```

### Scala cheat sheet (4): Traits

##### Trait in Scala:

```scala
trait T {
  def abstractMeth(x: String): String

  def concreteMeth(x: String) = x + field

  var field = "!"
}
```

##### Composizione mixin in Scala:

```scala
class C extends Super with T
```

##### Interfaccia Java:

```java
interface T {
  String abstractMeth(String x);
  // (nessun metodo concreto)
  // (nessun campo)
}
```

##### Estensione Java + implementazione:

```java
class C extends Super implements T
```

---

## Sintassi di base e concetti chiave

Scala è un linguaggio di programmazione orientato agli oggetti che integra funzionalità funzionali. Questo documento introduce i concetti di base della programmazione in Scala, inclusi la sintassi, i tipi di dati e i concetti chiave come oggetti, classi, metodi e tratti.

### Sintassi di base

Un programma Scala è composto da oggetti che comunicano tra loro invocando metodi. Gli oggetti hanno stati e comportamenti, e sono istanze di classi.

* **Classi:** Un modello che descrive gli stati e i comportamenti di un oggetto.
* **Metodi:** I comportamenti di un oggetto, implementati come funzioni all'interno di una classe.
* **Campi:** Variabili di istanza che definiscono lo stato di un oggetto.
* **Chiusure:** Funzioni che possono accedere a variabili locali del contesto in cui sono definite.
* **Tratti:** Incapsulano definizioni di metodi e campi, che possono essere riutilizzati mescolandoli in classi.

### Oggetti singleton

Scala offre la parola chiave `object` per definire oggetti singleton, che possono essere istanziati una sola volta.

```scala
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello World!")
  }
}
```

### Il trait `App`

Il trait `App` consente di eseguire un programma Scala senza definire un metodo `main`.

```scala
object HelloYou extends App {
  if (args.length == 0) {
    println("Hello You!")
  } else {
    println("Hello " + args(0))
  }
}
```

### Tipi di dati

Scala supporta una varietà di tipi di dati, tra cui:

* **Numerici:** `Byte`, `Short`, `Int`, `Long`, `Float`, `Double`
* **Caratteri:** `Char`
* **Stringhe:** `String`
* **Booleani:** `Boolean`
* **Unit:** Corrisponde a `void` in altri linguaggi.
* **Null:** Riferimento nullo o vuoto.
* **Nothing:** Sottotipo di ogni altro tipo, non contiene alcun valore.
* **Any:** Supertipo di ogni altro tipo, qualsiasi oggetto è di tipo `Any`.
* **AnyRef:** Supertipo di qualsiasi tipo di riferimento.

## Variabili in Scala: 

In Scala, le variabili possono essere dichiarate usando le parole chiave `var` e `val`:

* `var`: Dichiara una variabile mutabile, il cui valore può essere modificato dopo l'inizializzazione.
* `val`: Dichiara una variabile immutabile, il cui valore non può essere modificato dopo l'inizializzazione.

```scala
var myVar = 10 // Variabile mutabile, inizializzata a 10
val myVal = "Hello, senza dichiarazione di dati" // Variabile immutabile, inizializzata a "Hello, senza dichiarazione di dati"
```

È possibile dichiarare esplicitamente il tipo di una variabile:

```scala
var myVar2: Int = 3 // Variabile mutabile di tipo Int, inizializzata a 3
val myVal2: String = "Hello, con dichiarazione di dati" // Variabile immutabile di tipo String, inizializzata a "Hello, con dichiarazione di dati"
```

La sintassi generale per la dichiarazione di variabili è:

```scala
// val o var NomeVariabile: TipoDati = [ValoreIniziale]
```

### Scope delle variabili

Le variabili in Scala possono avere tre scope diverse:

* **Campi:** Variabili dichiarate all'interno di una classe.
* **Parametri di metodo:** Variabili dichiarate come argomenti di un metodo.
* **Variabili locali:** Variabili dichiarate all'interno di un metodo o di un blocco di codice.

### Inferenza di tipo

Scala deduce automaticamente il tipo di una variabile se il compilatore può determinarlo dal valore iniziale:

```scala
val a = 1 // Il compilatore deduce che 'a' è di tipo Int
val b = 2.0 // Il compilatore deduce che 'b' è di tipo Double
val c = "Hi!" // Il compilatore deduce che 'c' è di tipo String
```

### Tipizzazione statica vs. tipizzazione dinamica

I linguaggi a tipizzazione statica, come Scala, eseguono il controllo dei tipi a tempo di compilazione. Questo significa che gli errori di tipo vengono rilevati durante la compilazione, evitando errori a runtime. I linguaggi a tipizzazione dinamica, come Python, eseguono il controllo dei tipi a runtime. Questo può portare a errori a runtime se il tipo di un valore non è corretto.

### Dinamicamente vs Tipizzati Staticamente

```java
// Esempio Java
int num;
num = 5;
```

```groovy
// Esempio Groovy
num = 5;
```

```groovy
// Esempio Groovy
number = 5;
numbr = (number + 15) / 2;
// nota il typo. Questo codice si compilerà tranquillamente, ma potrebbe
// produrre un errore
```

## Variabili

```scala
// Interi
val b: Byte = 1
val s: Short = 2
val x: Int = 3
val l: Long = 4

// Decimali
val f: Float = 1
val d: Double = 2.0

// Grandi Numeri
var bigX = BigInt(1234567890)
var bigD = BigDecimal(123456.78)
```

```scala
// Un operatore è un metodo!!! 
bigX += 1
bigD += 1
```

```scala
val c: Char = 'c'
val str: String = "Sequenza di caratteri"
```

### String

```scala
val firstName = "Mario"
val lastName = "Rossi"

var completeName = firstName + " " + lastName
completeName = s"$firstName $lastName"

val multiLineStr =
  """Questa invece è
    | una stringa
    | su più righe;
    | basta limitarla tra 2 triple
    | di doppi apici.
    |""".stripMargin
```
## If/else

```scala
// if
var fileName = "default.txt"
if (args.length > 0) {
  fileName = args(0)
}

// if/else-if/else
if (args.isEmpty) {
  println("Nessun argomento passato in input.")
} else if (args.length == 1) {
  println("Un solo argomento passato in input.")
} else {
  println(s"${args.length} argomenti passati in input.")
}

// if
val a = 5
val b = 6
val c = if (b < a) b else a
```

### For

```scala
val upperBound = 5

for (i <- 1 to upperBound) println(s"Iterazione $i") // i=1...5

for (i <- 1 until upperBound) println(s"Iterazione $i") // i=1...4

for (i <- 0 to upperBound if (i % 2 != 0)) println(s"i = $i")
```

### For

```scala
val names = List("Cristiano", "Gonzalo", "Paulo")

for (name <- names) println(name)

val filmDirectors = Map(
  "Blackhat" -> "Michael Mann",
  "I.T. - Una mente pericolosa" -> "John Moore",
  "Snowden" -> "Oliver Stone"
)

for ((film, director) <- filmDirectors) 
  println(s"Film: $film, Regista: $director")
```

## Match

```scala
val x = 5
val aDayOfWeek = x match {
  case 1 => "Lunedì"
  case 2 => "Martedì"
  case 3 => "Mercoledì"
  case 4 => "Giovedì"
  case 5 => "Venerdì"
  case 6 => "Sabato"
  case 7 => "Domenica"
  case _ => "Nessun giorno della settimana." // default
}

val y = 4
y match {
  case 1 => {
    println("Uno, un numero solitario")
  }
  case x if x == 2 || x == 3 => {
    println("Due è una coppia, tre una compagnia")
  }
  case x if x >= 4 => {
    println("Quattro è una festa!")
  }
  case _ => println("Il tuo numero è minore o uguale a zero...")
}
```

## Gestione delle Eccezioni

```scala
try {
  print("Inserisci un numero pari >> ")
  val input = StdIn.readInt()
  if (input % 2 != 0) {
    throw new RuntimeException("Input non valido.")
  }
} catch {
  case ex: Exception => {
    println(ex.getMessage)
    println("Ops... Qualcosa è andato storto.")
  }
}
println("Complimenti! Hai inserito un numero pari.")
```

```scala
def toInt(s: String): Int = {
  try {
    Integer.parseInt(s.trim)
  } catch {
    case e: Exception => 0
  }
}
```

## Gestione delle Eccezioni

```scala
def toInt(s: String): Option[Int] = {
  try {
    Some(Integer.parseInt(s))
  } catch {
    case e: Exception => None
  }
}
```

**`Some`** (esecuzione riuscita) e **`None`** (esecuzione con eccezione) sono due sottoclassi di **`Option`**; possono essere intese come contenitori di oggetti (per questo sono parametrici) dove **`Some`** contiene QUALCOSA e **`None`** NIENTE.

Per estrarre l'oggetto contenuto, usa il metodo **`get()`**.

```scala
val result = toInt("2")
println(result) // OUTPUT: Some(2)
println(result.get) // OUTPUT: 2
```

## Option / Some / None

Una classe è un modello (cioè, un piano di design o un altro disegno tecnico) per oggetti. Una volta definita una classe, puoi creare oggetti dal modello della classe usando la parola chiave **`new`**. Attraverso l'oggetto, puoi utilizzare tutte le funzionalità della classe definita.

## Class and Instances (Object)

```scala
class Point(xc: Int, yc: Int) {
  var x: Int = xc
  var y: Int = yc

  def move(dx: Int, dy: Int) {
    x = x + dx
    y = y + dy
    println("Point x location: " + x)
    println("Point y location: " + y)
  }
}

object Demo {
  def main(args: Array[String]) {
    val pt = new Point(10, 20)

    // Move to a new location
    pt.move(10, 10)
  }
}
```

## Class

## Estensione di una classe

```scala
class Point(xc: Int, yc: Int) {
  var x: Int = xc
  var y: Int = yc

  def move(dx: Int, dy: Int) {
    x = x + dx
    y = y + dy
    println("Point x location: " + x)
    println("Point y location: " + y)
  }
}

class Location(override val xc: Int, override val yc: Int, val zc: Int) extends Point(xc, yc) {
  var z: Int = zc

  def move(dx: Int, dy: Int, dz: Int) {
    x = x + dx
    y = y + dy
    z = z + dz
    println("Point x location: " + x)
    println("Point y location: " + y)
    println("Point z location: " + z)
  }
}

class Student(id: Int, name: String) {
  var age: Int = 0

  def showDetails() {
    println(id + " " + name + " " + age)
  }

  def this(id: Int, name: String, age: Int) {
    this(id, name) // Chiamata al costruttore primario
    this.age = age
  }
}

object MainObject {
  def main(args: Array[String]) {
    var s = new Student(101, "Rama", 20)
    s.showDetails()
  }
}
```

### Costruttori

I costruttori in Scala sono utilizzati per inizializzare gli oggetti quando vengono creati. Non è necessario dichiarare esplicitamente un costruttore, in quanto Scala fornisce un costruttore predefinito. Tuttavia, è possibile definire costruttori personalizzati per controllare l'inizializzazione degli oggetti.

### Modificatori di Accesso

Scala supporta i seguenti modificatori di accesso:

* **Privato:** I membri privati sono visibili solo all'interno della classe o dell'oggetto in cui sono definiti.
* **Protetto:** I membri protetti sono accessibili dalle sottoclassi della classe in cui sono definiti.
* **Pubblico:** I membri pubblici sono accessibili da qualsiasi parte del codice.

### Operatori

Gli operatori sono simboli speciali che eseguono operazioni specifiche. Scala supporta diversi tipi di operatori:

* **Aritmetici:** `+`, `-`, `*`, `/`, `%`
* **Relazionali:** `==`, `!=`, `>`, `<`, `>=`, `<=`
* **Logici:** `&&`, `||`, `!`
* **Bitwise:** `&`, `|`, `^`, `~`, `<<`, `>>`, `>>>`
* **Assegnazione:** `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`

### Funzioni e Metodi

Scala distingue tra funzioni e metodi:

* **Funzioni:** Oggetti di prima classe che possono essere assegnati a variabili e passati come argomenti.
* **Metodi:** Funzioni definite come membri di una classe.

In pratica, i termini "funzione" e "metodo" sono spesso usati in modo intercambiabile in Scala.

### Esempio

```scala
class Persona(val nome: String, var eta: Int) {

  // Costruttore secondario
  def this(nome: String) = this(nome, 0)

  // Metodo
  def invecchia(): Unit = {
    eta += 1
  }

  // Funzione
  def saluta(): String = s"Ciao, mi chiamo $nome!"
}

object Main {
  def main(args: Array[String]): Unit = {
    val persona1 = new Persona("Mario", 30)
    val persona2 = new Persona("Luigi")

    println(persona1.saluta())
    println(persona2.eta)

    persona1.invecchia()
    println(persona1.eta)
  }
}
```

In questo esempio, `Persona` è una classe con un costruttore principale che accetta `nome` e `eta` come parametri. Il costruttore secondario `this(nome: String)` inizializza `eta` a 0. `invecchia` è un metodo che incrementa l'età della persona, mentre `saluta` è una funzione che restituisce un saluto.

## Funzioni

```scala
object Demo {
  def main(args: Array[String]) {
    println("Returned Value: " + addInt(5, 7))
    println("Returned Value: " + subInt(4, 3))
  }

  def addInt(a: Int, b: Int): Int = {
    var sum: Int = 0
    sum = a + b
    return sum
  }

  def subInt(a: Int, b: Int): Int = a - b
}
```

## Funzioni Anonymous

```scala
object Demo {
  def main(args: Array[String]) {
    val f = (x: Int, y: Int) => x * y + x
    println(f(3, 4))
  }
}
```

## Ricorsione

```scala
object e06FunctionRecursion {
  // Algoritmo di Euclide per il massimo comune divisore
  def gcd(x: Int, y: Int): Int = {
    if (y == 0) x else gcd(y, x % y)
  }

  def main(args: Array[String]): Unit = {
    println(gcd(48, 18))
  }
}
```

## Funzione di Ordine Superiore

```scala
object e07HigherOrderFunction {
  def main(args: Array[String]): Unit = {
    val list = List(1, 2, 3)
    list.foreach(x => println(x))
    list.foreach(println(_))

    val list2 = list.map(x => x * 2)
    println("Moltiplicato per 2, la lista è " + list2)
    val list2_1 = list.map(_ * 2)
    println("Moltiplicato per 2, la lista è " + list2_1)

    val sum = list.reduce((a, b) => a + b)
    println("La somma degli elementi della lista è " + sum)
    val sum_1 = list.reduce(_ + _)
    println("La somma degli elementi della lista è " + sum_1)
  }
}
```

#### Chiamare metodi senza parentesi e punti

```scala
object e08FunctionWithoutParAndDots {
  def main(args: Array[String]): Unit = {
    var ret = "1,2,3" split ", "
    // è equivalente a var ret = "1,2,3".split(", ")
    println(ret.toList.toString())
    // Aggiungere ". " è opzionale durante la chiamata a una funzione
    // () può anche essere rimosso al momento della chiamata se hai solo
    // un parametro o nessun parametro.
    var g = "Hello! I'm Fabrizio"
    println(g indexOf "o")
    // println(g.indexOf("o"))
    // println(g indexOf "o" 5) non funziona
    // println(g.indexOf("o", 5))
  }
}
```

```
Permette di aggiungere gruppi di parametri alla funzione iniziale
man mano che si procede, o quando appaiono/diventano disponibili.
```

##### Chiamare metodi con un insieme diverso di argomenti/parentesi

##### (Currying)

```scala
def queryDbUsingConn(dbConn: DbConnection)(query: Query) = {
  dbConn.query(query)
}

val dbConn = new DatabaseConnection

def queryDb = queryDbUsingConn(dbConn) _

queryDb(query) // non è più necessario preoccuparsi della connessione al db
mentre si passano le query!
```

## Trait

```scala
trait Pet {
  def comeToMaster(): Unit // metodo astratto
  def speak(): Unit = print("Hei! ") // metodo concreto
}

class Cat(name: String) extends Pet {
  override def comeToMaster(): Unit = println("...")
  override def speak(): Unit = println("Miao! ")
}
```

## Trait

```scala
trait Composition {
  var composer: String
  def compose(): String
}

trait SoundProduction {
  var engineer: String
  def produce(): String
}

class Score(var composer: String, var engineer: String) 
extends Composition with SoundProduction {
  override def compose(): String = s"The score is composed by $composer"
  override def produce(): String = s"The score is produced by $engineer"
}
```

Quando si eredita da più tratti in Scala, si utilizza la parola chiave `extends` solo per il primo tratto. Per i tratti successivi, si utilizza la parola chiave `with`

## Classe Astratta

 Una classe astratta in Scala si differenzia da un trait per la presenza di un costruttore e per la possibilità di essere invocata

```scala
abstract class Person(private val name: String) {
  def getName(): String = name
  override def toString: String = s"Mi chiamo $name."
}

class Student(name: String, private val number: Int) extends Person(name) {
  def getNumber(): Int = number
  override def toString: String = s"${super.toString} La mia matricola è $number."
}

class Teacher(name: String, private val subject: String) extends Person(name) {
  def getSubject(): String = subject
  override def toString: String = s"${super.toString} Insegno $subject."
}
```

## Classi Astratte

### Gerarchie di Tipo in Scala

Scala supporta un sistema di tipi ricco e flessibile, che consente di creare gerarchie di tipi complesse. Le classi astratte sono un elemento fondamentale di queste gerarchie.

### Collezioni

Le collezioni in Scala sono un tipo di dato fondamentale per la gestione di dati strutturati. Scala distingue tra collezioni mutabili e immutabili:

#### Collezioni Immutabili

Le collezioni immutabili non possono essere modificate dopo la loro creazione. Ogni operazione che sembra modificare la collezione restituisce in realtà una nuova collezione con le modifiche desiderate, lasciando la collezione originale intatta.

##### Esempi:

* `List`: Una lista immutabile ordinata.
* `Set`: Un insieme immutabile non ordinato di elementi unici.
* `Map`: Un dizionario immutabile che associa chiavi a valori.

#### Collezioni Mutabili

Le collezioni mutabili possono essere modificate in loco. È possibile aggiungere, rimuovere o modificare elementi direttamente nella collezione.

##### Esempi:

* `ArrayBuffer`: Una lista mutabile ordinata.
* `HashSet`: Un insieme mutabile non ordinato di elementi unici.
* `HashMap`: Un dizionario mutabile che associa chiavi a valori.

## Lista

```scala
val list1 = List("a", "b", "c", "d", "e")
print(list1)
for (i <- list1)
  print(i)

val list2 = List("1", "2", "3")

val list3 = List.concat(list1, list2)
println(list3)
list3.foreach(x => print(x))
```
* **Definizione:** Una lista è una collezione di dati **immutabili**.
* **Implementazione:** In Scala, `List` rappresenta una lista collegata. 

## ListBuffer

```scala
import scala.collection.mutable.ListBuffer

val list1 = ListBuffer("a", "b", "c", "d", "e")

for (i <- list1) print(i)

list1 += ("f", "g")

"z" +=: list1

list1.append("h")
list1.prepend("y")
```

## Buffer in Scala

* **Implementazione:** Questa implementazione di buffer utilizza una lista mutabile come struttura dati sottostante.
* **Tempo di esecuzione:**
 * **Prepend e Append:** Le operazioni di inserimento all'inizio (prepend) e alla fine (append) della lista hanno un tempo di esecuzione costante. Ciò significa che il tempo necessario per completare queste operazioni non dipende dalla dimensione della lista.
 * **Altre operazioni:** La maggior parte delle altre operazioni, come l'accesso a un elemento specifico o la rimozione di un elemento in una posizione specifica, hanno un tempo di esecuzione lineare. Ciò significa che il tempo necessario per completare queste operazioni aumenta linearmente con la dimensione della lista. 

## Set

```scala
var s1 = Set("a", "b", "c", "a")
println(s1)

val s2 = s1 + "d"
println(s2)

val s3 = s2 + ("d", "e")
println(s3)

val s4 = s2 ++ List("d", "e")
println(s4)
```

I Set sono Iterabili che non contengono elementi duplicati.

## Map

```scala
val m1 = Map("a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4)
println(m1)

for (i <- m1)
  print(i)
println()

for ((a, b) <- m1)
  print(a + " " + b + "; ")
println()

println(m1("a"))

val m2 = Map("z" -> 100, "y" -> 99)
val m3 = m1 ++ m2
println(m3)
```

Una Map è un Iterable costituito da coppie di chiavi e valori (chiamate anche mappature o associazioni).

## Tuple

```scala
val t1 = ("a", 1, true)
println(t1)

println(t1._1)
println(t1._2)

val (x: Int, y: Int, z: Int) = (1, 2, 3)

val (a: String, b: Int, c: Boolean) = ("b", 2, false)
println(a)
```

 Una tupla è un valore che contiene un numero fisso di elementi, ciascuno con il proprio tipo. Le tuple sono immutabili.

## Array

```scala
val arr1 = Array("a", "b", "c", "d", "e")
print(arr1)

for (i <- arr1)
  print(i)

println(arr1(4))

arr1(4) = "x"
println(arr1(4))

val arr2 = Array("1", "2", "3")

val arr3 = Array.concat(arr1, arr2)
println(arr3)
arr3.foreach(x => print(x))
```

* **Definizione:** Un Array è un tipo speciale di collezione in Scala. 
* **Caratteristiche:**
 * **Dimensione fissa:** Un array ha una dimensione fissa, che viene definita al momento della creazione.
 * **Tipo di dato uniforme:** Tutti gli elementi di un array devono essere dello stesso tipo di dato.
 * **Indici:** Gli elementi di un array sono indicizzati a partire da zero. L'ultimo elemento ha indice `n-1`, dove `n` è il numero totale di elementi.
 * **Mutabilità:** Gli elementi di un array sono mutabili, ovvero possono essere modificati dopo la creazione dell'array. 

## Equals

```scala
object c03IsEqual {
  // Metodo principale
  def main(args: Array[String]) {
    // Creazione degli oggetti
    var x = Subject("Scala", "Equality")
    var y = Subject("Scala", "Equality")
    var z = Subject("Java", "Array")

    // Visualizza true se le istanze
    // sono uguali, altrimenti false
    println(x.equals(y))
    println(x.equals(z))
    println(x == y)
  }
}

case class Subject(LanguageName: String, TopicName: String)
```

## Equals e hashcode

```scala
class ComputerProgrammer {
  def canEqual(other: Any): Boolean =
    other.isInstanceOf[ComputerProgrammer]

  override def equals(other: Any): Boolean = other match {
    case that: ComputerProgrammer =>
      (that canEqual this) && id == that.id
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(id)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
```

Il metodo hashCode dovrebbe SEMPRE essere sovrascritto e re- implementato quando il metodo equals viene sovrascritto

## Chaining di metodi

```scala
person.setFirstName("Leonard")
     .setLastName("Nimoy")
     .setAge(82)
     .setCity("Los Angeles")
     .setState("California")
```

Per supportare questo stile di programmazione:
- Se la tua classe può essere estesa, specifica questo tipo come tipo di ritorno dei metodi in stile fluente.
- Se sei sicuro che la tua classe non verrà estesa, puoi opzionalmente restituire `this` dai tuoi metodi in stile fluente.

```scala
class Person {
  protected var fname = ""
  protected var lname = ""

  def setFirstName(firstName: String): this.type = {
    fname = firstName
    this
  }

  def setLastName(lastName: String): this.type = {
    lname = lastName
    this
  }
}
```

## For con yield

```scala
scala> for (i <- 1 to 5) yield i * 2
res11: scala.collection.immutable.IndexedSeq[Int] = Vector(2, 4, 6, 8, 10)
```

Per ogni iterazione del tuo ciclo `for`, `yield` genera un valore che sarà ricordato. È come se il ciclo `for` avesse un buffer che non puoi vedere, e per ogni iterazione un altro elemento viene aggiunto a quel buffer. Quando il tuo ciclo `for` termina, restituirà questa collezione di tutti i valori generati.

## Case vs No Case (in funzione)

### Esempio:

```scala
val prices = Map("bread" -> 4.56, "eggs" -> 2.98, "butter" -> 4.35)
```

**Obiettivo:** Ridurre il prezzo di ogni articolo di 1.1 unità.

##### Soluzione 1 (senza `case`):

```scala
prices.map((k, v) => (k, v - 1.1)).toMap
```

##### Errore:

```
Error: Il tipo previsto richiede una funzione a un argomento che accetta un 2-Tuple.
Considera una funzione anonima di pattern matching, `{ case (k, v) => ... }`
```

##### Spiegazione dell'errore:

* `map` si aspetta una funzione che accetta un singolo argomento (una coppia chiave-valore in questo caso).
* La funzione `(k, v) => (k, v - 1.1)` è una funzione a due argomenti (k e v).

##### Soluzione 2 (con `case`):

```scala
prices.map { case (k, v) => (k, v - 1.1) }.toMap
```

##### Spiegazione:

* `case (k, v)` è un pattern matching che decompone la coppia chiave-valore in due variabili, `k` e `v`.
* La funzione anonima `{ case (k, v) => (k, v - 1.1) }` accetta una coppia chiave-valore come argomento e restituisce una nuova coppia con il valore modificato.

##### Conclusione:

* L'utilizzo di `case` nel pattern matching è necessario quando si lavora con funzioni che accettano tuple come argomenti.
* `case` consente di decomporre la tupla in variabili separate, rendendo la funzione più leggibile e facile da usare. 

