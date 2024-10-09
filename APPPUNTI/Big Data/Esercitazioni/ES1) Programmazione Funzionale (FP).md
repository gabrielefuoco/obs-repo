---

---
| Termine                            | Spiegazione                                                                                                                                                |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Programmazione Funzionale (FP)** | Un paradigma di programmazione che vede il calcolo come la valutazione di funzioni matematiche, evitando l'uso di iteratori e cicli.                       |
| **Funzione**                       | Un blocco di codice identificato da un nome, che può accettare parametri e restituire un valore, idealmente senza effetti collaterali.                     |
| **Metodo**                         | Un blocco di codice definito all'interno di un'istanza di classe, legato al contesto dell'oggetto a cui appartiene.                                        |
| **Funzione Pura**                  | Una funzione che rispetta due principi: determinismo (stesso input, stesso output) e assenza di effetti collaterali (non modifica lo stato esterno).       |
| **Effetto Collaterale**            | Un'azione che influisce su qualcosa al di fuori del contesto della funzione, come la modifica di variabili globali o l'esecuzione di operazioni di I/O.    |
| **Cittadini di Prima Classe**      | In programmazione funzionale, le funzioni sono considerate cittadini di prima classe, il che significa che possono essere utilizzate come valori ordinari. |
| **Espressione Lambda**             | Una funzione anonima, definita in modo conciso con la sintassi `(parametri) -> espressione`.                                                               |
| **Funzioni di Ordine Superiore**   | Funzioni che accettano altre funzioni come argomenti o che restituiscono funzioni come risultato.                                                          |
| **Stream**                         | Una sequenza di elementi da una sorgente di dati che supporta operazioni aggregate.                                                                        |
| **Operazioni Intermedie**          | Operazioni che trasformano uno Stream in un nuovo Stream, come `filter`, `map`, `sorted`.                                                                  |
| **Operazione Terminale**           | Un'operazione che conclude il flusso di uno Stream, come `reduce`, `collect`, `forEach`.                                                                   |
| **Collector**                      | Un'utilità per combinare gli elementi di uno Stream in un risultato finale, come una collezione o un valore singolo.                                       |
| **Optional**                       | Un contenitore che può contenere o meno un valore, utile per gestire situazioni in cui un risultato potrebbe non essere presente.                          |
| **Interfaccia Funzionale**         | Un'interfaccia con un solo metodo astratto, utilizzata per rappresentare funzioni.                                                                         |
| **Macchina di Turing**             | Un modello matematico per computer general-purpose, spesso associato alla programmazione imperativa.                                                       |
| **Lambda Calcolo**                 | Una notazione matematica per esprimere funzioni e applicazione di funzioni, spesso associata alla programmazione funzionale.                               |
| **Lazy Evaluation**                | Valutazione ritardata, dove le operazioni vengono eseguite solo quando necessario.                                                                         |
| **Eager Evaluation**               | Valutazione immediata, dove le operazioni vengono eseguite non appena vengono incontrate.                                                                  |
| **Peek**                           | Un'operazione intermedia che restituisce una copia dello stream e applica una funzione Consumer sul risultato.                                             |
| **Parallel Stream**                | Una versione multithread di uno Stream, che consente di eseguire le operazioni in parallelo.                                                               |
| **Function**                       | Un'interfaccia funzionale che accetta un singolo argomento e restituisce un risultato.                                                                     |
| **BiFunction**                     | Un'interfaccia funzionale che accetta due argomenti e restituisce un risultato.                                                                            |
| **Predicate**                      | Un'interfaccia funzionale che accetta un singolo argomento e restituisce un valore booleano.                                                               |
| **Consumer**                       | Un'interfaccia funzionale che accetta un singolo argomento e non restituisce alcun risultato.                                                              |
| **Supplier**                       | Un'interfaccia funzionale che non accetta argomenti e restituisce un risultato.                                                                            |
| **Compose**                        | Un metodo che concatena due funzioni, applicando la seconda funzione al risultato della prima.                                                             |
| **AndThen**                        | Un metodo che concatena due funzioni, applicando la prima funzione al risultato della seconda.                                                             |
| **Comparator**                     | Un'interfaccia funzionale che confronta due oggetti e restituisce un valore negativo, zero o positivo a seconda dell'ordine degli oggetti.                 |
| **BinaryOperator**                 | Un'interfaccia funzionale che accetta due argomenti dello stesso tipo e restituisce un risultato dello stesso tipo.                                        |

## Cos'è la Programmazione Funzionale?

La **Programmazione Funzionale** è un paradigma che vede il calcolo come la valutazione di funzioni matematiche:
  - Non si utilizzano iteratori, cicli `for` o `while`, ma solo funzioni.
  
- **Funzione** vs **Metodo**:
  - Una funzione è un blocco di codice identificato da un nome, che può accettare parametri e restituire un valore. A differenza dei metodi, le funzioni in programmazione funzionale sono idealmente prive di effetti collaterali.

#### Programmazione Imperativa vs Programmazione Funzionale

**Esempio: restituire tutti i valori pari dall'input [1, 4, 6, 9, 11]**

*Programmazione Imperativa:*
```java
public static void getEven(ArrayList<Integer> vals){
    ArrayList<Integer> odds = new ArrayList<>();
    for (Integer val : vals)
        if (val % 2 != 0) odds.add(val);
    vals.removeAll(odds);
}
```

*Programmazione Dichiarativa (Funzionale):*
```java
public static ArrayList<Integer> getEvenFunctional(ArrayList<Integer> vals){
    Stream<Integer> valsStream = vals.stream();
    return valsStream.filter(s -> s % 2 == 0)
                      .collect(Collectors.toCollection(ArrayList::new));
}
```

La versione dichiarativa non modifica la collezione originale, ma ne restituisce una nuova.

### Funzione vs Metodo

- **Metodo** (OOP):
  - È definito all'interno di un'istanza di classe.
  - Non è indipendente, poiché è legato al contesto dell'oggetto a cui appartiene.
  - Può dipendere da variabili esterne ai suoi argomenti.
  - Può modificare i valori degli argomenti o di altre variabili statiche.

- **Funzione** (FP):
  - Esiste indipendentemente da qualsiasi istanza di classe.
  - Può restituire un valore o un'altra funzione.
  - Non modifica né gli argomenti né variabili statiche.
  - Le funzioni possono essere composte liberamente, indipendentemente dal contesto.

---
### Vantaggi della Programmazione Funzionale

- **Assenza di stato condiviso**:
  - Facilita la comprensione del codice in termini di:
    - **Correttezza**: il programma svolge il compito previsto.
    - **Performance**: misurazione del tempo di esecuzione.
    - **Scalabilità**: impatto delle prestazioni con input crescenti.
    - **Sicurezza**: riduzione dei rischi di uso malevolo dell'algoritmo.
  
- **Concorrenza semplice**:
  - I problemi legati alla concorrenza sono minimizzati e l'ordine di esecuzione diventa irrilevante.

- **Testing e debugging**:
  - Le funzioni pure sono facili da testare in isolamento.
  - Il codice di test può coprire facilmente casi tipici, limiti validi e non validi.

- **Algoritmi più eleganti**:
  - Spesso più leggibili, compatti e semplici da capire.

---
### Funzione Pura
Una **funzione pura** deve rispettare due principi fondamentali:
- **Determinismo**: dati gli stessi input, produce sempre lo stesso output.
- **Assenza di effetti collaterali**: non deve eseguire operazioni che influenzano l'esterno, come:
  - Modificare variabili globali.
  - Alterare gli argomenti della funzione.
  - Effettuare operazioni di I/O (es. richieste HTTP, accesso a file).

Una funzione non pura viola questi principi, risultando imprevedibile e capace di generare effetti collaterali.

---
### Effetto Collaterale
- Si verifica quando la valutazione di un'espressione influisce su qualcosa al di fuori del suo contesto.
- **Funzioni pure**:
  - Non utilizzano valori esterni (come variabili globali).
  - Accettano solo parametri e restituiscono un valore senza modificare lo stato esterno.
  - Non hanno dipendenze dallo stato né dall'esecuzione precedente.
  - Non richiedono input dall'utente.

- **Problemi delle funzioni con effetti collaterali**:
  - Difficili da testare e debuggare.
  - Complicano il parallelismo.
 
### Funzioni Pure o Non Pure?

1. **`f(x) { return x^2 + 5 }` - Pura**  
   Questa funzione è pura perché il suo output dipende solo dal valore di input `x` e non modifica alcuna variabile esterna né interagisce con l'ambiente esterno. Non ci sono effetti collaterali.

2. **`f(x) { return x + x2 }` - Non pura (x2 è una variabile libera)**  
   Questa funzione non è pura perché `x2` è una variabile libera, ovvero non è un parametro della funzione, ma proviene dall'ambiente esterno. Questo rende il comportamento della funzione dipendente da variabili esterne, violando i principi delle funzioni pure.

3. **`f(x) { x2 = x; return x + x2 }` - Non pura (modifica x2)**  
   In questo caso, la funzione modifica una variabile (`x2`), il che comporta un effetto collaterale. Le funzioni pure non dovrebbero alterare lo stato di variabili esterne o globali, quindi questa non è una funzione pura.

4. **`f(x) { read x2; return x + x2}` - Non pura (dipende dall'input utente)**  
   Questa funzione legge un valore (`x2`) dall'esterno, probabilmente dall'input dell'utente o da una risorsa esterna. Questo significa che l'output non è determinato solo dall'input `x`, ma anche da fattori esterni. Pertanto, non è pura.

5. **`f(x, x2) { return x + x2*x }` - Pura**  
   Questa funzione è pura perché dipende solo dai suoi parametri di input (`x` e `x2`) e non ha effetti collaterali né interagisce con l'ambiente esterno.

6. **`f(<x1,...,xn>, x2) { return <x1,...,xn>.push(x2)}` - Non pura (modifica l'array di input)**  
   In questo caso, la funzione modifica l'array di input (`<x1,...,xn>`) aggiungendo un nuovo elemento (`x2`). Questa modifica dell'input esterno costituisce un effetto collaterale, quindi la funzione non è pura.

### Codice Senza Effetti Collaterali
Le seguenti affermazioni sono equivalenti solo se la funzione `f()` è priva di effetti collaterali:

1. **`int result = f(x) + f(x);`**  
   Qui, `f(x)` viene chiamata due volte, e se `f(x)` è pura (cioè non ha effetti collaterali), entrambe le chiamate restituiranno lo stesso risultato. Se `f()` non è pura, la seconda chiamata potrebbe dare un risultato diverso.

2. **`int result = 2 * f(x);`**  
   In questo caso, `f(x)` viene chiamata una sola volta, e il risultato viene moltiplicato per 2. Se `f(x)` è pura, questo codice sarà equivalente al primo esempio, poiché `f(x)` produce sempre lo stesso risultato.

### Cittadini di Prima Classe
In programmazione funzionale, le **funzioni** sono considerate **cittadini di prima classe**, il che significa che possono essere utilizzate come valori ordinari. In particolare, le funzioni possono:
- **Essere memorizzate in variabili**: Una funzione può essere assegnata a una variabile per un uso successivo.
- **Essere passate come parametri**: Una funzione può essere passata come argomento a un'altra funzione.
- **Essere restituite come valori**: Una funzione può restituire un'altra funzione come risultato.
- **Essere raccolte in collezioni**: Le funzioni possono essere incluse in array, liste o altre strutture dati.

### Espressioni Lambda
Le **espressioni lambda** sono funzioni anonime, definite in modo conciso. La sintassi è `(parametri) -> espressione`. Le lambdas sono utilizzate spesso in contesti funzionali.
- **Esempio**: `(x) -> x * x`  
  Questa lambda prende un parametro `x` e restituisce il valore di `x` al quadrato.

### Funzioni di Ordine Superiore
Le **funzioni di ordine superiore** sono funzioni che accettano altre funzioni come argomenti o che restituiscono funzioni come risultato. Queste sono fondamentali nella programmazione funzionale per creare operazioni flessibili e modulari.

- **Esempi**:
  - **`map`**: Applica una funzione a ciascun elemento di una collezione, trasformando ogni elemento.
  - **`filter`**: Restituisce solo gli elementi di una collezione che soddisfano una condizione specifica.


## Esempi di Espressioni Lambda

```java
// Uso di lambda in forEach
numbers.forEach( (n) -> { System.out.println(n); } );

// Uso di Consumer per memorizzare una lambda
Consumer<Integer> method = (n) -> { System.out.println(n); };
numbers.forEach( method );

// Metodo che accetta una lambda come parametro
public static void printFormatted(String str, StringFunction format) {
    String result = format.run(str);
    System.out.println(result);
}

// Uso di Function di Java
public static void printFormatted(String str, Function<String,String> format) {
    String result = format.apply(str);
    System.out.println(result);
}
```

## Funzionale vs Imperativo

| Imperativo | Funzionale |
|------------|------------|
| Iterazione | Ricorsione |
| Permette mutazioni | Evita mutazioni |
| Usa memoria per memorizzare stato | Non ha stato |
| Non usa funzioni di ordine superiore | Usa funzioni di ordine superiore |
| Basato su macchina di Turing | Basato su lambda calcolo |

## Macchina di Turing vs Lambda Calcolo

- **Fondamento Teorico**:
  - Macchina di Turing: modello matematico per computer general-purpose
  - Lambda Calcolo: notazione matematica per esprimere funzioni e applicazione di funzioni

- **Modello di Calcolo**:
  - Macchina di Turing: calcolo imperativo, passo-passo
  - Lambda Calcolo: valutazione di espressioni come funzioni

- **Manipolazione Dati**:
  - Macchina di Turing: attraverso celle di memoria e operazioni di lettura/scrittura
  - Lambda Calcolo: rappresenta i dati come funzioni e usa astrazione e applicazione di funzioni

# Macchina di Turing vs Lambda Calcolo

## Paradigma di Programmazione:

- **Macchina di Turing**: I linguaggi basati su Turing rientrano spesso nel paradigma di programmazione imperativa o procedurale, dove l'attenzione è posta sulla definizione di passaggi espliciti per il computer da seguire.

- **Lambda Calcolo**: I linguaggi basati su lambda sono tipicamente associati alla programmazione funzionale, enfatizzando l'uso di funzioni pure, immutabilità e funzioni di ordine superiore.

## Espressività e Astrazione:

- **Macchina di Turing**: I linguaggi basati su Turing sono meno espressivi in termini di gestione di operazioni matematiche complesse e manipolazioni simboliche. Richiedono un controllo del flusso più esplicito.

- **Lambda Calcolo**: I linguaggi basati su lambda offrono un livello più alto di astrazione, rendendoli adatti per compiti che coinvolgono operazioni matematiche o simboliche. Spesso forniscono soluzioni più concise ed eleganti per determinati problemi.

In sintesi, la programmazione basata su macchina di Turing e quella basata su lambda calcolo rappresentano due approcci diversi al calcolo, con la prima più imperativa e passo-passo, e la seconda più dichiarativa e funzionale. La scelta tra le due dipende dal dominio del problema e dal paradigma di programmazione desiderato.

# Java Stream

## Streams

- **Stream**: una sequenza di elementi da una sorgente di dati che supporta operazioni aggregate.
- Gli stream operano su una sorgente di dati e la modificano, ad esempio:
  - Stampare ogni elemento di una collezione
  - Sommare ogni intero in un file
  - Concatenare stringhe in una stringa più grande
  - Trovare il valore più grande in una collezione

## Codice con e senza Stream in Java

```java
// Senza stream
public static int findSum(int[] array) {
    int sum = 0;
    for (int value : array) {
        sum += value;
    }
    return sum;
}

// Con stream
public static int findSumUsingStream(int[] array) {
    return Arrays.stream(array).sum();
}
```

## Funzioni di Ordine Superiore: Funzioni Core

- **Map (e FlatMap)**: Applica un'operazione su ogni elemento di una collezione
- **Filter**: Elimina elementi basati su un criterio
- **Reduce**: Applica una funzione sull'intera collezione

## Java Stream ed Effetti Collaterali

- Gli effetti collaterali dovrebbero essere evitati nelle operazioni intermedie
- Le operazioni terminali come `forEach()` e `peek()` possono produrre effetti collaterali

## Il modificatore map

```java
// Calcola la somma dei quadrati degli interi da 1 a 5
int sum = IntStream.range(1, 6)
    .map(n -> n * n)
    .sum();
```

## FlatMap vs Map

- **FlatMap** combina una funzione map e un'operazione di appiattimento
- **Map** applica solo una funzione allo stream senza appiattirlo

## Il modificatore filter

```java
// Calcola la somma dei quadrati degli interi dispari
int sum = IntStream.of(3, 1, 4, 1, 5, 9, 2, 6, 5, 3)
    .filter(n -> n % 2 != 0)
    .map(n -> n * n)
    .sum();
```

## Il modificatore reduce

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6);
int result = numbers
    .stream()
    .reduce(0, (subtotal, element) -> subtotal + element);
```

## Stream e metodi

```java
// Restituisce true se l'intero dato è primo
public static boolean isPrime(int n) {
    return IntStream.range(1, n + 1)
        .filter(x -> n % x == 0)
        .count() == 2;
}
```

---
# Operatori Stream

| Nome metodo      | Descrizione                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| **anyMatch(f)**  | Restituisce true se qualsiasi elemento dello stream soddisfa il predicato dato   |
| **allMatch(f)**  | Restituisce true se tutti gli elementi dello stream soddisfano il predicato dato |
| **average()**    | Restituisce la media aritmetica dei numeri nello stream                          |
| **collect(f)**   | Converte lo stream in una collezione e la restituisce                            |
| **count()**      | Restituisce il numero di elementi nello stream                                   |
| **distinct()**   | Restituisce gli elementi unici dallo stream                                      |
| **filter(f)**    | Restituisce gli elementi che soddisfano il predicato dato                        |
| **forEach(f)**   | Esegue un'azione su ogni elemento dello stream                                   |
| **limit(size)**  | Restituisce solo i prossimi size elementi dello stream                           |
| **map(f)**       | Applica la funzione data ad ogni elemento dello stream                           |
| **noneMatch(f)** | Restituisce true se zero elementi dello stream soddisfano il predicato dato      |
| **parallel()**   | Restituisce una versione multithread di questo stream                            |
| **peek(f)**      | Esamina solo il primo elemento dello stream                                      |
| **reduce(f)**    | Applica la funzione di riduzione binaria data agli elementi dello stream         |
| **sequential()** | Single-threaded, opposto di parallel()                                           |
| **skip(n)**      | Omette i prossimi n elementi dallo stream                                        |
| **sorted()**     | Restituisce gli elementi dello stream in ordine ordinato                         |
| **sum()**        | Restituisce la somma degli elementi nello stream                                 |
| **toArray()**    | Converte lo stream in array                                                      |

| Metodo statico    | Descrizione                                                     |
| ----------------- | --------------------------------------------------------------- |
| **concat(s1, s2)**    | Unisce due stream                                               |
| **empty()**           | Restituisce uno stream di zero elementi                         |
| **iterate(seed, f)**  | Restituisce uno stream infinito con l'elemento di partenza dato |
| **of(values)**        | Converte i valori dati in uno stream                            |
| **range(start, end)** | Restituisce un intervallo di valori interi come stream          |

# Risultati Opzionali

Alcuni terminatori di stream come `max` restituiscono un risultato "opzionale" perché lo stream potrebbe essere vuoto o non contenere il risultato:

```java
OptionalInt largest = IntStream.of(55, 20, 19, 31, 40, -2, 62, 30)
    .filter(n -> n % 10 == 0)
    .max();
int largestInt = largest.getAsInt();
```

# Stream e Array

Un array può essere convertito in uno stream con `Arrays.stream`:

```java
int[] numbers = {3, -4, 8, 4, -2, 17, 9, -10, 14, 6, -12};
int sum = Arrays.stream(numbers)
    .map(Math::abs)
    .filter(n -> n % 2 == 0)
    .sum();
```

# Riferimenti a Metodi

Un riferimento a metodo permette di passare un metodo dove altrimenti ci si aspetterebbe una lambda:

```java
.map(Math::abs)
.forEach(System.out::println);
```

# Stream e Liste

Una collezione può essere convertita in uno stream chiamando il suo metodo `stream()`:

```java
ArrayList<Integer> list = new ArrayList<Integer>();
list.add(-42);
list.add(-17);
list.add(68);

list.stream()
    .map(Function.identity())
    .forEach(System.out::println);
```

# Stream e Stringhe

```java
List<String> words = Arrays.asList("To", "be", "or", "Not", "to", "be");
Set<String> words2 = words.stream()
    .map(String::toLowerCase)
    .collect(Collectors.toSet());
System.out.println("word set = " + words2);
```

# Stream e File

```java
int longest = Files.lines(Paths.get("haiku.txt"))
    .map(x -> x.length())
    .max()
    .getAsInt();
```

# Stream e Map (HashMap)

```java
List<Employee> employeeList = new ArrayList<>(Arrays.asList(
    new Employee(1, "A", 100),
    new Employee(2, "A", 200),
    new Employee(3, "B", 300),
    new Employee(4, "B", 400),
    new Employee(5, "C", 500),
    new Employee(6, "C", 600)));

Map<Long, Employee> employeesMap = employeeList.stream()
    .collect(Collectors.toMap(x -> x.getId(), x -> x));
System.out.println(employeesMap);
```

# Interfacce Funzionali Java

| Interfaccia         | Input                                    | Output                            | Scopo                                                                       |
| ------------------- | ---------------------------------------- | --------------------------------- | --------------------------------------------------------------------------- |
| **Function**            | Singolo oggetto di qualsiasi tipo        | Singolo oggetto di qualsiasi tipo | Applicare logica sull'input, permettere concatenazione di logica            |
| **BiFunction**          | Due oggetti di qualsiasi tipo            | Singolo oggetto di qualsiasi tipo | Applicare logica su entrambi gli input, permettere concatenazione di logica |
| **Predicate**           | Singolo oggetto di qualsiasi tipo        | Boolean                           | Testare se il valore è conforme alla logica booleana                        |
| **Consumer/BiConsumer** | Singolo/doppio oggetto di qualsiasi tipo | Nessuno                           | Usare un valore e output, eseguire qualche effetto collaterale              |
| **Supplier**            | Nessuno                                  | Singolo oggetto di qualsiasi tipo | Creare un oggetto del tipo desiderato                                       |

# Interfaccia Function Java

```java
Function<String, Integer> findWordCount = in -> {
    return in.split(" ").length;
};
System.out.println(findWordCount.apply("this sentence has five words"));
```

# Concatenazione di funzioni

```java
Function<Integer, Integer> multipliedBy2 = x -> x * 2;
Function<Integer, Integer> squared = x -> x * x;
System.out.println(multipliedBy2.compose(squared).apply(4)); // Restituisce 32
System.out.println(multipliedBy2.andThen(squared).apply(4)); // Restituisce 64
```

# BiFunctions

```java
BiFunction<String, String, String> function1 = (s1, s2) -> {
    String s3 = s1 + s2;
    return s3;
};
System.out.println(function1.apply("Hello", " world!"));

BiFunction<Integer, Integer, Integer> function2 = (a, b) -> a * b;
System.out.println(function2.apply(3, 4));
```

---
# Interfaccia Predicate Java

L'interfaccia Predicate è usata per testare se un dato è conforme a qualche valore o logica.

Interfaccia:

```java
public interface Predicate<T> {
    boolean test(T t);
    Predicate<T> and(Predicate<? super T> other);
    Predicate<T> or(Predicate<? super T> other);
    Predicate<T> negate();
}
```

## Uso dell'Interfaccia Predicate

```java
String sentence = "this sentence has five words";
Predicate<String> specialWordChecker = in -> in.contains("five");
Predicate<String> sizeChecker = in -> in.length() > 50;

System.out.println(specialWordChecker.test(sentence));
System.out.println(sizeChecker.test(sentence));
System.out.println(sizeChecker.and(specialWordChecker).test(sentence));
System.out.println(sizeChecker.or(specialWordChecker).test(sentence));
System.out.println(sizeChecker.negate().test(sentence));
```

# Interfacce Consumer/Supplier Java

$$Consumer<T>: (BiConsumer<T,U>)$$
- Rappresenta un'operazione che accetta un singolo argomento di input e non restituisce alcun risultato.
- Funzione: `void accept(T t)` (`void accept(T t, U u)`)

--- 
$$Supplier<T>:$$

- Rappresenta un fornitore di risultati.
- Funzione: `T get()`

## Esempio Supplier/Consumer

```java
Supplier<Long> fibonacciSupplier = new Supplier<Long>() {
    long n1 = 1;
    long n2 = 1;
    @Override
    public Long get() {
        long fibonacci = n1;
        long n3 = n2 + n1;
        n1 = n2;
        n2 = n3;
        return fibonacci;
    }
};

Consumer<Long> tabConsumer = o -> System.out.print(o + "\t");

Stream.generate(fibonacciSupplier).limit(50).forEach(tabConsumer);
```

# Riutilizzo degli Stream Java

Gli stream Java, una volta consumati, non possono essere riutilizzati per impostazione predefinita.

```java
List<Integer> tokens = Arrays.asList(1, 2, 3, 4, 5);

// primo uso
Optional<Integer> result = tokens.stream().max(Integer::compareTo);
System.out.println(result.get());

// secondo uso
result = tokens.stream().min(Integer::compareTo);
System.out.println(result.get());

// terzo uso
long count = tokens.stream().count();
System.out.println(count);
```

### Caratteristiche degli Stream Java
- **Stream**: Struttura che elabora una collezione in stile funzionale senza modificarla.
- **Una sola elaborazione**: Lo stream può essere utilizzato una sola volta.
- **Conversione**: Qualsiasi tipo di collezione può essere trasformata in uno stream.
- **Creazione da zero**: Gli stream possono essere creati anche senza partire da una collezione.
### Stream Java sono Lazy
- **Efficienza**: Le operazioni come filtraggio, mapping e somma vengono eseguite in un unico passaggio, grazie alla **laziness**.
- **Ottimizzazione**: Lo stream esamina solo i dati necessari, evitando elaborazioni superflue.
### Pipeline degli Stream (Java 8)
- Una pipeline di stream è composta da:
  1. **Sorgente**: Punto di partenza dello stream.
  2. **Operazioni Intermedie**: 0 o più operazioni (es. `filter`, `map`).
  3. **Operazione Terminale**: Una sola operazione che conclude il flusso (es. `reduce`, `collect`).
# Creazione di Stream Java

```java
// Da Array
Stream<String> stream = Stream.of("this", "is", "a", "string", "stream", "in", "Java");

// Da Collections
List<String> list = new ArrayList<String>();
list.add("java"); list.add("php"); list.add("python");
Stream<String> stream = list.stream();

// Usando generate()
Stream<String> stream = Stream.generate(() -> "test").limit(10);

// Usando iterate()
Stream<BigInteger> bigIntStream = Stream.iterate(BigInteger.ZERO, n -> n.add(BigInteger.ONE));
bigIntStream.limit(100).forEach(System.out::println);

// Usando API
String sentence = "This is a six word sentence.";
Stream<String> wordStream = Pattern.compile("\\W").splitAsStream(sentence);
```

# 'Riutilizzo' degli Stream

Gli stream possono essere 'riutilizzati' avvolgendoli con un Supplier:

```java
Supplier<Stream<String>> streamSupplier = () -> Stream.of("d2", "a2", "b1", "b3", "c")
    .filter(s -> s.startsWith("a"));
System.out.println(streamSupplier.get().anyMatch(s -> true)); // true
System.out.println(streamSupplier.get().noneMatch(s -> true)); // false
```

# Caratteristiche delle Operazioni Intermedie

- Qualsiasi operazione è denotata come operazione intermedia se restituisce un nuovo Stream.
- Le operazioni intermedie sono sempre lazy.
- L'attraversamento della sorgente della pipeline non inizia finché non viene eseguita l'operazione terminale della pipeline.

# Operazioni Intermedie

| Operazione Stream | Scopo                                                                                | Input          |
| ----------------- | ------------------------------------------------------------------------------------ | -------------- |
| **filter**            | Filtra gli elementi secondo un predicato                                             | Predicate      |
| **map/flatMap**       | Applica una funzione sugli elementi dello stream                                     | Function       |
| **limit**             | Restituisce i primi n elementi dello stream                                          | int            |
| **sorted**            | Ordina gli elementi dello stream                                                     | Comparator     |
| **distinct**          | Scarta i duplicati usando il metodo equals                                           |                |
| **Reduce/Fold**       | Applica una funzione su tutti gli elementi dello stream                              | BinaryOperator |
| **peek**              | Restituisce una copia dello stream e poi applica una funzione Consumer sul risultato | Consumer       |
| **skip**              | Scarta i primi n elementi dello stream risultante                                    | Long           |

# Operazioni di Terminazione

Queste operazioni attraversano lo stream per produrre un risultato o un effetto collaterale.

- Dopo l'esecuzione dell'operazione terminale, la pipeline dello stream è considerata consumata e non può più essere utilizzata.
- Se è necessario attraversare nuovamente la stessa fonte di dati, si deve tornare alla fonte per ottenere un nuovo stream.

Esempi di operazioni di terminazione:
- `Stream.forEach`
- `IntStream.sum`

## Tabella delle Operazioni di Terminazione

| Operazione Stream           | Scopo                                                                                              | Input      |
| --------------------------- | -------------------------------------------------------------------------------------------------- | ---------- |
| **forEach**                     | Per ogni elemento, applica una funzione Consumer                                                   | Consumer   |
| **count**                       | Conta gli elementi dello stream                                                                    |            |
| **collect**                     | Riduce lo stream in una collezione desiderata                                                      | Collector  |
| **min/max**                     | Restituisce l'elemento min/max secondo un Comparator                                               | Comparator |
| **anyMatch/allMatch/noneMatch** | Restituisce se qualsiasi/tutti/nessuno degli elementi dello stream soddisfano il predicato fornito | Predicate  |
| **findFirst/findAny**           | Restituisce un Optional contenente il primo/qualsiasi risultato                                    |            |

# collect() usando Collectors

- Collector è un'operazione di riduzione
- Prende una sequenza di elementi di input e li combina in un unico risultato riassuntivo.
- Il risultato può essere una singola collezione o qualsiasi tipo di istanza di oggetto

Esempi di Collectors:
- `.collect(Collectors.toList())`: accumula gli elementi di input in una nuova List
- `.collect(Collectors.toCollection(ArrayList::new))`: accumula gli elementi di input in qualsiasi contenitore

## Parte dell'API Collectors

| Metodo | Descrizione |
|--------|-------------|
| Collectors.toList() | Inserisce gli elementi in una lista |
| Collectors.toCollection(TreeSet::new) | Inserisce gli elementi in un contenitore desiderato |
| Collectors.joining(",") | Unisce più elementi in un singolo elemento concatenandoli |
| Collectors.summingInt(item::getAge) | Somma i valori di ogni elemento con un fornitore dato |
| Collectors.groupingBy() | Raggruppa gli elementi con un classificatore e un mapper dati |
| Collectors.partitioningBy() | Partiziona gli elementi con un predicato e un mapper dati |

# Esempi Java

## Map
```java
String[] myArray = new String[]{"bob", "alice", "paul", "ellie"};
Stream<String> myStream = Arrays.stream(myArray);
Stream<String> myNewStream = myStream.map(s -> s.toUpperCase());
String[] myNewArray = myNewStream.toArray(String[]::new);
```

## Filter
```java
String[] myArray = new String[]{"bob", "alice", "paul", "ellie"};
String[] myNewArray = Arrays.stream(myArray)
    .filter(s -> s.length() > 4)
    .toArray(String[]::new);
```

## Reduce
```java
List<String> myArray = Arrays.asList("bob", "alice", "paul", "ellie");
System.out.println(myArray.stream()
    .map(s -> s.length())
    .reduce((a,b) -> Math.min(a, b))
    .get());
```

# Esempio di Stream Java

## Classe Student
```java
class Student {
    private String name;
    private Set<String> books;
    
    // ... metodi getter e setter ...
}
```

## Soluzione per ottenere la lista di libri distinti
```java
List<String> result = list.stream()
    .flatMap(x -> x.getBooks().stream())
    .distinct()
    .collect(Collectors.toList());
result.forEach(x -> System.out.println(x));
```

# Esempio: ITCompany

## Classe ComputerProgrammer
```java
class ComputerProgrammer {
    private String name;
    private String department;
    private int salary;
    
    // ... costruttore e metodi ...
}
```

## Esempi di operazioni

Stampare Map<name, salary>:
```java
company.stream()
    .collect(Collectors.toMap(ComputerProgrammer::getName, ComputerProgrammer::getSalary))
    .forEach((s, d) -> System.out.println("Department: " + s + "\tTotal Salary: " + d));
```

Trovare il costo totale del salario per ogni dipartimento:
```java
company.stream()
    .collect(Collectors.groupingBy(ComputerProgrammer::getDepartment,
        Collectors.summingDouble(ComputerProgrammer::getSalary)))
    .forEach((s, d) -> System.out.println("Department: " + s + "\tTotal Salary: " + d));
```

Stampare due gruppi (true, false) e le loro dimensioni:
```java
company.stream()
    .collect(Collectors.partitioningBy(e -> e.getSalary() < 3000,
        Collectors.counting()))
    .forEach((s, d) -> System.out.println("Group: " + s + "\tTotal Employees: " + d));
```

# Operazione peek()

- Restituisce uno stream composto dagli elementi di questo stream.
- Esegue anche l'azione fornita su ogni elemento dello stream risultante.
- Questa funzione è utilizzata principalmente per supportare il debugging.
- API: `Stream<T> peek(Consumer)`

Esempio:
```java
Stream.of("one", "two", "three", "four")
    .filter(e -> e.length() > 3)
    .peek(e -> System.out.println("Filtered value: " + e))
    .map(String::toUpperCase)
    .peek(e -> System.out.println("Mapped value: " + e))
    .collect(Collectors.toList());
```

### Interfaccia Optional in Java

- Un oggetto `Optional` può contenere o meno un valore non nullo.
- In alcuni casi, i filtri possono restituire "nessun risultato".
- L'uso di `Optional` permette di gestire in modo sicuro l'eventualità che un valore non venga restituito.
- I controlli espliciti di `null` rendono il codice meno pulito e sono generalmente sconsigliati.
- Java 8 introduce `Optional` per sostituire i controlli `null`, incoraggiando una best practice più leggibile e sicura.

Esempio:
```java
List<String> myArray = Arrays.asList("bob", "alice", "paul", "ellie");
Optional<String> result = myArray.stream()
    .filter(s -> s.length() > 4)
    .findFirst();
if (result.isPresent())
    System.out.println(result.get());
```

# Ordinamento

```java
company.stream().sorted(Comparator.comparingInt(ComputerProgrammer::getSalary))
    .forEach(System.out::println);

company.stream().sorted((e1, e2) -> e1.getSalary()-e2.getSalary())
    .forEach(System.out::println);

company.stream().sorted()
    .forEach(System.out::println);

company.stream().sorted(Comparator.reverseOrder())
    .forEach(System.out::println);
```

# Stream.parallel

```java
public static void main(String[] args) {
    long start = System.currentTimeMillis();
    long limit = 500_000;
    long count = Stream.iterate(0, n -> n + 1).limit(limit)
        .parallel() //con parallel 37 secondi, senza 101 secondi
        .filter(c01Parallel::isPrime).peek(x -> System.out.println(x)).count();
    System.out.println("\n Il numero di numeri primi tra 0 e "+limit+" è " + count);
    long end = System.currentTimeMillis();
    System.out.println("Secondi : " + ((end - start) / 1000));
}

public static boolean isPrime(int number) {
    if (number <= 1)
        return false;
    return !IntStream.rangeClosed(2, number / 2).anyMatch(i -> number % i == 0);
}
```

# Classe ComputerProgrammer

```java
public class ComputerProgrammer implements Comparable<ComputerProgrammer> {
    private int id;
    private String name;
    private String surname;
    private String department;
    private int salary;
    private Set<String> skills;
    // ...
}
```

# Query su IT Company

1. Stampa i primi 100 dipendenti:
```java
Files.lines(new File("ITCompany_10000.txt").toPath())
    .map(x -> ComputerProgrammer.parse(x))
    .limit(100)
    .forEach(x -> System.out.println(x));
```

### Programmazione Funzionale ≠ Stream

- La programmazione funzionale non si riduce all'uso degli Stream!
- Essa rappresenta un paradigma molto più ampio rispetto agli Stream.
- Gli Stream implementano tecniche funzionali, ma sono solo una parte di ciò che offre la programmazione funzionale.

I principi chiave della programmazione funzionale includono:
- **Evitare lo stato condiviso**: ridurre i conflitti tra processi.
- **Evitare la mutazione**: preferire dati immutabili.
- **Costruire il software tramite funzioni**, idealmente **funzioni pure** (senza effetti collaterali).
- **Ricorsione** al posto dell'iterazione.
- **Calcolo lazy** (ritardato) anziché eager (immediato).
# Supporto Java alla Programmazione Funzionale

| Caratteristica | Linguaggi Funzionali | Java Object Oriented | Java 8 |
|----------------|----------------------|----------------------|--------|
| Funzioni | Indipendenti | Legate all'oggetto della classe, "metodi" | Usa interfacce funzionali |
| Stato dei valori | Immutabile | Mutabile | Immutabile se richiesto |
| Concatenazione di funzioni | Sì | Solo se nella stessa istanza | Le lambda permettono la concatenazione |
| Supporto alla concorrenza e multicore | Sì | Richiede strumenti di sincronizzazione | Sì - se puramente funzionale |

## Limitazioni di Java per la Programmazione Funzionale

- Differenze tra tipi base e oggetti
- Distinzione tra array e liste
- Gestione non facile di tuple (coppie, triple di oggetti anche eterogenei)
- Eccessiva verbosità
- Gestione non facile di mappe (chiave/valore)
- Combinazione non facile di valori per creare una lista
