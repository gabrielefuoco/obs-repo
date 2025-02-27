
# Modelli di Programmazione: MapReduce 

Un **modello di programmazione** è un'interfaccia che separa le proprietà di alto livello da quelle di basso livello, fornendo operazioni specifiche al livello di programmazione superiore e richiedendo implementazioni per tutte le operazioni al livello architettonico inferiore (Skillicorn e Talia, 1998). Un **modello di programmazione parallela** è un'astrazione per un'architettura di computer parallela che aiuta nell'espressione di algoritmi e applicazioni parallele. Può rappresentare una varietà di problemi per diversi sistemi paralleli e distribuiti.

I **modelli di programmazione parallela** sono spesso la caratteristica principale dei framework big data, poiché influenzano il paradigma di esecuzione dei motori di elaborazione big data e il modo in cui gli utenti progettano e creano applicazioni. Consentono la separazione delle problematiche di sviluppo software dalle problematiche di esecuzione parallela, fornendo *astrazione* e *stabilità*. L'*astrazione* è garantita perché le operazioni del modello sono a un livello superiore rispetto a quelle delle architetture sottostanti. Ciò semplifica la struttura del software e la difficoltà del suo sviluppo, garantendo anche la **stabilità** attraverso un'interfaccia standard. Pertanto, il modello può ridurre lo sforzo di implementazione, prendendo decisioni una sola volta per ogni sistema di destinazione, anziché per ognuno. I **modelli di programmazione** si distinguono in base al loro livello di **astrazione**, consentendo l'espressione di meccanismi di programmazione di alto e basso livello. I **modelli scalabili di alto livello** consentono ai programmatori di specificare la logica dell'applicazione nascondendo i dettagli di basso livello, affidandosi ai compilatori per l'ottimizzazione. I **modelli scalabili di basso livello** consentono l'interazione diretta con le unità di calcolo e di storage, consentendo una specifica precisa del parallelismo dell'applicazione.

I **sistemi di programmazione** sono implementazioni di uno o più modelli e possono essere sviluppati attraverso diverse strategie:

* **Sviluppo del linguaggio:** comporta la creazione di nuovi linguaggi di programmazione paralleli o l'integrazione di costrutti paralleli e strutture dati nei linguaggi esistenti.
* **Approccio mediante annotazioni:** utilizza simboli o parole chiave specifiche nelle annotazioni per identificare le istruzioni parallele nel codice del programma e indicare al compilatore quali istruzioni devono essere eseguite contemporaneamente.
* **Integrazione della libreria:** questo approccio prevede il miglioramento del parallelismo includendo librerie nel codice dell'applicazione, ed è l'approccio più popolare poiché è ortogonale ai linguaggi host.

Modelli di programmazione, come MapReduce e il passaggio di messaggi, forniscono astrazione per la programmazione parallela. Sistemi di programmazione come Apache Hadoop e MPI supportano questi modelli, soddisfacendo una vasta gamma di applicazioni big data e livelli di competenza degli utenti. Data la vasta gamma di applicazioni big data e categorie di utenti, sono stati proposti diversi modelli di programmazione parallela, che spaziano su vari livelli di astrazione (alto e basso).

## Il modello MapReduce

Il **modello MapReduce** è stato sviluppato da Google per affrontare la sfida dell'elaborazione efficace dei big data. Il suo paradigma si ispira alle funzioni **map** e **reduce** disponibili nei linguaggi di programmazione funzionali, come LISP, e consente ai progettisti di creare applicazioni distribuite basate su queste due operazioni. Il modello MapReduce sfrutta ampiamente la strategia **dividi e conquista** per affrontare i problemi relativi ai big data:

1. Dividere il problema in sottoproblemi più piccoli.
2. Eseguire sottoproblemi indipendenti in parallelo utilizzando diversi worker.
3. Combinare i risultati intermedi da ogni singolo worker.

Il programmatore definisce due fasi per l'elaborazione MapReduce: *map* e *reduce*.

La funzione **map** accetta una coppia (chiave, valore) come input e produce un elenco di coppie intermedie (chiave, valore):

#### **map (k1, v1) → list(k2, v2)**

La funzione **reduce** unisce tutti i valori intermedi con la stessa chiave intermedia:

#### **reduce (k2, list(v2)) → list(v3)**

## Parallelismo in MapReduce

Il parallelismo viene raggiunto in entrambe le fasi:

* Nella fase **map**, dove le chiavi possono essere elaborate contemporaneamente da computer diversi (le chiamate map vengono distribuite tra i computer mediante lo sharding dei dati di input).
* Nella fase **reduce**, dove i reducer che lavorano su chiavi distinte possono essere eseguiti contemporaneamente.

Di conseguenza, gli algoritmi MapReduce scalano da un singolo server a centinaia di migliaia di server. L'approccio MapReduce nasconde i dettagli della parallelizzazione sottostante al programmatore, rendendolo semplice da utilizzare.

## Sviluppo di Applicazioni MapReduce

Gli sviluppatori possono concentrarsi sulla definizione dei calcoli, senza addentrarsi nei dettagli di come vengono eseguiti o di come i dati vengono inviati ai processori.

![[Pasted image 20250223161820.png|480]]
### Esempio: Indice Invertito

Un esempio di applicazione MapReduce è la generazione di un **indice invertito**. Dato un insieme di documenti, questo indice contiene un insieme di parole (o termini di indice), specificando gli ID di tutti i documenti che contengono ciascuna parola. Un approccio MapReduce può essere efficacemente sfruttato in questo caso:

* la funzione **map** genera una sequenza di coppie `<parola, documentID>` per ogni documento di input.
* la funzione **reduce** prende tutte le coppie per una data parola, ordina i corrispondenti ID dei documenti ed emette una coppia `<parola, lista(documentID)>`.

L' *indice invertito* per i documenti di input è formato dall'insieme di tutte le coppie di output create dalla funzione reduce:

## Fase Map e Reduce

Un **job** è un programma MapReduce costituito dal codice per le fasi map e reduce, dalle impostazioni di configurazione (ad esempio, dove devono essere memorizzati i dati di output) e dal dataset di input, che è memorizzato su un file system distribuito. Ogni job MapReduce è diviso in unità più piccole note come **task**:

* I task Map sono chiamati **mapper**
* I task Reduce sono chiamati **reducer**

Per definire applicazioni complesse che non possono essere scritte come un singolo job MapReduce, gli utenti potrebbero dover comporre workflow di job MapReduce, che comportano più round di operazioni map e reduce.

I sistemi MapReduce attuali, come Apache Hadoop, seguono il modello **master-worker**:

* Un nodo **utente** invia un job a un nodo **master**, che identifica i **worker** inattivi e assegna a ciascuno di essi un task map o reduce.
* Il master coordina l'intero flusso di job MapReduce, gestendo sia i task map che quelli reduce.
* Una volta completati tutti i task, il nodo master fornisce il risultato al nodo utente.

L'elaborazione in un'applicazione MapReduce può essere descritta come segue:

1. Un **job descriptor** viene inviato a un processo master, descrivendo il task MapReduce da eseguire e altre informazioni, come la posizione dei dati di input.
2. Il **master** avvia diversi processi **mapper** e **reducer** su macchine diverse in base al descriptor. Distribuisce anche i dati di input, suddivisi in più chunk, a diversi mapper.
3. Ogni processo mapper utilizza la funzione **map** (definita nel job descriptor) per creare un elenco di coppie **intermedie (chiave, valore)** dopo aver ricevuto il suo chunk di dati.
4. Lo stesso processo **reducer** viene allocato a tutte le coppie con le stesse chiavi. Ogni reducer esegue la funzione **reduce** (specificata dal job descriptor), che unisce tutti i dati con la stessa chiave per produrre un set più piccolo di valori.
5. Gli output di ogni processo reducer vengono quindi raccolti e inviati alla posizione specificata dal job descriptor, formando i dati di output finali.

![[](_page_15_Figure_4.jpeg)]

### Fase Combine

Per aumentare la velocità, è possibile eseguire un passaggio di combinazione, che prevede una fase di minireduce sull'output map locale, che aggrega i dati prima di trasmetterli ai reducer sulla rete. Un **combiner** viene utilizzato per aggregare l'output map locale:

**combine (k2, list(v2)) → list(v3)**

In molti casi, la stessa funzione può essere utilizzata sia per la combinazione che per la riduzione finale, con i vantaggi di ridurre la quantità di dati intermedi e il traffico di rete.

### Fase Shuffle e Sort

Tra le fasi map (con combine) e reduce, avviene un'operazione *group-by* distribuita implicita, denominata **shuffle e sort**. Questa operazione trasferisce l'output del mapper ai reducer, unendo e ordinando i dati per chiave prima di raggiungere ogni reducer. Le **chiavi intermedie**, non memorizzate sul file system distribuito, vengono scritte sul disco locale di ogni computer nel cluster.

Dopo che un mapper completa i suoi file di output ordinati, lo scheduler di MapReduce avvisa i reducer, invitandoli a recuperare le coppie (chiave, valore) ordinate per le loro partizioni dai rispettivi mapper.

