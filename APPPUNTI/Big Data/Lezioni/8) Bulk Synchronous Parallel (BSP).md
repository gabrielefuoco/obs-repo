
| **Termine**                 | **Definizione**                                                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **BSP**                     | Bulk Synchronous Parallel, un modello di calcolo parallelo.                                                                          |
| **Superstep**               | Un'iterazione nel modello di programmazione BSP, composta da fasi di calcolo, comunicazione e sincronizzazione.                      |
| **PE (Processing Element)** | Un'unità di elaborazione in una macchina BSP.                                                                                        |
| **Router**                  | Un componente di una macchina BSP responsabile dell'instradamento dei messaggi tra le PE.                                            |
| **Sincronizzazione**        | Il processo di coordinamento delle PE in modo che raggiungano un punto specifico nell'esecuzione del programma contemporaneamente.   |
| **Memoria locale**          | La memoria direttamente accessibile da una singola PE.                                                                               |
| **Memoria globale**         | La memoria condivisa tra tutte le PE in una macchina BSP.                                                                            |
| **Funzione di hash**        | Una funzione che mappa i dati in un intervallo di valori, utilizzata in BSP per distribuire i dati nella memoria globale.            |
| **h**                       | Il numero massimo di messaggi che una PE può inviare o ricevere in un superstep.                                                     |
| **g**                       | La capacità di comunicazione della rete, che rappresenta il tempo necessario a un processore per inviare h messaggi di dimensione 1. |
| **l**                       | Il costo di una barriera di sincronizzazione.                                                                                        |
| **w**                       | Il costo massimo dell'elaborazione locale in un superstep.                                                                           |

---

- Il modello Bulk Synchronous Parallel (BSP) è stato sviluppato da Leslie Valiant dell'Università di Harvard a partire dalla fine degli anni '80.
- L'interesse per questo modello è cresciuto a partire dal 2010, quando Google l'ha adottato come una delle principali tecnologie per l'analisi dei grafi su larga scala (Google Pregel).

## Architettura di una macchina BSP
Tre elementi fondamentali:
1. Un insieme di p unità di elaborazione (PE) responsabili dell'elaborazione.
2. Un router responsabile della comunicazione. Si occupa di instradare i messaggi tra i diversi componenti, gestendo tale comunicazione punto-a-punto tra coppie di macchine. Si assume non siano presenti strumenti per la combinazione, duplicazione o broadcasting dei messaggi.
3. Meccanismi e strumenti per la sincronizzazione globale (per barriere) dei componenti (o di un loro sottoinsieme).

- Ogni unità di elaborazione (PE) ha una propria memoria.
- È presente anche un blocco responsabile della sincronizzazione, collegato ad un router.

### Rappresentazione interna di un PE
- Ogni PE è munita di un processore che opera su una memoria locale .
- La memoria globale è distribuita secondo una funzione di hash degli indirizzi.
- Per migliorare le prestazioni, l'operazione di hashing può essere effettuata in hardware a livello dell'interfaccia del router.

**Modello alternativo:** Un'alternativa al modello precedente distingue le unità di elaborazione (CE) da quelle di memorizzazione (ME).

## Modello di programmazione
L'elaborazione è composta da iterazioni successive, dette superstep.
Ogni superstep si compone di tre fasi:
1. **Computation**: Un insieme di elaborazioni locali su ciascun processore, effettuate ognuna sulla propria memoria locale.
2. **Communication**: Un insieme di operazione di comunicazione in cui vi è il trasferimento di dati tra processi.
3. **Synchronization**: Consiste nell'attendere la conclusione di tutte le operazioni di comunicazione; dopo la sincronizzazione, i dati sono visibili ai destinatari per l'inizio del superstep successivo.

## Modello di comunicazione
- Il numero massimo di messaggi in ingresso o in uscita per un superstep è indicato con h.
- La capacità di una rete di comunicazione di consegnare dati è caratterizzata da un parametro g, definito in modo tale che è necessario un tempo hg ad un processore per inviare h messaggi di dimensione 1.
- Il modello BSP non fa distinzione, in termini di costo, tra l'invio di 1 messaggio di lunghezza m o l'invio di m messaggi di lunghezza 1 (in entrambi i casi il costo è uguale a mg).

Il parametro g dipende da diversi fattori:
- I protocolli utilizzati per interagire all'interno della rete di comunicazione.
- Gestione del buffer da parte sia dei processori che della rete di comunicazione.
- La strategia di routing utilizzata nella rete.
- Il sistema di runtime BSP.

## Costo di un superstep
Il costo di un superstep è la somma di tre termini:
1. Il costo dell'elaborazione locale di maggior durata
2. Il costo della comunicazione globale tra i processori
3. Il costo della sincronizzazione alla fine del superstep

Il costo di un superstep per p processori è:
$$max^p_{i=1}(w_i)+max^p_{i=1}(h_i g)+l$$
dove:
- $w_i$ è il costo per l'elaborazione locale nel processo i
- $h_i$ è il numero di messaggi inviati o ricevuti dal processo i
- $l$ è il costo della barriera di sincronizzazione

Per maggiore semplicità, si può usare in alternativa la seguente espressione:
$$w+hg+l$$
dove w e h sono valori massimi

## Costo di un algoritmo BSP

Il costo di un algoritmo BSP è la somma dei costi di ciascun superstep:
$$W+Hh+Sl=\sum^S_{s=1}w_s+g\sum^S_{s=1}h_s+Sl$$
dove S è il numero di superstep.

---
## FAQ sul Modello Bulk Synchronous Parallel (BSP)

**1. Cos'è il modello Bulk Synchronous Parallel (BSP)?**

Il modello BSP, sviluppato da Leslie Valiant alla fine degli anni '80, è un modello di programmazione parallela che semplifica la progettazione di algoritmi per sistemi con memoria distribuita. Invece di concentrarsi sui dettagli di basso livello della comunicazione tra i processori, BSP suddivide l'esecuzione di un programma in iterazioni chiamate "superstep", ognuna composta da fasi di calcolo locale, comunicazione e sincronizzazione.

**2. Quali sono i componenti principali di un'architettura BSP?**

Un'architettura BSP si compone di tre elementi fondamentali:

- Un insieme di unità di elaborazione (PE), ognuna con la propria memoria locale, responsabili dell'esecuzione dei calcoli.
- Un router che gestisce la comunicazione punto-a-punto tra le PE, instradando i messaggi senza combinarli o duplicarli.
- Meccanismi di sincronizzazione globale, come le barriere, per coordinare le PE all'interno di un superstep.

**3. Come funziona un superstep nel modello BSP?**

Un superstep è l'unità di base di elaborazione in BSP e comprende tre fasi distinte:

1. **Calcolo:** Ogni PE esegue operazioni locali sui dati nella propria memoria.
2. **Comunicazione:** Le PE si scambiano dati inviando e ricevendo messaggi attraverso il router.
3. **Sincronizzazione:** Tutte le PE si sincronizzano tramite una barriera, garantendo che i dati inviati durante la fase di comunicazione siano disponibili ai destinatari prima dell'inizio del superstep successivo.

**4. Quali parametri influenzano il costo della comunicazione in BSP?**

Il parametro principale è _h_, il numero massimo di messaggi inviati o ricevuti da una PE in un superstep. La capacità della rete di consegnare i dati è rappresentata da _g_, il tempo necessario a una PE per inviare _h_ messaggi di dimensione 1. Il costo totale della comunicazione è proporzionale a _hg_.

**5. Come si calcola il costo di un singolo superstep?**

Il costo di un superstep tiene conto di tre fattori: il tempo massimo di calcolo locale (_w_), il costo massimo di comunicazione (_hg_) e il costo della barriera di sincronizzazione (_l_). In forma semplificata, il costo di un superstep è: _w + hg + l_.

**6. In che modo il modello BSP semplifica la progettazione di algoritmi paralleli?**

BSP astrae i dettagli di basso livello della comunicazione e della sincronizzazione, permettendo agli sviluppatori di concentrarsi sulla scomposizione del problema in superstep e sulla minimizzazione del numero di superstep necessari.

**7. Quali sono i vantaggi dell'utilizzo del modello BSP?**

BSP offre diversi vantaggi, tra cui:

- **Portabilità:** Gli algoritmi BSP possono essere eseguiti su diverse architetture parallele con modifiche minime.
- **Prevedibilità:** Il modello offre una stima precisa del costo computazionale degli algoritmi.
- **Scalabilità:** BSP si adatta bene a sistemi con un elevato numero di processori.

**8. Quali sono alcuni esempi di applicazioni che utilizzano il modello BSP?**

BSP è stato adottato in diverse applicazioni, tra cui:

- **Analisi di grafi su larga scala (es. Google Pregel)**
- **Algebra lineare**
- **Data mining**
- **Bioinformatica**

## Quiz

**Istruzioni:** Rispondi alle seguenti domande in modo conciso (2-3 frasi).

1. Descrivere brevemente l'origine e l'importanza del modello BSP.
2. Quali sono i tre elementi fondamentali dell'architettura di una macchina BSP?
3. Spiegare la differenza tra memoria locale e memoria globale in una macchina BSP.
4. Qual è lo scopo della funzione di hash nella rappresentazione interna di una PE?
5. Descrivere le tre fasi di un superstep nel modello di programmazione BSP.
6. Cosa rappresentano i parametri h e g nel modello di comunicazione BSP?
7. Da quali fattori dipende il parametro g?
8. Quali sono i tre componenti del costo di un superstep?
9. Scrivere la formula per il costo di un superstep per p processori.
10. Come si calcola il costo totale di un algoritmo BSP?

## Risposte al Quiz

1. Il modello BSP è stato sviluppato da Leslie Valiant alla fine degli anni '80. Ha guadagnato popolarità negli anni 2010 quando Google lo ha adottato per l'analisi dei grafi su larga scala (Google Pregel).
2. I tre elementi fondamentali di una macchina BSP sono: un insieme di unità di elaborazione (PE), un router per la comunicazione tra le PE e meccanismi per la sincronizzazione globale.
3. La memoria locale è specifica di ogni PE e accessibile solo da quella PE. La memoria globale è logicamente condivisa tra tutte le PE e accessibile tramite una funzione di hash.
4. La funzione di hash viene utilizzata per determinare la posizione di un dato nella memoria globale. In questo modo si distribuiscono i dati tra le PE e si accede alla memoria globale in modo efficiente.
5. Un superstep è composto dalle fasi di calcolo, comunicazione e sincronizzazione. Durante la fase di calcolo, le PE eseguono operazioni sui dati locali. Durante la fase di comunicazione, le PE si scambiano dati tramite il router. La fase di sincronizzazione garantisce che tutte le PE abbiano completato le operazioni di comunicazione prima di procedere al superstep successivo.
6. Il parametro h rappresenta il numero massimo di messaggi che una PE può inviare o ricevere in un superstep. Il parametro g rappresenta la capacità di comunicazione della rete, misurata come il tempo necessario a un processore per inviare h messaggi di dimensione 1.
7. Il parametro g dipende da diversi fattori, tra cui i protocolli di rete, la gestione del buffer, la strategia di routing e il sistema di runtime BSP.
8. I tre componenti del costo di un superstep sono: il costo massimo dell'elaborazione locale (w), il costo della comunicazione globale (hg) e il costo della sincronizzazione (l).
9. Il costo di un superstep per p processori è: $max^p_{i=1}(w_i) + max^p_{i=1}(h_i g) + l$, dove w_i è il costo dell'elaborazione locale nel processo i e $h_i$ è il numero di messaggi inviati o ricevuti dal processo i.
10. Il costo totale di un algoritmo BSP è la somma dei costi di tutti i superstep: $W + Hh + Sl = Σ^S_{s=1} w_s + gΣ^S_{s=1} h_s + Sl$, dove S è il numero di superstep.

