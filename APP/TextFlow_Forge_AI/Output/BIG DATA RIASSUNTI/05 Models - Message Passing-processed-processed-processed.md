
## Il Modello di Passaggio di Messaggi: Un Riassunto

Il modello di passaggio di messaggi è un paradigma di comunicazione inter-processo (IPC) in sistemi distribuiti, dove ogni processo possiede una memoria privata.  A differenza del modello a memoria condivisa, che utilizza uno spazio di indirizzi condiviso per la comunicazione, il passaggio di messaggi si basa sullo scambio esplicito di messaggi tra processi indipendenti.  Questo rende il modello più flessibile in ambienti distribuiti, sebbene possa introdurre un maggiore overhead di comunicazione. ![[|469](_page_3_Figure_3.jpeg)]

Le primitive fondamentali sono:

* `Send(destinazione, messaggio)`: Invia un messaggio a un processo specifico.
* `Receive(sorgente, messaggio)`: Riceve un messaggio da un processo specifico.


Le implementazioni del passaggio di messaggi variano in base a diversi fattori:

* **Direzione:** Il *passaggio di messaggi diretto* stabilisce un collegamento diretto tra mittente e ricevente, mentre il *passaggio di messaggi indiretto* utilizza mailbox o porte, permettendo maggiore flessibilità e modularità.  Nel passaggio diretto, la modifica dell'identità di un processo richiede l'aggiornamento di tutti i processi collegati.  Nel passaggio indiretto, una porta può essere riassegnata a diversi processi.

* **Bufferizzazione:** Può essere *bufferizzato* (con una coda di messaggi) o *non bufferizzato* (senza coda).

* **Blocco:** Può essere *bloccante* (sincrono), dove il processo chiamante si blocca fino al completamento dell'operazione di invio o ricezione, o *non bloccante* (asincrono), dove il processo chiamante continua l'esecuzione immediatamente. ![[|335](_page_5_Figure_6.jpeg)]

---

## Passaggio di Messaggi in Sistemi Distribuiti

Questo documento descrive i diversi modelli di passaggio di messaggi in sistemi distribuiti, focalizzandosi su aspetti di blocco/non blocco, buffering e comunicazione di gruppo.

### Modalità di Invio e Ricezione

Le operazioni di invio e ricezione possono essere *bloccanti* (synchronous) o *non bloccanti* (asynchronous).

* **Invio/Ricezione Bloccante:** Il mittente (invio) o il ricevente (ricezione) attendono il completamento dell'operazione prima di proseguire.  L'invio/ricezione bloccante è una comunicazione *rendez-vous*.

* **Invio/Ricezione Non Bloccante:** Il mittente (invio) continua l'esecuzione senza attendere la conferma, mentre il ricevente (ricezione) può ricevere un messaggio valido o nullo.  La combinazione più comune è invio non bloccante e ricezione bloccante.

### Buffering

Il buffering influenza il comportamento del passaggio di messaggi:

* **Coda a capacità zero:** Richiede un rendez-vous; il mittente blocca finché il ricevente non è pronto.
* **Coda limitata:** Il mittente blocca se la coda (di dimensione *n*) è piena.
* **Coda illimitata:** Il mittente non blocca mai, ma presenta rischi di esaurimento delle risorse.

### Comunicazione di Gruppo

La comunicazione di gruppo offre efficienza e semplicità in applicazioni distribuite parallele, con tre tipologie principali:

* **Uno-a-Molti (Multicast):** Un mittente invia a più ricevitori. I gruppi possono essere *chiusi* (solo membri interni possono inviare) o *aperti* (qualsiasi processo può inviare). Un caso speciale è il *broadcast*, dove il messaggio è inviato a tutti i processori.

* **Molti-a-Uno:** Più mittenti inviano a un singolo ricevitore, che può essere *selettivo* (sceglie il mittente) o *non selettivo* (accetta da un set predefinito). Il non determinismo nella ricezione rappresenta una sfida.

* **Molti-a-Molti:** Più mittenti inviano a più ricevitori, offrendo flessibilità e coordinamento decentralizzato.

---

La consegna ordinata dei messaggi è fondamentale nelle comunicazioni molti-a-molti.  È essenziale che tutti i messaggi arrivino ai destinatari nel corretto ordine per garantire il corretto funzionamento delle applicazioni che si basano su questo tipo di comunicazione.  L'ordine di consegna, infatti, è un requisito funzionale per molte applicazioni.

---
