
**Schema Riassuntivo: Modello di Passaggio di Messaggi**

**1. Introduzione al Modello di Passaggio di Messaggi**
    *   Definizione: Paradigma IPC in informatica distribuita dove ogni processo ha memoria privata.
    *   Meccanismi IPC:
        *   Memoria Condivisa
        *   Memoria Distribuita / Passaggio di Messaggi

**2. Confronto tra Modelli di Memoria Condivisa e Passaggio di Messaggi**
    *   **Modello a Memoria Condivisa:**
        *   Processi accedono a uno spazio di indirizzi condiviso.
        *   Comunicazione tramite condivisione diretta di variabili.
        *   Vantaggi: Comunicazione più veloce.
        *   Svantaggi: Richiede meccanismi di sincronizzazione.
    *   **Modello a Passaggio di Messaggi:**
        *   Processi indipendenti con memoria locale.
        *   Comunicazione tramite scambio di messaggi.
        *   Vantaggi: Flessibile in sistemi distribuiti.
        *   Svantaggi: Maggiore overhead di comunicazione.
    *   Distinzione Chiave:
        *   Memoria Condivisa: Spazio di indirizzi comune.
        *   Passaggio di Messaggi: Scambio esplicito di messaggi.

**3. Primitive di Passaggio di Messaggi**
    *   `Send(destinazione, messaggio)`: Invia un messaggio a un processo (destinazione).
    *   `Receive(sorgente, messaggio)`: Riceve un messaggio da un processo (sorgente).
    *   Processo di Invio: Crea e trasmette il messaggio.
    *   Processo di Ricezione: Indica la disponibilità a ricevere.

**4. Implementazioni del Passaggio di Messaggi**
    *   Categorie:
        *   Dirette vs. Indirette
        *   Bufferizzate vs. Non Bufferizzate
        *   Bloccanti vs. Non Bloccanti

**5. Passaggio di Messaggi Diretto e Indiretto**
    *   **Passaggio di Messaggi Diretto:**
        *   Collegamento diretto tra processi.
        *   Identità del ricevente nota.
        *   Svantaggi: Manca di modularità (modifica identità richiede aggiornamenti).
    *   **Passaggio di Messaggi Indiretto:**
        *   Utilizzo di mailbox/porte per la consegna.
        *   Porte possono essere riassegnate.
        *   Mittente non conosce il ricevente.
        *   Vantaggi: Collegamenti multi-processo, maggiore flessibilità.

**6. Passaggio di Messaggi Bloccante e Non Bloccante**
    *   Distinzione: Sincrono (bloccante) vs. Asincrono (non bloccante).

---

**I. Messaggi di Blocco e Non di Blocco**

*   **A. Operazioni Bloccanti (Sincrone):**
    *   Il mittente attende la conferma di ricezione dal ricevente (invio bloccante).
    *   Il ricevente attende un messaggio prima di procedere (ricezione bloccante).
*   **B. Operazioni Non Bloccanti (Asincrone):**
    *   Il mittente continua senza attendere la conferma (invio non bloccante), ma si aspetta una conferma in caso di fallimento.
    *   Il ricevente può accettare un messaggio valido o nullo (ricezione non bloccante).
        *   Sfida: determinare se un messaggio è arrivato.
        *   Rischio: attesa indefinita in caso di fallimento continuo.
*   **C. Combinazioni Raccomandate:**
    *   Invio bloccante e ricezione bloccante (Rendez-vous).
    *   Invio non bloccante e ricezione non bloccante.
    *   Invio non bloccante e ricezione bloccante (più utilizzata).

**II. Buffering**

*   **A. Coda a Capacità Zero (Nessuna Coda):**
    *   Richiede un rendez-vous (il mittente attende che il ricevente sia pronto).
*   **B. Coda Limitata:**
    *   Limitata a *n* messaggi/byte.
    *   Il mittente si blocca quando la coda è piena.
*   **C. Coda Illimitata:**
    *   I mittenti procedono senza attendere.
    *   Rischio: risorse fisiche limitate.

**III. Comunicazione di Gruppo**

*   **A. Comunicazione Uno-a-Molti (Multicast):**
    *   Un mittente trasmette a più ricevitori.
    *   Tipi di Gruppo:
        *   Gruppo Chiuso: solo i membri possono inviare messaggi internamente.
        *   Gruppo Aperto: qualsiasi processo può inviare messaggi al gruppo.
    *   Caso Speciale: Broadcast (messaggio inviato a tutti i processori).
*   **B. Comunicazione Molti-a-Uno:**
    *   Più mittenti trasmettono a un ricevitore.
    *   Tipi di Ricevitore:
        *   Selettivo: identifica un mittente specifico.
        *   Non Selettivo: risponde a qualsiasi mittente da un set predefinito.
    *   Sfida: Non determinismo (incertezza su quale mittente avrà l'informazione disponibile per primo).
*   **C. Comunicazione Molti-a-Molti:**
    *   Più mittenti trasmettono a più ricevitori.
    *   Flessibile e utile per interazioni complesse e coordinamento decentralizzato.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Importanza della Consegna Ordinata dei Messaggi**

   *   **A.** Cruciale nelle comunicazioni molti-a-molti.
   *   **B.** Garantisce che i messaggi raggiungano i destinatari in un ordine accettabile.
   *   **C.** Essenziale per le applicazioni coinvolte.

---
