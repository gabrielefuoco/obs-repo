
| **Termine**                       | **Definizione**                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **SMA4TD**                        | Social Media Analysis for Trajectory Discovery: Metodologia per scoprire pattern di mobilità da dati social.             |
| **Luoghi di Interesse (PoI)**     | Aree geografiche rilevanti per una comunità (es. monumenti, negozi, parchi).                                             |
| **Eventi**                        | Occasioni che attirano un gran numero di persone in un luogo e tempo specifici (es. concerti, partite).                  |
| **Elementi Geo-taggati**          | Contenuti social media con coordinate geografiche (es. tweet, post Instagram).                                           |
| **Pattern di Mobilità**           | Movimenti ricorrenti di persone tra luoghi di interesse.                                                                 |
| **Mining di Traiettorie**         | Scoperta di pattern di mobilità da dati sequenziali.                                                                     |
| **Pattern Associativi**           | Scoperta di elementi che si verificano insieme frequentemente.                                                           |
| **Pattern Sequenziali**           | Scoperta di sequenze di elementi che si verificano frequentemente in un ordine specifico.                                |
| **Regione di Interesse (RoI)**    | Area geografica che racchiude completamente un luogo di interesse.                                                       |
| **Dataset di Input**              | Insieme di dati strutturati contenenti informazioni su utenti, eventi e luoghi, utilizzato per il mining di traiettorie. |
| **Visualizzazione dei Risultati** | Rappresentazione grafica dei pattern di mobilità scoperti, per facilitarne la comprensione e l'interpretazione.          |
| **Originale e Destinazione**      | Punti di partenza e di arrivo dei movimenti degli utenti.                                                                |

---
## Analisi dei social media
L'analisi dei social media si concentra sull'uso di tecniche e strumenti per:
- **Raccogliere** dati dai social network e dai media online.
- **Analizzare** tali dati utilizzando metodi di analisi.
- **Estrarre** informazioni implicite e conoscenze nascoste.
- **Supportare** decisioni basate sui risultati delle analisi.

## Post sui social media geo-taggati
I post sui social media sono spesso arricchiti con **coordinate geografiche** o altre informazioni (ad es. testo, campi di localizzazione), che consentono di identificare le **posizioni** degli utenti. 

Quando gli utenti si spostano attraverso diversi luoghi, producono **dati geo-referenziati** che contengono informazioni rilevanti sui loro comportamenti di mobilità. Questi dati offrono grandi opportunità per il **mining delle traiettorie**.

## SMA4TD (Social Media Analysis for Trajectory Discovery)
SMA4TD è una metodologia ideata per scoprire i **pattern di mobilità** degli utenti che partecipano a eventi pubblici su larga scala.

### Obiettivi di SMA4TD
1. **Scoperta dei luoghi più visitati** e degli eventi più frequentati: Analisi dei dati per identificare i luoghi ed eventi con maggiore affluenza.
2. **Scoperta di insiemi di luoghi/eventi più frequentati** insieme: Estrazione di insiemi di luoghi visitati in combinazione o eventi partecipati collettivamente.
3. **Scoperta di pattern di mobilità** tra i luoghi e sequenze di eventi frequentati: Identificazione di comportamenti di mobilità e sequenze rilevanti di partecipazione.
4. **Scoperta di origine e destinazione** dei visitatori: Studio dei flussi di provenienza e destinazione per comprendere l'impatto turistico.

## Definizioni formali
- **P = {p₁, p₂, ...}** è un insieme di **Luoghi di Interesse** (Points of Interest, Pols). Ciascun *pᵢ* rappresenta un'area di rilevanza per una comunità in un dato periodo di tempo. Esempi di Pols includono:
  - **Location commerciali** (es. centri commerciali),
  - **Attrazioni turistiche** (teatri, musei, parchi),
  - **Luoghi rilevanti** per eventi (piazze, stadi).
  
  I Pols possono anche essere chiamati **Regioni di Interesse** (Regions of Interest, Rols), e rappresentano aree geografiche più ampie (ad esempio, poligoni su una mappa).

- **E = {e₁, e₂, ...}** è un insieme di **eventi** che coinvolgono grandi quantità di persone, dove $eᵢ = ⟨pⱼ, [tᵢ^begin, tᵢ^end]⟩$ rappresenta un evento che si svolge in *pⱼ* durante l'intervallo di tempo $[tᵢ^begin, tᵢ^end]$. Gli eventi possono includere:
  - **Partite** in stadi sportivi,
  - **Showcases** in padiglioni di esposizione,
  - **Concerti** in teatri o piazze.

- **G = {g₁, g₂, ...}** rappresenta un insieme di **elementi geo-taggati**: ogni *gᵢ* è un contenuto social (ad es. tweet, post, foto, video) pubblicato durante un evento *eᵢ* in un luogo *pⱼ*. Ciascun elemento *gᵢ* include:
  - **userID**: identificatore dell'utente,
  - **Coordinate geografiche** (latitudine e longitudine),
  - **Timestamp**: data e ora di pubblicazione,
  - **Testo**: descrizione testuale del contenuto,
  - **Tag**: etichette associate.

- **U = {u₁, u₂, ...}** è un insieme di **utenti**, dove *uᵢ* ha pubblicato almeno un elemento geo-taggato in *G*. 

Queste definizioni formali sono cruciali per analizzare e scoprire pattern di mobilità a partire dai dati estratti dai social media.

---
## SMA4TD: Metodologia in 7 Fasi

### 1. Definizione degli eventi (E)
   Si definisce l'insieme degli **eventi** *E*, dove ogni evento è descritto da:
   - **Luogo** di interesse (Pol),
   - **Orario** di inizio/fine,
   - **Caratteristiche** aggiuntive (es. gratuito/a pagamento, tipo di evento).

   **Esempio**:
   ```
   e1 = (Colosseo, (2016-11-01T19:00, 2016-11-01T23:00), A pagamento, Concerto)
   e2 = (Foro Romano, (2016-11-02T10:00, 2016-11-02T19:00), Gratuito, Visita guidata)
   ```

### 2. Definizione dei luoghi di interesse (P)
   Si definiscono i confini geografici dei **luoghi di interesse** (Pols). Questo può avvenire:
   - **Manuale**: disegnando poligoni su una mappa,
   - **Automatizzata**: utilizzando servizi esterni (es. OpenStreetMap).

### 3. Raccolta e pre-elaborazione degli elementi geo-taggati (G)
   Si raccolgono e si pre-elaborano gli **elementi geo-taggati** *G* relativi agli eventi in *E*:
   - Utilizzo di **API** dei social media,
   - **Pre-elaborazione**: pulizia, selezione e trasformazione dei dati.

### 4. Identificazione degli utenti (U)
   Si identifica l'insieme di **utenti** *U* che hanno pubblicato almeno un elemento geo-taggato in *G*.

### 5. Creazione del dataset di input (D)
   Si crea il dataset di input *D = {T₁, T₂, ...}* dove ogni *Tᵢ* è una tupla:
   ```
   < uᵢ, [eᵢ₁, eᵢ₂, ..., eᵢₖ], optFields >
   ```
   dove *uᵢ* è l'utente, *eᵢⱼ* sono gli eventi partecipati, e *optFields* contiene informazioni aggiuntive (es. nazionalità, interessi).

   **Esempio**:
   ```
   < u1, {e1, e2, e4, e5}, Italiano, arte >
   ```

### 6. Analisi dei dati e mining delle traiettorie
   Si esegue il **mining dei pattern associativi** e **sequenziali** su *D*:
   - **Pattern associativi**: utilizzati con l'obiettivo di scoprire i valori degli elementi che si verificano insieme con alta frequenza.
   - **Pattern sequenziali**: scoperta delle sequenze di elementi che si verificano più frequentemente nei dati.

   **Esempio di regole**:
   ```
   (e1, e5), 3  ←  3 utenti hanno partecipato agli eventi e1 ed e5
   (e1, e2, e5), 2  ←  2 utenti hanno partecipato agli eventi e1, e2 ed e5
   ```

### 7. Visualizzazione dei risultati
   Si creano **infografiche** per rendere i risultati comprensibili. Linee guida:
   - Rappresentazione **visiva** delle informazioni,
   - Minimizzare lo **sforzo cognitivo**,
   - Strutturare graficamente in **gerarchie**.

----
# Caso di studio 1: FIFA World Cup 2014

### Contesto generale
- **20ª edizione** del Campionato Mondiale FIFA.
- **Periodo**: 12 giugno - 13 luglio 2014.
- **Luogo**: Brasile.
- **Squadre**: 32 nazionali.
- **Partite**: 64 incontri giocati in 12 stadi.
- **Partecipazione**: Oltre 5 milioni di spettatori tra partite ed eventi correlati.

## 1. Definizione degli eventi
L'insieme degli eventi *E* include le 64 partite giocate durante la competizione:
$$E = \{e_1, e_2, ..., e_{64}\}$$
Ogni evento è rappresentato come:

$$e_i = <p_i, [t_i^{begin}, t_i^{end}], team_1, team2_, fase>$$

- **pi**: stadio,
- **tibegin**: 3 ore prima dell'inizio,
- **tiend**: 3 ore dopo la fine,
- **team1**: prima squadra,
- **team2**: seconda squadra,
- **fase**: fase della competizione (inaugurale, gironi, ottavi...).

**Esempi**:
```
e1 = (San Paolo, [2014-06-12T14:00, 2014-06-12T20:00], Brasile, Croazia, Partita Inaugurale)
e2 = (Natal, [2014-06-13T10:00, 2014-06-13T16:00], Messico, Camerun, Fase a Gironi)
```

## 2. Definizione dei luoghi di interesse
I luoghi di interesse *P* includono i 12 stadi utilizzati durante la Coppa del Mondo:
$P = \{p_1, p_2, ..., p_{12}\}$
Ogni stadio è rappresentato dalla **Regione di Interesse (RoI)**, cioè il più piccolo rettangolo che racchiude completamente i suoi confini.

## 3. Raccolta degli elementi geo-taggati
Sono stati raccolti **526.000 tweet** geo-taggati, pubblicati da utenti situati all'interno delle RoI durante le partite:
```
G = {g1, g2, ...}
```
Ogni elemento *gᵢ* contiene:
- **userID**: identificatore utente,
- **coordinate**: latitudine e longitudine,
- **timestamp**: data e ora del tweet,
- **testo**: contenuto del tweet,
- **tag**: hashtag associati,
- **applicazione** (opzionale): es. TwitterForAndroid.

**Esempio**:
```
g1 = (11223344, [-23.545531, -46.473372], 2014-06-12T16:59, "Orgoglioso di essere qui.", [#BRAvsCRO, #Brasil2014], TwitterForAndroid)
```

## 4. Pre-elaborazione dei dati
La pre-elaborazione su *G* ha incluso:
1. **Pulizia**: rimozione di tweet con posizioni inaffidabili.
2. **Selezione**: eliminazione di re-tweet e tweet non pertinenti.
3. **Trasformazione**: mantenuto un solo tweet per utente per partita per semplificare l'analisi.

## 5. Creazione del dataset di input
Il dataset di input *D* contiene la lista di partite a cui ha assistito ogni utente:
$D = \{T_1, T_2, ..., T_n\}$
Dove *Tᵢ = <uᵢ, {m₁, m₂, ..., mₖ}>*, e *m₁, m₂, ..., mₖ* sono le partite a cui ha partecipato l'utente *uᵢ*.

## 6. Mining delle traiettorie
È stato eseguito il **mining delle traiettorie** per individuare i movimenti più frequenti dei tifosi tra gli stadi. Un pattern di traiettoria frequente *fp* è rappresentato come:
```
fp = <m₁, m₂, ..., mₖ>(s)
```
Dove *s* indica il supporto del pattern (percentuale di transazioni che contengono la sequenza *fp*).

## 7. Risultati
- Numero di spettatori per ogni partita nel tempo.
- Numero di partite seguite dai tifosi.
- **Sequenze più frequenti** di partite seguite nello stesso stadio o per una stessa squadra.
- **Pattern di movimento** raggruppati per fase della competizione.

---

# Caso di studio 2: EXPO 2015

### Contesto generale
- **Visitatori**: Circa 22 milioni di persone in 6 mesi, il più grande evento del 2015.
- **Espositori**: 188 spazi espositivi (paesi, organizzazioni internazionali, aziende).
- **Tema**: "Nutrire il Pianeta, Energia per la Vita".

## 1. Definizione degli eventi
L'insieme degli eventi *E* include 188 showcase esposti nei padiglioni:
```
E = {e1, e2, ..., e188}
```
Ogni evento è rappresentato come:
$$e_i = <p_i, [t_i^{begin}, t_i^{end}]>$$
- **pi**: padiglione,
- $t_i^{begin}$: 1 maggio 2015,
- $t_i^{end}$: 31 ottobre 2015.

## 2. Definizione dei luoghi di interesse
I luoghi di interesse *P* sono i 188 padiglioni:
```
P = {p1, p2, ..., p188}
```
Ogni padiglione è rappresentato dalla sua **Regione di Interesse (RoI)**, identificata sulla mappa di EXPO.

## 3. Raccolta e pre-elaborazione dei dati
- **Fonti**: Post geolocalizzati da utenti Instagram che hanno visitato almeno un padiglione.
- **Numeri**:
  - 238.000 utenti Instagram che hanno visitato EXPO.
  - 570.000 post pubblicati durante la visita.
  - 2,63 milioni di post pubblicati da 1 mese prima a 1 mese dopo la visita.

### Formalmente:
```
G = {g1, g2, ...}
```
Ogni elemento *gᵢ* contiene:
- **userID**: identificatore utente,
- **coordinate**: latitudine e longitudine,
- **timestamp**: data e ora del post,
- **testo**: contenuto del post,
- **tag**: hashtag associati.

**Esempio**:
```
gᵢ = <111222333, [45.521443, 9.096251], 2015-09-03T11:27, "Cibo verde da un Paradiso Verde", [#SriLanka, #food, #green]>
```

I dati sono stati pre-elaborati mantenendo un solo post per utente per padiglione per giorno.

## 4. Identificazione degli utenti
L'insieme degli utenti *U* include chi ha pubblicato almeno un post a EXPO 2015:
```
U = {u1, u2, ...}
```
Ogni utente *uᵢ* contiene **userID** e **nazionalità**. Per gli italiani, sono state registrate anche città e regione di origine, ricavate dai post pubblicati nei 30 giorni precedenti la visita.

## 5. Creazione del dataset di input
Il dataset di input *D* contiene i PoI visitati dagli utenti:
```
D = {T1, T2, ..., Tn}
```
Ogni *Tᵢ = <uᵢ, {p₁, p₂, ..., pₖ}>* rappresenta la lista di padiglioni visitati dall'utente *uᵢ*.

Il **PoI** di un post durante EXPO corrisponde a un padiglione, mentre prima o dopo corrisponde a una località esterna (città, regione, stato).

## 6. Risultati
- **Tendenze di visita** nel tempo.
- **Padiglioni più visitati**.
- **Pattern di visite** più frequenti tra i padiglioni.
- **Origine e destinazione** dei visitatori, con analisi specifica per gli stranieri.

### Osservazioni principali:
- Aumento delle visite negli ultimi due mesi (settembre e ottobre 2015).
- Picchi di visite durante i fine settimana (sabato).
- Correlazione forte (Pearson 0,7) tra numeri ufficiali e utenti Instagram.
- I visitatori stranieri provenivano principalmente da **Spagna** (19,3%) e **Francia** (19,1%), seguiti da **Regno Unito** (13,3%) e **USA** (10,9%).

## 7. Conclusioni
L'approccio SMA4TD è utile per pianificare eventi futuri, monitorare servizi essenziali (trasporti, sicurezza, logistica) e comprendere i movimenti delle persone tramite l'analisi di grandi quantità di dati provenienti dai social media.

---
### Quiz

1. **Quali sono i quattro obiettivi principali di SMA4TD?**
2. **Cosa rappresentano gli insiemi P, E, G e U nella definizione formale di SMA4TD?**
3. **Descrivi brevemente i sette passi della metodologia SMA4TD.**
4. **Nel caso di studio della Coppa del Mondo FIFA 2014, cosa rappresenta l'insieme degli eventi E e come viene definito ciascun evento?**
5. **Nel caso di studio di EXPO 2015, da quale piattaforma di social media sono stati raccolti i dati e quali informazioni sono state estratte da ogni post?**
6. **Quali sono le principali differenze nella definizione degli eventi e dei luoghi di interesse tra i casi di studio della Coppa del Mondo e di EXPO?**
7. **Quali sono i vantaggi di utilizzare i dati dei social media per l'analisi delle traiettorie?**
8. **Quali sono le sfide che si possono incontrare nell'utilizzare i dati dei social media per l'analisi delle traiettorie?**
9. **Oltre ai casi di studio presentati, in quali altri contesti potrebbe essere applicata la metodologia SMA4TD?**
10. **Quali sono le implicazioni etiche da considerare nell'analisi dei dati dei social media per la scoperta delle traiettorie?**

### Risposte al Quiz

1. I quattro obiettivi principali di SMA4TD sono: scoprire i luoghi/eventi più visitati, scoprire insiemi di luoghi/eventi frequentati insieme, scoprire pattern di mobilità tra luoghi/eventi e scoprire l'origine e la destinazione dei visitatori.
2. P rappresenta l'insieme dei Luoghi di Interesse (PoI), E l'insieme degli eventi, G l'insieme degli elementi geo-taggati (post sui social media) e U l'insieme degli utenti che hanno pubblicato tali elementi.
3. I sette passi di SMA4TD sono: definire gli eventi, definire i luoghi di interesse, raccogliere e pre-elaborare i dati geo-taggati, identificare gli utenti, creare il dataset di input, eseguire il mining delle traiettorie e visualizzare i risultati.
4. L'insieme E rappresenta le 64 partite della Coppa del Mondo. Ogni evento è definito da stadio, orario di inizio/fine, squadre partecipanti e fase della competizione.
5. I dati sono stati raccolti da Instagram. Da ogni post sono stati estratti: userID, coordinate, timestamp, testo, hashtag.
6. Nella Coppa del Mondo, gli eventi erano partite con durata limitata, mentre in EXPO erano mostre permanenti. I luoghi di interesse erano stadi nella Coppa del Mondo, padiglioni in EXPO.
7. I dati dei social media sono abbondanti, aggiornati e permettono di analizzare comportamenti su larga scala a basso costo.
8. Le sfide includono la gestione di dati rumorosi e incompleti, la tutela della privacy degli utenti e la corretta interpretazione dei risultati.
9. SMA4TD potrebbe essere applicata ad altri eventi come concerti, festival, manifestazioni, ma anche per analizzare flussi turistici in una città.
10. Bisogna garantire l'anonimato degli utenti, ottenere il consenso informato per l'utilizzo dei dati e evitare di generare discriminazioni o pregiudizi nell'analisi e interpretazione dei risultati.
---
# FAQ: Analisi dei Social Media e Scoperta delle Traiettorie (SMA4TD)

## 1. Cos'è l'analisi dei social media e come funziona?

L'analisi dei social media utilizza tecniche e strumenti per raccogliere dati da social network e media online, analizzarli per estrarre informazioni implicite e conoscenze nascoste, e supportare decisioni basate sui risultati. Nel contesto della scoperta delle traiettorie, si concentra sui post geo-taggati per comprendere i movimenti e i comportamenti degli utenti.

## 2. Cosa significa SMA4TD e quali sono i suoi obiettivi?

SMA4TD sta per "Social Media Analysis for Trajectory Discovery" ed è una metodologia che analizza i dati dei social media per scoprire i pattern di mobilità degli utenti durante eventi su larga scala. I suoi obiettivi principali sono:

- Identificare i luoghi e gli eventi più frequentati.
- Scoprire insiemi di luoghi/eventi visitati insieme.
- Identificare pattern di mobilità e sequenze di eventi frequentati.
- Studiare i flussi di provenienza e destinazione dei visitatori.

## 3. Quali tipi di dati vengono utilizzati in SMA4TD e come vengono definiti formalmente?

SMA4TD utilizza principalmente post geo-taggati provenienti da piattaforme di social media. Formalmente, vengono definiti:

- **P**: Insieme di Luoghi di Interesse (PoI), come attrazioni turistiche, centri commerciali e luoghi di eventi.
- **E**: Insieme di eventi, ognuno caratterizzato da luogo, orario di inizio/fine e altre caratteristiche.
- **G**: Insieme di elementi geo-taggati, come tweet o post di Instagram, contenenti ID utente, coordinate, timestamp, testo e tag.
- **U**: Insieme di utenti che hanno pubblicato almeno un elemento geo-taggato in G.

## 4. Quali sono le fasi principali della metodologia SMA4TD?

SMA4TD si articola in sette fasi principali:

1. **Definizione degli eventi (E)**: Identificazione degli eventi rilevanti, specificando luogo, orario e caratteristiche.
2. **Definizione dei luoghi di interesse (P)**: Definizione dei confini geografici dei luoghi di interesse.
3. **Raccolta e pre-elaborazione degli elementi geo-taggati (G)**: Raccolta dei dati dai social media e pulizia, selezione e trasformazione degli stessi.
4. **Identificazione degli utenti (U)**: Identificazione degli utenti che hanno pubblicato elementi geo-taggati rilevanti.
5. **Creazione del dataset di input (D)**: Creazione di un dataset strutturato contenente informazioni su utenti, eventi a cui hanno partecipato e altri dati opzionali.
6. **Analisi dei dati e mining delle traiettorie**: Esecuzione di algoritmi di mining per scoprire pattern associativi e sequenziali nei dati.
7. **Visualizzazione dei risultati**: Creazione di infografiche e visualizzazioni per presentare i risultati in modo chiaro e comprensibile.

## 5. In che modo è stato applicato SMA4TD al caso di studio della Coppa del Mondo FIFA 2014?

Per la Coppa del Mondo FIFA 2014, SMA4TD è stato utilizzato per analizzare i tweet geo-taggati pubblicati durante le partite. Sono stati definiti eventi (partite), luoghi di interesse (stadi) e raccolti dati sui tweet degli utenti presenti. L'analisi ha permesso di identificare i movimenti dei tifosi tra gli stadi, le sequenze di partite seguite e i pattern di mobilità durante la competizione.

## 6. Quali sono state le principali conclusioni dell'applicazione di SMA4TD all'EXPO 2015?

L'analisi dei dati Instagram per EXPO 2015 ha rivelato informazioni sulle tendenze di visita, i padiglioni più popolari e i pattern di visita. È stata evidenziata una forte correlazione tra i dati ufficiali sui visitatori e quelli raccolti da Instagram. L'analisi della provenienza geografica dei visitatori ha fornito informazioni utili per la pianificazione di eventi futuri e la promozione turistica.

## 7. Quali sono i potenziali benefici e le applicazioni di SMA4TD?

SMA4TD offre diversi benefici, tra cui:

- Migliorare la pianificazione di eventi su larga scala.
- Monitorare i servizi essenziali durante gli eventi (trasporti, sicurezza).
- Comprendere i comportamenti e i movimenti delle persone in contesti specifici.
- Ottimizzare le strategie di marketing e promozione turistica.

## 8. Quali sono le sfide e le limitazioni di SMA4TD?

Nonostante i suoi vantaggi, SMA4TD presenta anche alcune sfide:

- **Privacy**: La raccolta e l'analisi dei dati dei social media sollevano preoccupazioni sulla privacy degli utenti.
- **Distorsioni**: I dati dei social media potrebbero non essere rappresentativi dell'intera popolazione, portando a distorsioni nei risultati.
- **Affidabilità**: La qualità e l'accuratezza dei dati raccolti possono variare a seconda della piattaforma e del tipo di dati.