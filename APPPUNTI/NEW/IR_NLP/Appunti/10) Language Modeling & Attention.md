## Language Modeling

Il *language modeling* è un task autoregressivo che si concentra sulla generazione di testo.  L'input consiste in una sequenza di parole osservate, $x_1, ..., x_t$ (dove *t* rappresenta il time step). Il task consiste nel predire la parola successiva, $x_{t+1}$.

Si assume che il vocabolario sia noto a priori e che il generatore campioni da esso secondo specifiche strategie.

![[10)-20241118151227953.png|552]]

La probabilità di generare un testo T può essere vista come il prodotto delle probabilità condizionate di osservare ogni parola, data la sequenza di parole precedenti:

![[10)-20241118151329195.png|585]]

Il language modeling è importante non solo per la semplice predizione della parola successiva, ma anche per una vasta gamma di applicazioni nel campo del linguaggio naturale, tra cui:

* **Machine Translation:**  Può essere considerato un caso particolare di language modeling, in quanto implica una logica di encoding nel linguaggio sorgente e decoding nel linguaggio target.
* **Speech Recognition:**  La predizione di parole successive è fondamentale per la trascrizione accurata del parlato.
* **Spelling/Grammar Correction:**  Il modello può identificare e correggere errori ortografici e grammaticali.
* **Summarization:**
    * **Estrattiva:** Evidenzia le frasi più importanti da un testo.
    * **Astrattiva:** Rimodula il testo originale creando un riassunto.  Anche la summarization astrattiva può essere considerata un caso particolare di language modeling, poiché, data una sequenza di testo in input, genera una nuova sequenza di testo in output.


# N-gram Language Models

Un n-gram è una porzione di testo composta da *n* token consecutivi.  I modelli n-gram collezionano statistiche di occorrenza di n-gram per stimare la probabilità della parola successiva.

Esempi:

* **Unigrammi:** "the", "students", "opened", "their"
* **Bigrammi:** "the students", "students opened", "opened their"
* **Trigrammi:** "the students opened", "students opened their"
* **Four-grammi:** "the students opened their"

Invece di considerare l'intero testo precedente, si utilizza una finestra di *n-1* parole per semplificare il problema.

Si fa l'assunzione di Markov:  $x^{(t+1)}$ dipende solo dalle *n-1* parole precedenti.

![[10)-20241118152157750.png]]

**Come ottenere le probabilità degli n-gram e (n-1)-gram?**

Contandole in un ampio corpus di testo!

![[10)-20241118152225539.png]]

![[10)-20241118152314916.png]]


![[10)-20241118152452827.png]]

**Problemi:**

* **Problema 1: Numeratore = 0:**  La probabilità di un n-gram potrebbe essere zero se non è presente nel corpus.
* **Problema 2: Denominatore = 0:** La probabilità di un (n-1)-gram potrebbe essere zero.  Una soluzione potrebbe essere quella di valutare l'n-2-gram, aumentando la finestra di contesto, ma non garantisce il successo.

L'utilizzo di un valore di *n* relativamente grande aumenta le dimensioni del modello (model size) e non garantisce un miglioramento delle prestazioni.  Questo approccio porta a problemi di:

* **Sparsità:**  Aumenta all'aumentare di *n*.
* **Granularità:** Diminuisce all'aumentare di *n*.

Di conseguenza, si rischia di ottenere probabilità piatte e poco informative.  Anche con un corpus di testo di dimensioni adeguate, un modello con *n* grande potrebbe generare testo grammaticalmente corretto ma privo di coerenza e fluidità.

## Costruire un Language Model Neurale

![[10)-20241118153020961.png]]

L'obiettivo è costruire un modello neurale basato su una finestra di contesto (window-based).

![[10)-20241118153136194.png]]

Ogni parola possiede un encoding.  Questo encoding viene ulteriormente trasformato tramite una matrice di parametri U. Questa trasformazione permette di individuare la parola con il punteggio (score) più alto, corrispondente alla parola più probabile nel contesto.

![[10)-20241118153152658.png]]

Dato un certo numero di input (rappresentati da one-hot vector),  gli embeddings delle parole vengono trasformati da una funzione f (es. regressione logistica).  L'apprendimento della codifica avviene nel layer nascosto (consideriamo un singolo hidden layer).  Questa codifica intermedia viene poi utilizzata in una nuova trasformazione lineare. Infine, la funzione softmax converte i punteggi grezzi (raw scores) in probabilità.

## A Fixed-Window Neural Language Model

![[10)-20241118153353157.png]]

Questo modello non richiede la memorizzazione delle statistiche di conteggio degli n-gram.

**Problemi:**

* **Dimensione della matrice W:**  Una finestra di contesto ampia implica una matrice W di grandi dimensioni.
* **Asimmetria dell'input:** Gli input vengono moltiplicati per pesi completamente diversi.  Non c'è simmetria rispetto all'ordine di presentazione dell'input.  Il fatto che una parola preceda o segua un'altra è rilevante.  Dato un contesto con una parola centrale, si desidera una rappresentazione globale del contesto per il riconoscimento di entità nominate (named entity).  Per questo, si utilizza una matrice W che non viene applicata allo stesso modo a ogni input, causando asimmetria. L'ottimizzazione dei valori in W avvantaggia le parole in momenti diversi (quelle all'inizio della sequenza potrebbero utilizzare una parte non ottimizzata della matrice).


## Reti Neurali Ricorrenti (RNN)

L'obiettivo è condividere i pesi (w) per ogni parola nella sequenza di input.

![[10)-20241118153747550.png]]

L'output può essere generato ad ogni time step o solo all'ultimo, a seconda del task specifico.  Ad esempio, nell'analisi del sentiment, interessa solo l'output finale. In questo esempio, w contribuisce ad ogni passo, quindi la codifica del passo precedente influenza ogni time step successivo. Un'architettura neurale che segue questo principio, prendendo in input una sequenza di parole, è detta rete neurale ricorrente.

![[10)-20241118154009409.png]]

Se ogni x è un vettore di dimensione b con tutti 0 e un solo 1, abbiamo una codifica one-hot di ogni parola. Questa codifica viene utilizzata nella trasformazione descritta nell'immagine.

Distinguiamo due matrici di pesi:  $w_h$ (trasformazioni dallo stato precedente) e $w_e$ (per l'input corrente al passo t).

Ogni blocco al passo *t* prende in input la codifica della parola al passo *t* e l'output trasformato (moltiplicato per la sua matrice di pesi) del passo precedente.

Ad ogni passo *t* otteniamo la codifica $h_t$.


**Pro:**
* **Simmetria dei pesi:** I pesi vengono applicati ad ogni timestep, garantendo simmetria nell'elaborazione della sequenza.
* **Dimensione del modello costante:** La dimensione del modello non aumenta con l'aumentare della lunghezza della sequenza di input.

**Contro:**
* **Lunghezza della sequenza limitata:** La lunghezza della sequenza non è arbitraria.  Il modello ha difficoltà nell'elaborare sequenze lunghe a causa di un effetto di "perdita di memoria".  Quando si valuta la probabilità della parola successiva, si osserva un'attenuazione significativa dei valori delle probabilità delle parole precedenti nella sequenza.
* **Tempo di addestramento:** L'addestramento del modello RNN richiede tempi lunghi.

## Addestramento di un Modello Linguistico RNN

L'addestramento di un modello linguistico RNN richiede un ampio corpus di testo con sequenze molto lunghe. Ad ogni timestep, il modello riceve in input una sequenza e predice la distribuzione di probabilità per la parola successiva.  La funzione di costo (loss) ad ogni timestep *t* è la cross-entropy:

$j^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w)=-\log \hat{y}^{(t)}_{x_{t+1}}$

dove:

* $y^{(t)}$ è il one-hot vector rappresentante la parola effettiva al passo (t+1).
* $\hat{y}^{(t)}$ è la distribuzione di probabilità predetta dal modello al passo *t*.
* V è il vocabolario.


La loss complessiva sull'intero training set è la media delle loss calcolate ad ogni timestep *t*.  Si noti che ad ogni passo *t* si ha una predizione $\hat{y}^{(t)}$.

![[10)-20241118155458516.png]]

La loss totale è la somma cumulativa delle loss individuali ad ogni timestep.

Tuttavia, calcolare la loss e i gradienti sull'intero corpus contemporaneamente è computazionalmente troppo costoso in termini di memoria.

# Backpropagation Through Time (BPTT) per RNNs

Per addestrare una Recurrent Neural Network (RNN) si utilizza la Backpropagation Through Time (BPTT), una variante dell'algoritmo di backpropagation.  BPTT calcola i gradienti dei pesi rispetto alla funzione di costo.

![[10)-20241118160122136.png]]

Il gradiente rispetto ad un peso ripetuto (come $W_h$ nella figura) è la somma dei gradienti calcolati ad ogni timestep in cui quel peso contribuisce al calcolo.  In altre parole, è la somma di gradienti di forma identica calcolati ad ogni timestep.

Questo è un'applicazione della regola della catena, nello specifico la *Multivariable Chain Rule*.

![[10)-20241118160326719.png|690]]

Per chiarire, se dovessimo calcolare la derivata parziale di una funzione composta $f(a(b(x)))$ rispetto a $x$, dovremmo applicare la regola della catena due volte: una volta per la funzione $a$ e una volta per la funzione $b$.  Analogamente, nel caso delle RNN, ad ogni timestep dobbiamo valutare la derivata rispetto a $W_h$.  La derivata di $W_h$ al passo *t* rispetto a $W_h$ al passo *t* è 1.

Quindi, ad un generico timestep *t*, per calcolare il gradiente rispetto a $W_h$, dobbiamo propagare all'indietro il gradiente cumulativo fino all'inizio della sequenza, sfruttando il fatto che la matrice $W_h$ rimane costante ad ogni passo.  Questo processo di propagazione all'indietro dei gradienti attraverso il tempo è ciò che caratterizza la BPTT.

## Valutazione del modello

Il language model viene valutato tramite la **Perplexity**
$$\prod_{t=1}^t\left( \frac{1}{P_{ML}(x^{(t+1)})|x^{(t)},\dots,x^{(1)}} \right)^{1/t}$$
rappresenta l'inverso della probabilità del corpus, normalizzato dall'esponente che rappresetna il numero di parole
![[10)-20241118162753924.png]]

## Vanishing Gradient

nel calcolare derivate ricorsivamente, andiamo a  valutarle per probabilità molto piccole: i gradienti diventano sempre più piccoli e viene abbattuto il gradiente al passo t

![[10)-20241118163104268.png]]
definizione di h^{(t)}: applicazione di una funzione di attivazione (solitamente una funzione non lineare come la tangente iperbolica o la funzione ReLU) alla combinazione lineare dell'embedding dell'input al timestep *t*, del bias e della trasformazione dello stato nascosto al timestep precedente.


diagonalizzazione della derivata della funzione di attivazione per w_h

# registrazione 2 Lunedì

spiegazione della derivata di J


![[10)-20241118163617737.png]]

quanto è vero che a creare il vanishing del gradiente è la potenza L di valori molto piccoli?
vale quando gli autovalori della matrice sono <1
il valore del gradiente al passo i può essere riscritto usando gli autovettori di W_h. con valori molto piccoli approssima a zero, dunque abbiamo dimostrato che in teroria è possibile
nella pratica tale problema esiste. 
può accadere che gli autovalori di W possono essere >1, e questo porta a un effetto contrario: esplosione del gradiente

![[10)-20241118163905967.png]]
![[10)-20241118163913300.png]]
è un problema di divergenza del gradiente, si risolve più facilmente del problema vanishing.
è molto frequente, si risolve con operazioni di normalizzazione: può essere uno scaling tale per cui i valori abbiamo norma pari a 1
in una rete ricorrente si usa spesso il clipping, che è un tresholding del gradiente
si sceglie una soglia e ad ogni passo scaliamo il gradiente rispetto una soglia fissata

## Fixing the vanishing gradient problem
Gradient Signal from far away iS Iost because it's much smaller than gradient Signal from close-by.
So, model weights are updated only with respect to near effects, not long-term effects.

![[10)-20241118164417586.png]]
ha bisogno di un intervento architetturale per essere risolto: anzichè riscrivere lo stato corrente tenendo conto dell'intera lunghezza, aggiorniamo lo stato rispetto a un contesto più breve ma tenendo separatamente una sorta di buffer che ci dice quanto usare dal contesto precedente nella generazione delle nuove parole


# Long Short-Term Memory

A type of RNN proposed by Hochreiter and Schmidhuber in 1997 as a solution to the problem of
vanishing gradients
• Everyone Cites that paper but really a crucial part of the modern LSTM is from Gers et al. (2000)
Only started to be recognized as promising through the work of S's student Alex Graves c. 2006
• Work in which he also invented CTC (connectionist temporal classification) for speech recognition
But only really became well-known after Hinton brought it to Google in 2013
• Following Graves having been a postdoc with Hinton


vogliamo riprogettare una rnn con una sorta di memoria, per sistemare la parte iniziale che danneggia la back propagation
introduciamo la notazione c, che sta per cella di memoria, serve per gestire l'informazione a lungo termine
abilitiamo delle informazioni di lettura scrittura e cancellazione
la selezione di quale informazione deve essere gestita è controllata da determinati gates
vettori della stessa dimensionalità, ad ogni timestep il vettore dei gates sarà o aperto o chiuso
i loro valori sono dinamici e cambiano a seconda di input e contesto

![[10)-20241118164929500.png]]
partendo dal basso, bogliamo calcolare gli hidden $h^{(t)}$ e le celle $c^{(t)}$
$h^{(t)}$ è una combinazioone element wise tra l'attivazione dello stato della cella (tangente iperbloica) per $o^{(t)}$, che è l output gate(filtro), controlla quali parte della cella di memoria vanno a contribuire allo stato hidden al passo t

c^t è la combinazione tra c al passo precedente per lo stato al passo corrente di quella che il nuovo contenuto da inserire in memoria
la combinazioneè controllata da due gate f(forget) e i(input)
lo stato di memoria al passo t è la combinazione tra una parte dello stato di memoria al passo precedente e la combinazione del nuovo contenuto, determinato dal nostro input trasformato combinandolo linearmente con l'hidden state al passo precedente
il risultato è $\tilde{c}^{(t)}$

ciascun gate è ottenuto come trasformazione della regressione dell input x al passo t  con la codifica intermedia al passo t+1
ogni gate ha associato dei parametri distinti


![[10)-20241119095018808.png]]
nell'immagine precedente vengono messi in evidenza i flussi
![[10)-20241119095044205.png]]

How does LSTM solve vanishing gradients?
• The LSTM architecture makes it much easier for an RNN to
preserve information over many timesteps
• e.g., if the forget gate is set to 1 for a cell dimension and the input gate
set to O, then the information of that cell is preserved indefinitely.
• In contrast, it's harder for a vanilla RNN to learn a recurrent weight
matrix Wh that preserves info in the hidden State
• In practice, you get about 100 timesteps rather than about 7

forget gate impostato a 1 per una dimensione della cella e input gate a 0, allora l'info della cella è preservata
esistono dei modi alternativi di creare delle connessioni all'interno della rete per preservare le dipendenze a lunga distanza: l'aggiunta di connessioni dirette è un'esigenza nelle rnn, che fanno passare il gradiente direttamente senza trasformazioni
l'input per un layer subisce una trasformazione per cui la funzione è quella d'identità
l'input di un layer si combina con l output di quel layer:il gradiente può passare da un timestep a uno precedente, evitando il problema del vanishing gradient
sono chiamati skip connection o connessioni residue (resnet)

connsessioni dense: connettere ogni layer a ogni altro layer che lo segue (densenet)

highway connections (highwayNet): invece di avere funzione di attivazione identiotà vi è un meccanismo di gating che determina quale parte far passare direttamente


## estensione con bidirezionalità

abbiamo una certa frase e immaginiamo un task di sentiment analisys: 
![[10)-20241119100108956.png]]
l ambiguità dell'aggettivo terribly in esempio ci porta a pensare che sia necessario aggiungere bidirezionalità

l input entra in due layer paralleli, uno che mantine la direzionalità classica da sinistra a destra e uno al contrario
![[10)-20241119100151634.png]]
la bidirezionalità è ottenuta combinando la rnn con una rnn che è il reverse
verranno dunque combinati i due output
per fare predizione della parola questa cosa non è utile
se il task non è autoregressivo, possiamo intervenire con la bidirezionalità per migliorare la rappresentazione contestuale
$h^{(t)}$ è la combinazione di due stati: $h^{(t)}$ della forward e $h^{(t)}$ della backward, ciascuno coi propri parametri


![[10)-20241119100621466.png]]

la seconda estensione è quella di rendere "deep" la rete, aggiungendo più dimensioni

![[10)-20241119100629530.png]]

![[10)-20241119100706568.png]]una rappresentazione di questo tipo serve a catturare rappresentazioni che corrispondono a relazioni di diverso ordine tra le parole: facendo leva sull effetto della prossimità di località o globalità tra le parole, la rete rende piu efficace la rnn per catturare proprieta grammaticali o sintattiche, coi layer piu vicini all input, e man mano che aggiungiamo layer, le rappresentazioni finali servono per ottenere embeddings che catturano relazioni di alto livello (di carattere semantico)
anche questa idea sarà parte integrante dei transformers

per language modelling, con una rete del genere potremmo ottenere testi generati non solo grammaticalmente coerenti ma tali da avere una maggiore coerenza linguistica

tipicamente non andiamo oltre i 3 layer per catturare al meglio sia le proprietà grammaticali che semantiche (4 è il massimo, ma può non valerne la pena rispetto a 3)

con skip connections si può arrivare a 8 layer,12 per bert e 24 per i transformer specifici per l'encoding

## Machine Translation

task considerato particoalrmente difficile fino al 2015.
non è language modelling perchè non è next word predictions, è un caso rappresentativo in cui abbiamo un input e un contesto di riferimento che svolgono due ruoli completamente diversi
un task simile è la summarization
la dimensione deve essere comparabile sia per risorse per il target
![[10)-20241119101809788.png]]
l'idea centrale è quella di apprendere un modello probabilistico dai dati: supponiamo di voler tradurre da francese a inglese
data la frase in francese x vogliamo generare la migliore frase nel linguaggio target, y, e vogliamo massimizzare questa probabilità 
dunque vogliamo massimizzare quella y da generare che massimizza la joint probability tra la probabilità a priori della frase da generare e la likelihood x|y
dobbiamo dunque apprendere due componenti, il modello di traduzione, che deve apprendere come parole costituenti delle frase devono essere tradotte, e l'approccio è usare dati paralleli, ovvero dati formati da coppie di traduzioni letterali (linguaggio sorgente, linguaggio target).
i modelli non vanno addestrati separatamente: il task di language modelling non fa leva sulle proprieta intrinseche del modello decoder addestrato su grandi corpus ma deve essere condizionato all'input, ci serve uno spazio di rappresentazione comune in cui l input è codificato in uno spazio di rappresentazione denso(produciamo l embedding della frase) e ce ne serviamo per condizionare parola per parola la frase prodotta

## Seq2Seq Model

![[10)-20241119102537146.png]]
frase in input in francese, vogliamo fare tutto con due architetture rnn, questa cosa può essere anche fatta con lstm multi layer
abbiamo bisogno di codificare la frase in input e ce ne serviamo per inizializzare lo stato iniziale $h^{(0)}$ della seconda rnn, che ad ogni passo predice la prossima parola
il primo input per l'rnn decoder è un token speciale, indicato genericamente con "START", che rappresenta l'inizio della frase. con una cross entropy diventa la possibilità di predire la parola che meglio si avvicina alla parola successiva nella sequenza.

la seconda rnn è autoregressiva 

nella parte di encoding potremmo avere una rete bidirezionale

![[10)-20241119103053527.png]]

![[10)-20241119103147192.png]]

come si addestra su un task di conditional language model?
i pesi diversi di encoder e decoder vengono aggiornati ad ogni step di backpropagation insieme
ci aspettiamo una convergenza del training lenta e complicata, ma è inevitabile se vogliamo condizionare il decoder all'encoder
![[10)-20241119104448084.png]]
il condizionamento sul decoder emerge come un collo di bottiglia: il decoder è condizionato dall'output globale dell encoder. ci chiediamo dunque se non sarebbe meglio che in ogni passo dell decoding ci sia un riferimento non solo globale ma relativo a ciascun elemnto della frase in input

## greedy decoding
![[10)-20241119104913791.png]]
produci in output la parola che ha massimizzato la probabilità
in quanto greedy non possiamo valutare a ogni step la decisione che va a condizionare gli step successivi

possiamo pensare a un decoding basato su scelta esaustiva:
$$P(y|x)=\prod_{t=1}^TP(y_{t}|y_{1},\dots,y_{t-1},x)$$
![[10)-20241119105100890.png]]

il tradeoff prende il nome di beam search
![[10)-20241119105336897.png]]
teniamo conto dell k traduzioni più probabili a ogni step

l'espressione ipotesi si riferisce a una frase candidata: abbiamo k ipotesi ad ogni step, k è un parametro
ogni ipotesi al passo t ha uno score associato che tiene conto delle probabilità cumulate ad ognuno dei passi precedenti
la frase tradotta migliore corrisponderà all'ipotesi che ha accumulato il miglior score
non cerchiamo l'ottimo globale ma non è una ricerca esaustiva perchè dobbiamo esplorare un albero

![[10)-20241119105722368.png]]

## BLEU
si occupa di confrontare l output di un machine traslator con l output di una traduzione umana

calcola uno score di matching tra la traduzione e il riferimento alla stessa
calcolato con una combinazione di valori di precisione basati su n gram(n varia da 1 a 4)
questa misura è un accumulo di precisioni di n gram con un coefficiente di penality per traduzioni molto brevi(potrei ingannare l accuracy con traduzioni di lunghezza 1)
ssues with Precision
● repetition
● multiple target sentences
Clipped Precision
● Compare each word from the predicted sentence with all of the target sentences
○ If the word matches any target sentence, it is considered to be correct
● Limit the count for each correct word to the maximum number of times that word occurs in
the target sentence

![[10)-20241119110907996.png]]

BLEU score as a product of the geometric average precision and brevity penalty

Pros:
● Quick to calculate and easy to understand
● Corresponds to the way a human would evaluate the same text
● Language-independent
● Can be used when you have more than one ground truth sentence
Cons:
● Does not consider the meaning of words
● Looks only for exact word matches
● Ignores the importance of words
● Ignores the order of words
○ e.g., “The guard arrived late because of the rain” and “The rain arrived late because of the
guard” would get the same (unigram) Bleu Score

volendo fare qualcosa di diverso avremmo bisongo di un modello multilingua che funge da "oracolo", abile nel codificare embeddings di sentence e non singole parole

## Attention

piuttosto che affidarci alla codifica finale del contesto in input per l'encoder e quindi scontrarci col conditioning bottleneck, vogliamo che ogni passo di generazione del decoder sia guidato da un aggregazione pesata dei contributi di ogni topic che possano condizionare la generazione del decoder a ogni step:

![[10)-20241119111802849.png]]
introdotto per risolvere il conditioning bottleneck, il decoder è condizionato dall'output globale dell'encoder

vogliamo introdurre delle connessioni dirette dal decoder all'encoder in modo tale che il decoder possa svolgere il suo ruolo sapendo su quali pezzi dell'encoder concentrarsi di volta in volta
si dice anche che il decoder attende all'encoder
si parte con il simbolo di start e c'è un attention, vengono calcolati i score e vengono poi aggregati. il risultato viene concatenato con l output del decoder al passo t e entrambi supportano la generazione della paraole.
attention output fa da summary per l'info codificata dal decoder che ha ricevuto maggior attenzione

![[10)-20241119112324451.png]]
![[10)-20241119112330390.png]]
![[10)-20241119112338437.png]]
![[10)-20241119112402479.png]]
![[10)-20241119112206826.png]]

vogliamo prendere una scelta che deriva dai vari contributi che ciascun pezzo dell'encoder fornisce in quel passo al decoder

![[10)-20241119112412234.png]]
**Notazione:** 
$h_1,...,h_N$ è il vettore che indica la codifica del layer nascosto, 
$N$ è la lunghezza della frase in input, 
$e^{(t)}$ è l'attention score$e^{(t)}=[s_{t}^Th_{1},\dots,s_{t}^Th_{n}] \in \mathbb{R}^N$
$s_{t}$ è lo stato hidden del decoder ad ogni passo $t$ abbiamo
 $a$ e $\alpha$ sono i coefficenti di attenzione, in particolare:
$\alpha^{(t)}$ è la probabilità ottenuta con la softmax di $e^{(t)}$
$a$ è la combinazione degli stati nascosti degli encoder

![[10)-20241119112420126.png]]
L'attenzione migliora le performance di Neural Machine Traslation (NMT)
Fornisce un modello human like del modello machine traslation, perchè possiamo guardare alla frase sorgente durante il processo piuttosto che doverla ricordare
risolve il problema di bottleneck
aiuta col problema del vanishing gradient
fornisce interpretabilità, ispezionando i pesi di attenzione possiamo sapere su quali parti il decoder si è concentrato step by step
![[10)-20241119112426025.png]]
si può utilizzare una dimensionalità nello stato nascosto diversa per l'encoder  e il decoder, anche se in pratica le due dimensionalità coincidono. per architetture encoding only può essere più basso
i valori di h(stato dell'encoder)
gli stati del decoder fanno da query a coppie chiave-valore che corrispondono ai stati dell'encoder
abbiamo tante query quante la lunghezza della seguenza generata dal decoder ($T$) e abbiamo $N$ chiavi, tante quando la lunghezza della parola  in input
![[10)-20241119112430591.png]]


### Attention as a general DL technique
![[10)-20241119112436419.png]]
Possiamo generalizzare il meccanismo di attenzione: noi abbiamo visto la cross attention
l'attenzione non è da utilizzare solo per architetture sequence to sequence
intendiamo l'attenzione come una tecnica generale per calcolare una somma pesata di valori condizionatamente a una certa query
questa somma è un summary selettivo dell'informazione contenuta nei valori (stati codifica), mentre la query determina su quali valori concentrarsi durante la generazione


![[Pasted image 20241119112446.png]]

