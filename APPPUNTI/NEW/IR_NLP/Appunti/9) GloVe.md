## GloVe: Un nuovo modello di regressione log-bilineare globale

**GloVe** introduce un nuovo modello di regressione log-bilineare globale che combina i vantaggi sia della fattorizzazione matriciale globale che dei metodi di finestra di contesto locale.

**Miglioramenti:**

* **Metodi di fattorizzazione matriciale globale (es. LSA):**
    * Scarsa performance nel compito di analogia delle parole.
    * Le parole frequenti contribuiscono in modo sproporzionato alla misura di similarità.
* **Metodi di finestra di contesto locale, poco profondi:**
    * Non operano direttamente sui conteggi di co-occorrenza globali (cioè, usano solo parole adiacenti), non possono sfruttare le statistiche del corpus.

**GloVe supera significativamente tutti gli altri modelli:**

* Restituisce risultati decrescenti per vettori di dimensioni superiori a 200.
* Finestre di contesto piccole e asimmetriche (finestra di contesto solo a sinistra) funzionano meglio per i compiti sintattici.
* Finestre di contesto lunghe e simmetriche (finestra di contesto su entrambi i lati) funzionano meglio per i compiti semantici.
* I compiti sintattici hanno beneficiato di un corpus più ampio, mentre i compiti semantici hanno ottenuto risultati migliori con Wikipedia invece di Gigaword5, probabilmente a causa della completezza di Wikipedia e della natura leggermente obsoleta di Gigaword5.
* Le prestazioni di Word2vec diminuiscono se il numero di campioni negativi aumenta oltre circa 10.
* Per lo stesso corpus, vocabolario e dimensione della finestra, GloVe ottiene costantemente risultati migliori e più veloci.

## Come valutare i vettori di parole?

Relativamente alla valutazione generale in NLP: **intrinseca vs. estrinseca**

**Intrinseca:**

* Valutazione su un sottocompito specifico/intermedio.
* Veloce da calcolare.
* Aiuta a comprendere il sistema.
* Non è chiaro se sia realmente utile a meno che non sia stabilita una correlazione con un compito reale.

**Estrinseca:**

* Valutazione su un compito reale.
* Può richiedere molto tempo per calcolare l'accuratezza.
* Non è chiaro se il sottosistema sia il problema o la sua interazione con altri sottosistemi.
* Se la sostituzione di un solo sottosistema con un altro migliora l'accuratezza -5 Winningl.

## Analogie di vettori di parole

**uomo:donna :: re:?**
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114093845201.png]]
un termine sta ad un altro come un terzo termine c(probe) sta a un termine d

Valutare i vettori di parole in base a quanto bene la loro distanza coseno dopo l'addizione cattura domande intuitive di analogia semantica e sintattica.

vogliamo massimizzare la differenza tra l'addizione b a c e confrontarlo con un candidato i e normalizzare 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114093852413.png]]

Se faccio king-man otterrei lo stesso risulato di fare queen-woman
**Escludere le parole di input dalla ricerca**

**Problema:** Cosa succede se l'informazione è presente ma non lineare? 

### Similarità del significato

Questi metodi vengono valutati confrontando il risultato della prossimità nello spazio della rappresentazione con alcuni riferimenti.
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114101143315.png]]
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114101307393.png]]
GloVe performa meglio degli altri modelli, anche di CBOW che ha dimensionalità doppia.

**Valutazione dei modelli su determinati task:**
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114101431436.png]]

## La polisemia delle parole

La maggior parte delle parole ha molti significati.
* Soprattutto le parole comuni.
* Soprattutto le parole che esistono da molto tempo.

**Esempio: Pike**

* Un punto o un bastone affilato.
* Una linea o un sistema ferroviario.
* Una strada.

Avendo a che fare con rappresentazioni polisemiche, questa rappresentazione nello spazio deve catturare tutti i significati delle parol.
Un singolo vettore riesce a catturare tutti questi significati o abbiamo un pasticcio? 

Contesto globale e contesto locale delle parole per catturare tutti i significati. Per ogni parola, eseguire un clustering K-means per clusterizzare i contesti di occorrenza. rappresentazione densa attraverso compressione matrice LSA e questo diventa input del k-means. 
Costruiamo questi cluster di vettori di contesto e all iterazione successiva li utilizziamo per aggiornare i pesi. Invece di avere un unico vettore di embeddings per parola, ne abbiamo tanti quanti sono i contesti della parola

## Named Entity Recognition
Il task è trovare e classificare nomi nei testi, etichettando toker di parole
categorie lessicali. è importante fare alcuni step di pre processing: Come trattiamo le compound word e gli acronimi (sigle) 
generalizzando, si parla di rendere in forma canonica un termine: significa esprimere le entità tutte allo stesso modo (stabilire quindi una convenzione) 

si risolve con classificazione binaria
addestriamo un classificatore logistico su dati annotati (ci serve la y) per poter classificare una parola centrale per ogni classe (entity type)
potremmo anche trattare il problema come classificazione multiclasse
esempio:
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114113859881.png]]
nel caso binario: decisa una dimensione di contesto(nell'esempio 2), per la parola target (nell'esempio Paris), si costruisce il vettore x_{windows} con associata la classe Paris

potremmo servircene per risolvere altri problemi (Ad esempio sentyment analisys)


## Training con la loss corss entropy
Until now, our objective was stated as to maximize the probability of the correct class y
or equivalently we can minimize the negative log probability of that class
Now restated in terms of cross entropv
Let the true probability distribution be p; Iet our computed model probability be q

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114105648796.png]]
in questo caso y può assumere uno dei valori di c.
la funzione di costo è binary cross entropy: i valori che può assumere y sono {0,1}. per ogni input abbiamo quindi solo 1 dei due termini attivati. 


![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110129948.png]]

softmax in questo caso ègeneralizzazione del regressore logistico a più classi
La predizione, dato x di y, è una funzione softmax in cui vogliamo massimizzare l'allineamento dell'iostanza x con la matrice dei pesi per la predizione della classe y e poi normalizziamo per la somma sulle varie classi per avere un valore pari a 1
w è parte di $\theta$ 
una classificazione di rete neurale apprende sia la rappresentazione delle parole sia la matrice dei pesi che è ciò che guida il classificatore softmax

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110412410.png]]
per ogni esempio di training, massimizzare la probabilità della classe y o minimizzare la negative log likehood di quella stessa classe


## Classificatore lineare neurale

è un regressore logistico, dati input x utilizziamo una funzione di trasformazione (di attivazione) sigmoide. gli input x sono trasformati in accordo ai parametri da apprednere (pesi che regolano la pendenza della superficie di boundary e la sua traslazione nello spazio (bias))

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110524210.png]]
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110636182.png]]
abbiamo input (istanza di training che data una parola centrale di lunghezza contesto più 1), trasformazione in accordo a una funzione f per riconoscere non linearità (sigmoide, logistica, tangente iperbolica, ReLU o altre). il risultato è indicato con h
poi si allinea la trasformazione con il vettore su cui viene calcolata la logistica per il calcolo delle probabilità 

## Neural network
If we feed a vector of inputs through a bunch of logistic regression functions, then we get a vector of outputs which we can feed into another logistic regression function, giving composed functions.

le reti neurali sono multilayer, noi vogliamo abilitare il modello a catturare relazioni non lineari tra gli oggetti
aggiungendo al percettrone uno strato di rappresentazione intermedia riusciamo a gestire le non-linearità,
più pronda è la rete maggiore è l'abilità del modello di catturare non linearittà per proprietà diverse
layer più vicini all'input riescono a catturare relazioni non lineari a livello sintattico tra le parole (determinato da un focus relativo al contesto)
quando ci allonaniamo dall input catturiamo relazioni non lineari tra contesti, dunque allondanandoci dal target catturiamo relazioni semantiche tra una parola e altre distanti dal contesto

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114111604129.png|542]]

REgolarizzazione
legata alla complessità del modello.  vogliamo evitare overfitting
lasso, ridge e elastic net

dropout
evita l overgfitting riducendo la co feature adaptation: le varie dimensioni che rappresentrano l input non vogliamo che una dimensione contribuisca alla rappresentaszione di un oggetto in funzioen alla relazione con un altra relazione
disattivando alcuni neuroni evitiamo che le dimensioni possano lavorare in maniera congiunta, come se fosse indivisibile. 

vectorization
nota: utilizzare trasformazioni matriciali 


inizializzazione dei paramtri:
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114112627802.png]]
inizializzazione di xavier: come alternativa all inizializzazione dei param da una distribuzione uniforme su range troppo piccoli per favorire medie intorno allo 0.
ha una varianza proporzionale al fan-in(numero di neuroni del layer precedente) e al fan-out(numero di neuroni del layer successivo)
$$var(W_{i})=\frac{2}{n_{in}+n_{out}}$$
