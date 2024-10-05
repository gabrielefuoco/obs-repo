Il **core** di un CSP (Constraint Satisfaction Problem) è la versione più semplice e ridotta del problema originale, ottenuta attraverso un processo di semplificazione. 
Immagina le variabili e le relazioni di un CSP come un database di fatti (tuple). Quando troviamo una struttura ridondante o relazioni tra variabili che possono essere "compresse", possiamo ridurre il numero di vincoli senza perdere informazioni importanti.

Per esempio, se in un database ho le relazioni *a -> b* e *b -> d*, posso ridurre queste a *a -> d*, semplificando il problema. Se applico una mappatura (endomorfismo) che collega due variabili, posso ottenere un sottoproblema più piccolo e più semplice da risolvere.

Il **core** è quindi questa struttura ridotta: il più piccolo sottoproblema che mantiene la stessa essenza del problema originale. Un core esiste quando possiamo trovare un **endomorfismo** (una funzione che mappa gli elementi di una struttura su se stessa) che semplifica il problema senza perdere informazioni essenziali

In altre parole:
- Il **core** è la versione più piccola di un CSP che contiene tutte le informazioni necessarie per risolvere il problema originale.
- È unico a meno di **isomorfismi** (cioè, possono esistere più core equivalenti, ma sono tutti strutturalmente uguali a meno delle etichette delle variabili). 

La soluzione del problema semplificato può essere usata per risolvere il problema originale, semplificando il processo di risoluzione.