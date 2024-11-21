
Si usano reti riccorenti stacked (multy layer) e meccanismo di attenzione
il passo di forward guarda l input da sinistra a destra e codifica una località da un punto di vista lineare
l'impatto è dovuto alle parole adiacenti, ovvero quelle che costituiscono il suo contesto
una rnn ha bisogno di un numero di step grande quando la codifica del testo in input per poter far interagire le due parti
![[11) Transormers-20241121102221889.png]]
quando devo codificare lo stato al passo t, il num minimo di step prima di calcolare questo è ...

![[11) Transormers-20241121102411711.png]]
con l'attenzione posso generalizzare con più layer
richiede uno step di trasformazione perchè ogni stato è confrontato con ogni altro
il meccanismo dell'attenzione richiede un numero quadratico di confronti

## self attention
![[11) Transormers-20241121102551928.png]]
è una sorta di lookup soft
in quello tradizionale abbiamo query e indice, con le proprie chiavi
col meccanismo di attenzione è simile, solo che data la query andiamo a valutare la rilevanza di ogni chiave e ne facciamo una somma pesata che agisce da summary selettivo

![[11) Transormers-20241121102753042.png]]
$w_{i}$ è l'iesima parola
$x_{i}$ è l'embedding della parola, ottenuto tramite una matrice di embedding di dimensione $d\times d$, dove d è la dimensionalità
questo viene trasformato in 3 modi: rispetto la matrice delle query $Q$, rispetto la matrice delle chiavi $K$ e rispetto la matrice dei valori $V$
self attention perchè ogni stato attende a qualcun altro
le similarità sono calcolate a coppie tra query e chiave ($e_{ij}$ è il prodotto tra l'i-esima query e la j-esima chiave)
l'isima parola target fa da query e vogliamo confrontarla con tutte le altre, che avranno il ruolo di chiave
otteniamo la probabilità con la softmax e facciamo la somma pesata


barriere e soluzioni di self attention
self attention stiamo facendo in modo che ogni stato che possa considerare ogni altro token nella propria codifica. sarebbe bene anche integrare il concetto di posizione
![[11) Transormers-20241121103602495.png]]
$p_i$ è un vettore posizionale di dimensione $d$, vogliamo incorporare questo vettore posizionale all'interno del blocco di self attention. andiamo quindi a sommare l'embedding del token i esimo $x_i$  con il corrispondente vettore embedding posizionale $p_i$ 
$\tilde{x}_{i}=x_i+p_{i}$ 
a livello implementativo questa è la soluzione più usata


se adotto la **rappresentazione posizionale sinusoidale**, quello che conta non è la posizione assoluta ma quella relativa tra le parole
questa tecnica è difficilmente apprendibile.


![[11) Transormers-20241121105506777.png]]
## Components of self-attention as a building block
![[11) Transormers-20241121105511982.png]]
potrebbe essere utile inerire delle connessioni residue 

![[11) Transormers-20241121105739624.png]]
![[11) Transormers-20241121105831215.png]]
è una normalizzazione dimensione per dimensione
l'obiettivo è rendere stabile il learning, all'interno di ogni layer
se abbiamo un vettore di dimensione d possiamo standardizzare 
questo viene indicato come blocco add & norm nel transformer 
ciò che rende il training efficiente è la pre normalizzazione e non la post normalizzazione
## Sequence-Stacked form of Attention
![[11) Transormers-20241121110337027.png]]
fa coesistere la profondità della rete col concetto di molteplcità delle teste
un transformer è un'evoluzione della rnn, è una rete multi layer ma anche multi head
avere più teste significa avere più matrici query valore. ogni testa lavora in parlallelo insieme alle altre, non ha impatto a livello di tempo
aggiungere una complessità ad ogni layer porta a offrire più interpretazioni di come si combinano le chiavi con le query
ha senso visto che gli input sono parole e hanno una polisemia intrinseca
i layer stabiliscono un controllo sulla tipologia di relazione tra le parole

introduciamo un iperparametro che è il numero di teste, e questo vale per ogni layer
date le matrici k,q,v  di dimensione $d \times d$ , possiamo riscrivere questa cosa con la matrice x
avviene la trasformazioen con la matrice delle query e delle chiavi, la softmax e poi la combinazione con la somma pesata
se invece di avere una matrice abbiamo un tensore il cui spessore è il numero di head

![[11) Transormers-20241121111639444.png]]

postriamo distribuire la multimensionalità sulle h teste
ogni matrice di trasformazione di una testa ha shape $h \times d \times n$

![[11) Transormers-20241121111739483.png]]
tensore con ogni slice di dimensione $d \times d/h$ e le combina
calcoliamo la sofrmax aggiungendo un fattore di scaling perchè gli score di attenzione erano troppo alti
![[11) Transormers-20241121111935799.png]]
## The Transformer decoder
Now that we've replaced self-
attention with multi-head self-
attention, we'll go through two
optimization tricks that end up
being :
• Residual Connections
• Layer Normalization
In most Transformer diagrams,
these are often written
together as "Add & Norm"
![[11) Transormers-20241121112146306.png]]

il modulo di self attention è masked e multihead

se il transformer deve fare clasfficiazioone o sentiment analisys e non predizione di parola, non ha più senso mascherare la self attention

![[11) Transormers-20241121112251982.png]]
mettendo insieme encoder e decoder: abbiamo 3 tipi di attenzione
multi head self attention nel encoder
masked multi head self attention nel decoder
cross attention nel encoder, che attende all'output del decoder
l'outp di questa attenzione passa da un'ulteriore testa di attention add & norm

Cross attention
![[11) Transormers-20241121112426817.png]]

con h abbiamo la codifica finale dell'encoder

## Subword tokenization
unica preoccupazione a livello di pre processing da tenere in conto per la preparazione dell'input