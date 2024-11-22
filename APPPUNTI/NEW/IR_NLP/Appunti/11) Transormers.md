## Reti Ricorrenti Stacked e Meccanismo di Attenzione

Si utilizzano reti ricorrenti stacked (multi-layer) e un meccanismo di attenzione.  Il passo forward di una RNN standard legge l'input da sinistra a destra, codificando ogni località da un punto di vista lineare. L'impatto di una parola sulla rappresentazione è determinato dalle parole adiacenti, ovvero dal suo contesto.  Una RNN richiede un numero elevato di step per codificare testi lunghi, permettendo l'interazione tra parti distanti del testo.  Questo presenta un problema significativo:

**Il passo di forward e quello di backward hanno O (Lunghezza della sequenza) operazioni non parallelizzabili**

* Le GPU possono eseguire molte computazioni indipendenti contemporaneamente.
* Tuttavia, gli stati nascosti futuri di una RNN non possono essere calcolati completamente prima che siano stati calcolati gli stati nascosti passati.
* Questo limita l'addestramento su dataset molto grandi.

![[11) Transormers-20241122125219299.png]]

**Numbers indicate min # of steps before a state can be computed**

Il numero minimo di step prima di poter calcolare lo stato al passo *t* dipende dalla lunghezza della sequenza.

## Meccanismo di Attenzione

L'attenzione tratta la rappresentazione di ogni parola come una query per accedere e incorporare informazioni da un insieme di valori.

* Abbiamo visto l'attenzione dal decoder all'encoder; oggi considereremo l'attenzione all'interno di una singola frase.
* Il numero di operazioni non parallelizzabili non aumenta con la lunghezza della sequenza.
* Distanza di interazione massima: O(1), poiché tutte le parole interagiscono a ogni livello!

![[11) Transormers-20241122125255115.png]]

**All words attend to all words in previous layer; most arrows here are omitted**

Con l'attenzione, è possibile generalizzare con più layer.  Richiede uno step di trasformazione perché ogni stato viene confrontato con ogni altro stato. Il meccanismo di attenzione richiede un numero quadratico di confronti.

## Self-Attention

Possiamo pensare all'attenzione come a una ricerca "fuzzy" in un archivio chiave-valore.

In una tabella di ricerca, abbiamo una tabella di chiavi che mappano a valori. La query corrisponde a una delle chiavi, restituendo il suo valore.

![[11) Transormers-20241122130437426.png]]

Nell'attenzione, la query corrisponde a tutte le chiavi "softly", con un peso compreso tra 0 e 1. I valori delle chiavi vengono moltiplicati per i pesi e sommati.

![[11) Transormers-20241122130454760.png]]

È una sorta di lookup "soft". Nel lookup tradizionale abbiamo query e indice, con le proprie chiavi. Con il meccanismo di attenzione è simile, solo che data la query valutiamo la rilevanza di ogni chiave e ne facciamo una somma pesata che agisce come un riassunto selettivo.


Sia $\mathbf{w}_{1:n}$ una sequenza di parole nel vocabolario $V$, ad esempio "Zuko preparò il tè per suo zio".

Per ogni $\mathbf{w}_{i}$, sia $\mathbf{x}_{i} = E \mathbf{w}_{i}$, dove $E \in \mathbb{R}^{d \times |V|}$ è una matrice di embedding.

1. Trasformiamo ogni embedding di parola con matrici di peso $Q, K, V$, ciascuna in $\mathbb{R}^{d \times d}$:

$$
\begin{aligned}
\mathbf{q}_{i} &= Q \mathbf{x}_{i} \quad \text{(queries)} \\
\mathbf{k}_{i} &= K \mathbf{x}_{i} \quad \text{(keys)} \\
\mathbf{v}_{i} &= V \mathbf{x}_{i} \quad \text{(values)}
\end{aligned}
$$

2. Calcoliamo le similarità a coppie tra chiavi e query; normalizziamo con softmax:

$$
\begin{aligned}
e_{i j} &= \mathbf{q}_{i}^{\top} \mathbf{k}_{j} \\
\alpha_{i j} &= \frac{\exp\left(e_{i j}\right)}{\sum_{j^{\prime}} \exp\left(e_{i j^{\prime}}\right)}
\end{aligned}
$$

3. Calcoliamo l'output per ogni parola come somma pesata dei valori:

$$
\mathbf{o}_{i} = \sum_{j} \alpha_{i j} \mathbf{v}_{j}
$$

$w_{i}$ è l'i-esima parola; $x_{i}$ è l'embedding della parola, ottenuto tramite una matrice di embedding di dimensione $d \times |V|$, dove $d$ è la dimensionalità e $|V|$ la dimensione del vocabolario. Questo viene trasformato in tre modi: rispetto alla matrice delle query $Q$, rispetto alla matrice delle chiavi $K$ e rispetto alla matrice dei valori $V$. È chiamata self-attention perché ogni stato "attende" a tutti gli altri. Le similarità sono calcolate a coppie tra query e chiave ($e_{ij}$ è il prodotto scalare tra l'i-esima query e la j-esima chiave). L'i-esima parola target funge da query e viene confrontata con tutte le altre, che hanno il ruolo di chiave. Otteniamo la probabilità con la softmax e facciamo la somma pesata.


### Barriere e Soluzioni della Self-Attention

La self-attention non tiene conto intrinsecamente dell'ordine delle parole; quindi, dobbiamo codificare l'ordine delle parole nelle chiavi, query e valori.

* Consideriamo ogni indice di sequenza come un vettore $p_i \in \mathbb{R}^d$, per $i \in \{1,2,...,n\}$, che rappresentano i vettori di posizione.
* Non preoccupatevi di cosa siano fatti i $p_i$ per ora!
* È facile incorporare queste informazioni nel nostro blocco di auto-attenzione: basta aggiungere i $p_i$ ai nostri input!
* Ricordate che $x_i$ è l'incorporazione della parola all'indice $i$. L'incorporazione posizionata è:

$$\tilde{x}_i = x_i + p_i$$

* In profonde reti di auto-attenzione, facciamo questo al primo strato. Potreste concatenarli, ma la maggior parte delle persone li aggiunge semplicemente.

$p_i$ è un vettore posizionale di dimensione $d$; vogliamo incorporare questo vettore posizionale all'interno del blocco di self-attention. Sommiamo l'embedding del token i-esimo $x_i$ con il corrispondente vettore embedding posizionale $p_i$: $\tilde{x}_{i}=x_i+p_{i}$. A livello implementativo questa è la soluzione più usata.

Se adottiamo la **rappresentazione posizionale sinusoidale**, quello che conta non è la posizione assoluta ma quella relativa tra le parole. Questa tecnica è difficilmente apprendibile.


**Barriere:**

* **Ordine delle parole:** Non ha una nozione intrinseca di ordine!  *Soluzioni:* Aggiungere rappresentazioni posizionali agli input.
* **Non linearità:** Non ci sono non linearità per la "magia" del deep learning. Sono solo medie pesate.  *Soluzioni:* Applicare la stessa rete feedforward a ciascun output della self-attention.
* **Futuro:** Bisogna assicurarsi di non "guardare al futuro" quando si prevede una sequenza (come nella traduzione automatica o nel language modeling).  *Soluzioni:* Mascherare il futuro impostando artificialmente i pesi dell'attenzione a 0.

## Componenti della Self-Attention come Blocco Costruttivo

La **self-attention** è alla base del metodo.

**Rappresentazioni Posizionali**
Specificano l'ordine della sequenza, poiché la self-attention è una funzione non ordinata dei suoi input.

**Non Linearità**
Sono applicate all'output del blocco di self-attention e sono spesso implementate come una semplice rete feed-forward.

**Mascheramento**
Serve per parallelizzare le operazioni senza "guardare al futuro". Impedisce che le informazioni sul futuro "trapelino" nel passato.

![[11) Transormers-20241122125633686.png]]

Potrebbe essere utile inserire connessioni residuali.

### Connessioni Residuali

Le connessioni residuali sono un trucco per migliorare l'addestramento dei modelli.

* Invece di $X^{(i)} = \text{Layer}(X^{(i-1)})$ (dove $i$ rappresenta il layer):

   $X^{(i-1)} \xrightarrow{\text{Layer}} X^{(i)}$

* Si usa $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$ (quindi si deve imparare solo "il residuo" dal layer precedente):

   $X^{(i-1)} \xrightarrow{\text{Layer}} X^{(i)}$

* Il gradiente è elevato attraverso la connessione residuale; è 1!

* Bias verso la funzione identità!


### Normalizzazione per Layer (Layer Normalization)

La normalizzazione per layer è un trucco per velocizzare l'addestramento dei modelli.

**Idea:** Ridurre la variazione non informativa nei valori dei vettori nascosti normalizzando a media unitaria e deviazione standard all'interno di ogni layer.

**Successo della LayerNorm:**

Il successo della LayerNorm potrebbe essere dovuto alla normalizzazione dei gradienti.

**Formulazione Matematica:**

Sia $x \in \mathbb{R}^d$ un singolo vettore (di parola) nel modello.

Sia $\mu = \frac{1}{d} \sum_{j=1}^d x_j$; questa è la media; $\mu \in \mathbb{R}$.

Sia $\sigma = \sqrt{\frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2}$; questa è la deviazione standard; $\sigma \in \mathbb{R}$.

Siano $y \in \mathbb{R}^d$ e $\beta \in \mathbb{R}^d$ parametri di "guadagno" e "bias" appresi (possono essere omessi!).

Allora la normalizzazione per layer calcola:

$$
\text{output} = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$$

dove $\gamma$ e $\beta$ sono parametri di "guadagno" e "bias" appresi.  $\epsilon$ è un piccolo valore aggiunto per evitare divisioni per zero.

**Processo di Normalizzazione:**

* Normalizzazione tramite media e varianza scalari.
* Modulazione tramite guadagno e bias elementari appresi.

È una normalizzazione dimensione per dimensione. L'obiettivo è stabilizzare l'apprendimento all'interno di ogni layer. Se abbiamo un vettore di dimensione d, possiamo standardizzarlo. Questo è indicato come blocco "add & norm" nel Transformer. Ciò che rende l'addestramento efficiente è la pre-normalizzazione e non la post-normalizzazione.

## Sequence-Stacked Attention

### Definizione delle Variabili

* $X = [x_1; x_2; ...; x_n] \in \mathbb{R}^{n \times d}$: concatenazione dei vettori di input, dove  `n` è il numero di elementi nella sequenza e `d` la dimensione di ogni vettore.
* $X_K \in \mathbb{R}^{n \times d}$, $X_Q \in \mathbb{R}^{n \times d}$, $X_V \in \mathbb{R}^{n \times d}$: matrici ottenute trasformando X tramite le matrici di trasformazione per le chiavi (K), le query (Q) e i valori (V) rispettivamente.

### Calcolo dell'Output

L'output è definito come:

$$\text{output} = \text{softmax}(XQ(XK)^T)XV \in \mathbb{R}^{n \times d}$$

### Passaggi del Calcolo

1. **Calcolo dei prodotti scalari query-key:** $XQ(XK)^T$ produce una matrice di prodotti scalari tra le query e le chiavi.

2. **Applicazione della funzione softmax:** $\text{softmax}(XQ(XK)^T)$ normalizza i prodotti scalari, generando una matrice di pesi di attenzione.

3. **Calcolo della media pesata:** $\text{softmax}(XQ(XK)^T)XV$ calcola la media pesata dei valori, utilizzando i pesi di attenzione calcolati nel passaggio precedente.


![[11) Transormers-20241122125849498.png]]

Questo approccio fa coesistere la profondità della rete con il concetto di molteplicità delle teste. Un Transformer è un'evoluzione delle RNN: è una rete multi-layer ma anche multi-head. Avere più teste significa avere più matrici query-valore. Ogni testa lavora in parallelo con le altre, senza impatto sul tempo di calcolo. Aggiungere complessità a ogni layer offre più interpretazioni di come si combinano le chiavi con le query. Questo è sensato dato che gli input sono parole, intrinsecamente polisemiche. I layer stabiliscono un controllo sulla tipologia di relazione tra le parole.

Introduciamo un iperparametro, il numero di teste, valido per ogni layer. Date le matrici K, Q, V di dimensione $d \times d$, possiamo riscrivere il calcolo usando la matrice X.  Avviene la trasformazione con le matrici delle query e delle chiavi, la softmax e poi la combinazione con la somma pesata. Se invece di avere una matrice abbiamo un tensore, il cui spessore rappresenta il numero di teste, il calcolo si estende naturalmente.


* Anche calcolando molte teste di attenzione, il costo computazionale non aumenta significativamente.
* Calcoliamo $XQ \in \mathbb{R}^{n \times d}$ e poi riformatiamo a $\mathbb{R}^{n \times h \times d / h}$ (allo stesso modo per $XK$, $XV$), dove `h` è il numero di teste.
* Quindi trasponiamo a $\mathbb{R}^{h \times n \times d / h}$: ora l'asse della testa è come un asse batch.
* Quasi tutto il resto è identico e le matrici hanno le stesse dimensioni.


### Calcolo dei Prodotti Scalari Query-Key

Prima, prendiamo i prodotti scalari query-key con una moltiplicazione di matrice: $XQ(XK)^T$.  Calcoliamo così tutti i set di coppie di punteggi di attenzione.

### Calcolo della Media Ponderata

Successivamente, calcoliamo la media ponderata con un'altra moltiplicazione di matrice: $\text{softmax}\left(\frac{XQ(XK)^T}{\sqrt{d}}\right) XV$.  Il termine $\sqrt{d}$ è un fattore di scala per stabilizzare il gradiente.

### Output

L'output è una matrice di dimensioni $n \times d$.

![[11) Transormers-20241122125925709.png]]

Possiamo distribuire la multidimensionalità sulle h teste. Ogni matrice di trasformazione di una testa ha forma $h \times d \times n$.

## Attenzione Multi-Head

**Problema:** Cosa succede se vogliamo "guardare" in più punti della frase contemporaneamente?

* Per la parola *i*, la self-attention "guarda" dove $x_i^T Q^T K x_j$ è alto, ma forse vogliamo concentrarci su diversi *j* per motivi diversi?

**Soluzione: Definiamo più "teste" di attenzione tramite molteplici matrici Q, K, V.**

* Siano $Q_\ell, K_\ell, V_\ell \in \mathbb{R}^{d \times d/h}$, dove *h* è il numero di teste di attenzione e *ℓ* varia da 1 a *h*.

**Ogni testa di attenzione esegue l'attenzione in modo indipendente:**

* $\text{output}_\ell = \text{softmax}(X Q_\ell K_\ell^T X^T) \times X V_\ell$, dove $\text{output}_\ell \in \mathbb{R}^{n \times d/h}$  (si noti che l'output ha dimensione n x d/h, dove n è il numero di parole nella sequenza).

**Quindi gli output di tutte le teste vengono combinati:**

* $\text{output} = [\text{output}_1; ...;\text{output}_h]Y$, dove $Y \in \mathbb{R}^{d \times d}$ è una matrice di trasformazione che combina gli output delle diverse teste.

**Ogni testa può "guardare" cose diverse e costruire vettori di valori in modo diverso.**  Il risultato è un tensore con ogni slice di dimensione $d \times d/h$, che poi vengono combinate.


### Attenzione Scalata con Prodotto Scalare (Scaled Dot Product Attention)

L'attenzione scalata con prodotto scalare aiuta l'addestramento.

* Quando la dimensionalità *d* diventa grande, i prodotti scalari tra i vettori tendono a diventare grandi.
* Di conseguenza, gli input alla funzione softmax possono essere grandi, rendendo i gradienti piccoli.

Invece della funzione di self-attention vista prima:

$$
\text{output}_{e} = \text{softmax}\left(X Q_{e} K_{e}^{\top} X^{\top}\right) \times X V_{e}
$$

Dividiamo gli score di attenzione per $\sqrt{d/h}$, per evitare che gli score diventino grandi solo in funzione di $d/h$ (la dimensionalità divisa per il numero di teste).

$$
\text{output}_{e} = \text{softmax}\left(\frac{X Q_{e} K_{e}^{\top} X^{\top}}{\sqrt{d/h}}\right) \times X V_{e}
$$

Questo scaling migliora la stabilità del training, evitando gradienti troppo piccoli o troppo grandi.

## Il Decoder del Transformer

Dopo aver sostituito la self-attention con la multi-head self-attention, analizziamo due ottimizzazioni:

* **Connessioni Residuali:** Aggiungono l'input al risultato di un layer, migliorando il flusso del gradiente.
* **Normalizzazione per Layer:** Normalizza i valori dei vettori nascosti a media unitaria e deviazione standard unitaria, stabilizzando l'addestramento.

Nella maggior parte dei diagrammi dei Transformer, queste sono spesso scritte insieme come "Add & Norm".


### Struttura del Transformer Decoder

Il Transformer Decoder è costituito da una pila di blocchi di decodifica. Ogni blocco contiene:

* **Self-Attention:** Permette al modello di focalizzarsi su diverse parti dell'input, assegnando pesi diversi alle parole in base alla loro importanza.  Nel decoder, questa è una *masked self-attention*, che impedisce al modello di "vedere" le parole future durante la generazione sequenziale.
* **Add & Norm:** Aggiunge l'output della self-attention all'input e normalizza il risultato.
* **Feed-Forward:** Rete neurale feed-forward che trasforma l'input.
* **Add & Norm:** Aggiunge l'output della rete feed-forward all'input e normalizza il risultato.

### Funzionamento del Transformer Decoder

1. L'input viene inserito nel primo blocco di decodifica.
2. La self-attention assegna pesi alle parole.
3. I pesi vengono aggiunti all'input e normalizzati (Add & Norm).
4. Il risultato passa attraverso la rete feed-forward.
5. L'output della rete feed-forward viene aggiunto all'input e normalizzato (Add & Norm).
6. Il processo si ripete per ogni blocco.

![[11) Transormers-20241122130213176.png]]

Il modulo di self-attention è mascherato (masked) e multi-head. Se il Transformer esegue classificazione o analisi del sentiment (e non predizione di parole), non è più necessario mascherare la self-attention.


### Il Transformer Encoder-Decoder

* Nella traduzione automatica, la frase sorgente viene elaborata con un modello bidirezionale e la frase target viene generata con un modello unidirezionale.
* Per questo tipo di formato seq2seq, si usa spesso un Transformer Encoder-Decoder.
* Si usa un normale Transformer Encoder.
* Il Transformer Decoder è modificato per eseguire la cross-attention sull'output dell'Encoder.

![[11) Transormers-20241122130245556.png]]

Combinando encoder e decoder, abbiamo tre tipi di attenzione:

* Multi-head self-attention nell'encoder.
* Masked multi-head self-attention nel decoder.
* Cross-attention dal decoder all'output dell'encoder. L'output di questa attenzione passa attraverso un ulteriore blocco Add & Norm.


### Cross-Attention

* Nella self-attention, chiavi, query e valori provengono dalla stessa sorgente.
* Nel decoder, l'attenzione è simile a quella vista precedentemente, ma le query provengono dal decoder e le chiavi e i valori provengono dall'encoder.
* Siano $h_{1},\dots,h_n$ i vettori di output dell'encoder ($h_i \in \mathbb{R}^d$) e $z_1, \ldots, z_n$ i vettori di input del decoder ($z_i \in \mathbb{R}^d$).
* Chiavi e valori sono presi dall'encoder (come una memoria): $k_i = K h_j$, $v_i = V h_j$.
* Le query sono prese dal decoder: $q_i = Q z_i$.

![[11) Transormers-20241122130352942.png]]

`h` rappresenta la codifica finale dell'encoder.


## Tokenizzazione Subword

L'unica preoccupazione a livello di pre-processing da tenere in considerazione per la preparazione dell'input.
