
## Reti Ricorrenti Stacked e Meccanismo di Attenzione: Un Riassunto

Le reti ricorrenti (RNN) standard, pur efficaci, presentano limitazioni nell'elaborazione di sequenze lunghe.  Il loro meccanismo di elaborazione sequenziale, da sinistra a destra, implica un numero di operazioni non parallelizzabili proporzionale alla lunghezza della sequenza (O(Lunghezza della sequenza)), limitando l'efficienza su GPU e l'addestramento su dataset di grandi dimensioni.  Questo è visualizzato nell'immagine `![11) Transormers-20241122125219299.png]`, che mostra la dipendenza sequenziale nel calcolo degli stati nascosti.

Per ovviare a questo problema, si utilizzano reti ricorrenti stacked e, soprattutto, il meccanismo di **attenzione**.  A differenza delle RNN, l'attenzione permette a ogni parola di interagire con tutte le altre in parallelo, indipendentemente dalla loro posizione nella sequenza.  La distanza di interazione massima diventa O(1), come illustrato in `![11) Transormers-20241122125255115.png]`.

## Meccanismo di Self-Attention

L'attenzione può essere interpretata come una ricerca "fuzzy" in un archivio chiave-valore.  Ogni parola viene rappresentata da tre vettori: query (**q**), chiave (**k**) e valore (**v**), ottenuti trasformando l'embedding della parola tramite matrici di peso  `Q`, `K` e `V` rispettivamente.  Le similarità tra le query e le chiavi di tutte le parole vengono calcolate (prodotto scalare `e<sub>ij</sub> = q<sub>i</sub><sup>T</sup>k<sub>j</sub>`), normalizzate con softmax (`α<sub>ij</sub>`), e utilizzate come pesi per una somma pesata dei vettori valore (`o<sub>i</sub> = Σ<sub>j</sub> α<sub>ij</sub>v<sub>j</sub>`). Questo processo, illustrato nelle immagini `![11) Transormers-20241122130437426.png]` e `![11) Transormers-20241122130454760.png]`,  consente a ogni parola di "attendere" a tutte le altre, creando una rappresentazione contestuale più efficiente.  Le formule chiave sono:

```
q<sub>i</sub> = Qx<sub>i</sub>  (queries)
k<sub>i</sub> = Kx<sub>i</sub>  (keys)
v<sub>i</sub> = Vx<sub>i</sub>  (values)

e<sub>ij</sub> = q<sub>i</sub><sup>T</sup>k<sub>j</sub>
α<sub>ij</sub> = exp(e<sub>ij</sub>) / Σ<sub>j'</sub> exp(e<sub>ij'</sub>)

o<sub>i</sub> = Σ<sub>j</sub> α<sub>ij</sub>v<sub>j</sub>
```

dove  `x<sub>i</sub> = Ew<sub>i</sub>` è l'embedding della parola `w<sub>i</sub>`, e `E` è la matrice di embedding.  La self-attention, quindi, risolve il problema della dipendenza sequenziale delle RNN, permettendo un'elaborazione parallela e più efficiente di sequenze lunghe.

---

# Self-Attention: Meccanismi e Miglioramenti

Questo documento descrive il meccanismo di self-attention e le tecniche utilizzate per migliorarne le prestazioni.

## Self-Attention: Funzionamento e Limiti

La self-attention confronta ogni parola (query) di una sequenza con tutte le altre (chiavi), ottenendo una probabilità tramite la softmax e calcolando una somma pesata.  Un limite fondamentale è la mancanza di una nozione intrinseca dell'ordine delle parole.

## Miglioramenti alla Self-Attention

Per risolvere i limiti della self-attention, vengono adottate diverse strategie:

### Rappresentazioni Posizionali

L'ordine delle parole viene codificato aggiungendo vettori posizionali ($p_i$) agli embedding delle parole ($x_i$), creando embedding posizionati: $\tilde{x}_i = x_i + p_i$.  La rappresentazione posizionale sinusoidale considera la posizione relativa tra le parole, risultando meno facilmente apprendibile rispetto ad altre tecniche.

### Non Linearità

La self-attention, essendo una media pesata, manca di non linearità.  Per ovviare a questo, si applica una rete feed-forward all'output del blocco di self-attention.

### Mascheramento

Per evitare che il modello "veda" il futuro durante l'elaborazione sequenziale (es. in traduzione automatica), si utilizza il mascheramento, impostando a 0 i pesi di attenzione futuri.

## Componenti della Self-Attention

La self-attention si basa su tre componenti principali:

* **Rappresentazioni Posizionali:** Specificano l'ordine delle parole nella sequenza.
* **Non Linearità:** Applicate all'output tramite una rete feed-forward.
* **Mascheramento:** Impedisce al modello di accedere a informazioni future.  ![[]]

## Tecniche Aggiuntive per Migliorare l'Addestramento

### Connessioni Residuali

Per facilitare l'addestramento, si utilizzano connessioni residuali: $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$, permettendo al gradiente di fluire più facilmente e introducendo un bias verso la funzione identità.

### Normalizzazione per Layer (Layer Normalization)

La Layer Normalization stabilizza l'addestramento normalizzando i valori dei vettori nascosti a media unitaria e deviazione standard unitaria all'interno di ogni layer.  La formula è:

$$ \text{output} = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta $$

dove $\mu$ è la media, $\sigma$ la deviazione standard, $\gamma$ e $\beta$ sono parametri appresi, e $\epsilon$ evita divisioni per zero.  Questo processo normalizza dimensione per dimensione, modulando poi tramite guadagno e bias appresi.

---

Il testo descrive il meccanismo di attenzione multi-head nei Transformer, partendo dalla standardizzazione dei vettori di input (blocco "add & norm").  L'attenzione, applicata a una sequenza di vettori di input  $X \in \mathbb{R}^{n \times d}$ (dove *n* è la lunghezza della sequenza e *d* la dimensione del vettore),  calcola un output tramite le matrici di trasformazione per chiavi (K), query (Q) e valori (V):

$\text{output} = \text{softmax}(XQ(XK)^T)XV \in \mathbb{R}^{n \times d}$

Questo processo avviene in tre fasi:  1) calcolo dei prodotti scalari query-key ($XQ(XK)^T$); 2) applicazione della softmax per ottenere pesi di attenzione; 3) calcolo della media pesata dei valori (V) usando i pesi di attenzione.  Un fattore di scala $\sqrt{d}$ è incluso nella softmax per stabilizzare il gradiente. ![[11) Transormers-20241122125849498.png]]

L'attenzione multi-head estende questo concetto utilizzando *h* teste di attenzione, ciascuna con le proprie matrici $Q_\ell, K_\ell, V_\ell \in \mathbb{R}^{d \times d/h}$.  Invece di una singola matrice X, si usa un tensore, dove la terza dimensione rappresenta il numero di teste.  Le matrici $XQ$, $XK$, $XV$ vengono rimodellate in $\mathbb{R}^{n \times h \times d/h}$ e poi trasposte in $\mathbb{R}^{h \times n \times d/h}$ per parallelizzare il calcolo delle diverse teste.  Il costo computazionale non aumenta significativamente nonostante l'aumento delle teste. ![[11) Transormers-20241122125925709.png]]

L'utilizzo di più teste permette di "guardare" in più punti della sequenza contemporaneamente, catturando diverse relazioni tra le parole (polisemia), offrendo interpretazioni più ricche rispetto ad un'unica testa di attenzione.  Il numero di teste è un iperparametro del modello.

---

## Riassunto del Transformer Encoder-Decoder

Questo documento descrive l'architettura del Transformer, focalizzandosi sull'Encoder e sul Decoder e sul meccanismo di attenzione.

### Multi-Head Self-Attention

L'attenzione multi-testa opera indipendentemente su diverse rappresentazioni dell'input ($X$), utilizzando matrici di trasformazione separate ($Q_\ell, K_\ell, V_\ell$ per ogni testa $\ell$).  L'output di ogni testa ($\text{output}_\ell \in \mathbb{R}^{n \times d/h}$) è calcolato come:

$\text{output}_\ell = \text{softmax}(X Q_\ell K_\ell^T X^T) \times X V_\ell$

Gli output di tutte le teste vengono poi concatenati e trasformati tramite una matrice $Y \in \mathbb{R}^{d \times d}$:  $\text{output} = [\text{output}_1; ...;\text{output}_h]Y$.  Questo permette al modello di catturare diverse relazioni tra le parole dell'input.

Per migliorare la stabilità dell'addestramento, viene utilizzata l'**attenzione scalata con prodotto scalare**:

$\text{output}_{e} = \text{softmax}\left(\frac{X Q_{e} K_{e}^{\top} X^{\top}}{\sqrt{d/h}}\right) \times X V_{e}$

dove $d/h$ è la dimensionalità di ogni testa.

### Transformer Decoder

Il Decoder del Transformer è composto da una pila di blocchi identici. Ogni blocco include:

1. **Masked Multi-Head Self-Attention:**  Permette al decoder di focalizzarsi su diverse parti dell'input corrente, impedendo la "visione" di parole future (cruciale per la generazione sequenziale).
2. **Add & Norm:**  Aggiunge l'output della self-attention all'input e lo normalizza (connessione residuale e normalizzazione per layer).
3. **Feed-Forward Network:**  Una rete neurale feed-forward che trasforma ulteriormente l'input.
4. **Add & Norm:**  Analogo al punto 2.

![11) Transormers-20241122130213176.png]

Se il Transformer non genera sequenze (es. classificazione), la maschera sulla self-attention non è necessaria.


### Transformer Encoder-Decoder

Per compiti seq2seq come la traduzione automatica, si utilizza un Encoder-Decoder.

* **Encoder:** Un Transformer standard che elabora l'input (es. frase sorgente) bidirezionalmente.
* **Decoder:** Un Transformer modificato che include, oltre alla self-attention mascherata, la **cross-attention**.  Questa attenzione usa le query dal decoder e le chiavi e i valori dall'output dell'encoder, permettendo al decoder di focalizzarsi sulle parti rilevanti dell'input dell'encoder.

![11) Transormers-20241122130245556.png]

In sintesi, il Transformer Encoder-Decoder utilizza tre tipi di attenzione: multi-head self-attention nell'encoder, masked multi-head self-attention nel decoder e cross-attention dal decoder all'encoder. L'output della cross-attention passa attraverso un ulteriore blocco Add & Norm.

---

Il testo descrive un modello di tipo Transformer, focalizzandosi sul meccanismo di attenzione.  L'encoder produce vettori di output $h_1, \dots, h_n \in \mathbb{R}^d$, mentre il decoder riceve vettori di input $z_1, \dots, z_n \in \mathbb{R}^d$.  Chiavi ($k_i$) e valori ($v_i$) sono derivati dai vettori dell'encoder tramite matrici di trasformazione $K$ e $V$ ($k_i = Kh_j$, $v_i = Vh_j$), mentre le query ($q_i$) sono generate dai vettori del decoder tramite una matrice $Q$ ($q_i = Qz_i$).  L'immagine `![[]]` illustra probabilmente questo processo.  Infine, il testo evidenzia che la tokenizzazione subword è l'unica fase di pre-processing rilevante.  Il vettore `h` rappresenta la codifica finale dell'encoder.

---
