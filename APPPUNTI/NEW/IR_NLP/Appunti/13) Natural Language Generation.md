## Natural Language Generation

La generazione del linguaggio naturale (*NLG*) è un ramo dell'elaborazione del linguaggio naturale (*NLP*). Possiamo definire la relazione tra *NLG* e *NLP* come segue:

##### NLP = Comprensione del Linguaggio Naturale (NLU) + Generazione del Linguaggio Naturale (NLG)

L'obiettivo della *NLG* è sviluppare sistemi in grado di produrre output linguistici fluidi, coerenti e utili per gli esseri umani.

![[13) Natural Language Generation-20241203100245027.png|600]]

Open-ended generation: the output distribution Still has high freedom
Non-open-ended generation: the input mostly determines the output
generation.
Remark: One way of formalizing categorization this is by entropy.
These two classes of NLG tasks require different decoding and/or training
approaches!

## Meccanismo di Produzione dell'Output

### Greedy Decoding

Il *greedy decoding* seleziona il token che, dato il contesto corrente, massimizza la *negative log-likelihood*. In altre parole, sceglie il token con la probabilità più alta:

$$\hat{y}_{t}=\arg\max_{w\in V}P(y_{t}=w|y_{<t})$$

Questa strategia è preferibile nella maggior parte dei modelli, soprattutto per task classici di encoder-decoder come la traduzione automatica e la summarizzazione.

### Beam Search Decoding

In alternativa al *greedy decoding*, il *beam search decoding* genera contemporaneamente più traiettorie di generazione (ipotesi). Il numero di traiettorie da esplorare è un parametro controllabile. Il modello si ferma quando tutte le traiettorie sono completate. L'output finale è la traiettoria con la probabilità cumulativa più alta.

Il *beam search* è la scelta migliore quando il modello viene utilizzato in contesti aperti e astratti.

### Ridurre le ripetizioni

L'idea di ridurre le ripetizioni rientra nell'implementazione delle strategie di decoding. Consiste nel controllare la taglia degli n-gram che vogliamo evitare (non-repeated n-gram size). Questo approccio è agnostico rispetto alla probabilità di generazione: specificare di non ripetere un n-gram è una forzatura eccessiva o una deformazione, perché se nella generazione alcuni token hanno una maggiore probabilità di essere campionati, non c'è motivo di vincolare il decoder a usarli una sola volta.

Dobbiamo controllare il giusto bilanciamento tra coerenza e fluency, e creatività/diversità e novelty nella scelta delle parole.

Se vogliamo massimizzare la creatività, possiamo campionare in maniera random ogni token. Questa strategia è però usabile solo in task open-ended estremizzati.

### Top-k sampling

Il top-k sampling vincola il modello a campionare tra i *k* token con la maggiore probabilità di essere generati. Se un token ha una probabilità non trascurabile di essere selezionato, da un certo punto di vista sarebbe sensato poterlo includere nel campionamento.
La massa della distribuzione di probabilità si concentra su un sottoinsieme relativamente piccolo di token. La forma (shape) della distribuzione dei token cambia ad ogni step.
La soglia scelta (*k*) non è necessariamente adeguata ad ogni step di generazione.
![[13) Natural Language Generation-20241203102200812.png|604]]

### Top-p (nucleus) sampling

Il top-p sampling, o nucleus sampling, specifica una soglia di probabilità cumulativa da utilizzare durante il campionamento. Il decoder seleziona il prossimo token tra quelli che contribuiscono maggiormente alla massa di probabilità cumulativa. In pratica:
* Si campionano tutti i token che contribuiscono alla massa di probabilità cumulativa superiore a *p*. (ovvero, dove la massa è concentrata).
* Il valore di *k* (numero di token considerati) varia a seconda dell'uniformità della distribuzione di probabilità dei token.

![[13) Natural Language Generation-20241203102538497.png|597]]

È possibile utilizzare contemporaneamente il top-k e il top-p sampling. Si può impostare un valore di top-k relativamente alto (circa 50), anche con un vocabolario molto ampio (centinaia di migliaia di token). Applicando successivamente una selezione basata sulla concentrazione della massa di probabilità (top-p), si risolvono i problemi derivanti da distribuzioni di probabilità non uniformi. Al contrario, quando la distribuzione è uniforme, l'applicazione del top-p dopo il top-k non apporta miglioramenti significativi.

### Scaling della Randomness: Temperatura

La temperatura agisce come un fattore di scaling dei logit ed è sempre maggiore di 0. Interveniamo nella softmax scalando l'esponente per un parametro $\tau$.

Quando usiamo una temperatura maggiore di 1, il risultato della softmax tende a essere più piatto, favorendo la possibilità che il prossimo token da generare possa essere uno qualsiasi. Questo favorisce la creatività (diversità) nella selezione dei token.

Se vogliamo irrigidire il modello dal punto di vista della novità, usiamo una temperatura < 1 per rendere la distribuzione più skewed, favorendo così una minore creatività.

Una temperatura alta implica che il modello inizi ad allucinare (o a essere confabulatorio).

### Recall

Al passo temporale *t*, il modello calcola una distribuzione di probabilità *P<sub>t</sub>* applicando la funzione softmax a un vettore di punteggi *s<sub>w</sub>* ∈ ℝ<sup>|V|</sup>:

$$P_t(y_t = w) = \frac{exp(s_w)}{\sum_{w'∈V} exp(s_{w'})}$$

È possibile applicare un iperparametro di temperatura *τ* alla softmax per ribilanciare *P<sub>t</sub>*:

$$P_t(y_t = w) = \frac{exp(s_w / \tau)}{\sum_{w'∈V} exp(s_{w'} / \tau)}$$

* **Aumentare la temperatura τ > 1:** *P<sub>t</sub>* diventa più uniforme.
* Output più diversificato (la probabilità è distribuita su tutto il vocabolario).
* **Abbassare la temperatura τ < 1:** *P<sub>t</sub>* diventa più appuntita.
* Output meno diversificato (la probabilità è concentrata sulle parole principali).

La temperatura è un iperparametro per il decoding: può essere ottimizzata sia per la ricerca beam search che per il sampling.
