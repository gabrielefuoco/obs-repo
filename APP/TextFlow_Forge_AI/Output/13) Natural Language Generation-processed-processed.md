
# Natural Language Generation (NLG)

## Definizione e Relazione con NLP

NLG è un ramo dell'elaborazione del linguaggio naturale (NLP).  NLP può essere definito come la combinazione di comprensione del linguaggio naturale (NLU) e generazione del linguaggio naturale (NLG). L'obiettivo di NLG è produrre output linguistici fluidi, coerenti e utili.

## Tipi di Generazione

* **Generazione Open-ended:**  Alta libertà nella distribuzione dell'output.
* **Generazione Non-open-ended:** L'input determina principalmente l'output.  Formalizzabile tramite entropia. Richiede diverse strategie di decodifica/addestramento.


## Meccanismi di Produzione dell'Output

* **Greedy Decoding:** Seleziona il token con la probabilità più alta dato il contesto:  $\hat{y}_{t}=\arg\max_{w\in V}P(y_{t}=w|y_{<t})$.  Preferibile per task encoder-decoder classici (traduzione automatica, summarizzazione).

* **Beam Search Decoding:** Genera contemporaneamente più traiettorie (ipotesi). Il numero di traiettorie è controllabile. L'output finale è la traiettoria con la probabilità cumulativa più alta. Ottimale per contesti aperti e astratti.


## Ridurre le Ripetizioni

* Controllo della taglia degli n-gram da evitare (non-repeated n-gram size).
* Approccio agnostico rispetto alla probabilità di generazione.
* Necessario bilanciare coerenza/fluency con creatività/diversità/novelty.
* Campionamento random per massimizzare la creatività (solo in task open-ended estremizzati).

* **Top-k sampling:** Campionamento tra i *k* token con maggiore probabilità. Considera la massa della distribuzione di probabilità concentrata su un piccolo sottoinsieme di token.


## Schema Riassuntivo: Tecniche di Campionamento e Controllo della Randomness nella Generazione di Linguaggio Naturale

**I. Tecniche di Campionamento**

**A. Top-k Sampling:** Seleziona i *k* token con la probabilità più alta.  *k* è fisso e potrebbe non essere ottimale per tutte le distribuzioni.

**B. Top-p (Nucleus) Sampling:** Seleziona i token che contribuiscono cumulativamente a una probabilità superiore a *p*. *k* varia dinamicamente a seconda della forma della distribuzione. Può essere combinato con il top-k per gestire distribuzioni non uniformi.


**II. Controllo della Randomness: Temperatura (τ)**

**A. Funzione:** Scala i logit nella funzione softmax: $P_t(y_t = w) = \frac{exp(s_w / \tau)}{\sum_{w'∈V} exp(s_{w'} / \tau)}$

**B. Effetto della Temperatura:**

* **τ > 1:** Distribuzione più uniforme, output più diversificato (creativo), rischio di allucinazioni.
* **τ < 1:** Distribuzione più appuntita, output meno diversificato (meno creativo).


**III. Recall e Softmax**

**A. Calcolo della Probabilità:** Al passo *t*, il modello calcola la distribuzione di probabilità *P<sub>t</sub>* usando la softmax: $P_t(y_t = w) = \frac{exp(s_w)}{\sum_{w'∈V} exp(s_{w'})}$

**B. Influenza della Temperatura sulla Softmax:** La temperatura *τ* ribilancia *P<sub>t</sub>*, modificando la forma della distribuzione di probabilità come descritto nel punto II.B.


## Ottimizzazione della Temperatura nel Decoding

La temperatura è un iperparametro cruciale nel processo di decoding. Influenza sia la ricerca beam search, modificando la probabilità di selezione dei token, sia il processo di sampling, modificando la distribuzione di probabilità dei token.  Necessita di ottimizzazione per ottenere risultati ottimali sia con beam search che con sampling.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
