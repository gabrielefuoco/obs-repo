
## Natural Language Generation (NLG)

La Generazione del Linguaggio Naturale (NLG) è un sottocampo dell'Elaborazione del Linguaggio Naturale (NLP), definibile come:  `NLP = Comprensione del Linguaggio Naturale (NLU) + NLG`.  L'obiettivo dell'NLG è generare output linguistici fluidi, coerenti e utili.  Esistono due principali categorie di task NLG:  *open-ended generation*, dove l'output ha ampia libertà, e *non-open-ended generation*, dove l'input determina fortemente l'output.  La distinzione può essere formalizzata tramite l'entropia.  Queste categorie richiedono approcci di decodifica e/o addestramento differenti. ![[13) Natural Language Generation-20241203100245027.png|600]]


## Meccanismi di Generazione dell'Output

Diversi metodi governano la generazione dell'output:

* **Greedy Decoding:** Seleziona il token con la probabilità più alta dato il contesto corrente:  `$\hat{y}_{t}=\arg\max_{w\in V}P(y_{t}=w|y_{<t})$`.  È preferibile per task encoder-decoder classici come traduzione e summarizzazione.

* **Beam Search Decoding:** Esplora contemporaneamente più traiettorie di generazione (ipotesi), scegliendo infine quella con la probabilità cumulativa più alta.  È ottimale per contesti aperti e astratti.

* **Ridurre le Ripetizioni:**  Questa tecnica controlla la dimensione degli n-gram da evitare.  Tuttavia,  è importante bilanciare la coerenza e la fluidità con la creatività e la novità nella scelta delle parole.  Un campionamento completamente casuale massimizza la creatività, ma è applicabile solo a task *open-ended* estremi.

* **Top-k Sampling:**  Campiona tra i *k* token con la maggiore probabilità.  Questo approccio permette di includere token con probabilità non trascurabili, concentrando l'attenzione su un sottoinsieme rilevante della distribuzione di probabilità.

---

Il testo descrive tre tecniche per migliorare la generazione di testo da modelli di linguaggio: Top-k sampling, Top-p (nucleus) sampling e la regolazione della temperatura.

**Top-k sampling:** Seleziona i *k* token con la probabilità più alta per il prossimo token da generare.  La scelta di *k* è critica e non è ottimale per ogni step, dato che la distribuzione di probabilità dei token cambia ad ogni iterazione.  ![[]]

**Top-p (nucleus) sampling:**  Seleziona i token che cumulativamente contribuiscono ad una massa di probabilità superiore a una soglia *p*.  Questo metodo adatta dinamicamente il numero di token considerati, a differenza del top-k, concentrandosi sulla massa di probabilità e non su un numero fisso di token. ![[ ]]  È possibile combinare top-k e top-p, usando un top-k alto seguito da un top-p per gestire distribuzioni di probabilità non uniformi.

**Temperatura:**  Questo iperparametro scala i logit prima dell'applicazione della softmax, modificando la forma della distribuzione di probabilità.  Una temperatura $\tau > 1$ rende la distribuzione più piatta, aumentando la diversità (e il rischio di "allucinazioni") nell'output.  Una temperatura $\tau < 1$ rende la distribuzione più appuntita, riducendo la diversità e favorendo i token più probabili. La formula della softmax con temperatura è:

$P_t(y_t = w) = \frac{exp(s_w / \tau)}{\sum_{w'∈V} exp(s_{w'} / \tau)}$

dove  $P_t(y_t = w)$ è la probabilità del token *w* al passo temporale *t*,  $s_w$ è il punteggio del token *w*, e *V* è il vocabolario.  Aumentare τ rende *P<sub>t</sub>* più uniforme, mentre diminuire τ la rende più appuntita.

---

La temperatura è un iperparametro cruciale nel processo di *decoding* dei modelli linguistici.  Influenza sia la *beam search* che il *sampling*, metodi utilizzati per generare testo a partire da una rappresentazione latente.  Ottimizzando la temperatura, si può controllare la creatività e la diversità del testo generato.  Una temperatura bassa favorisce testi più coerenti e prevedibili, mentre una temperatura alta genera testi più creativi e sorprendenti, ma potenzialmente meno coerenti.

---
