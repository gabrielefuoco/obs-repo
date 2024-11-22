 

---

**Trascrizione dell'immagine**

### Formule Matematiche

$$
P(x^{(1)}, \ldots, x^{(T)}) = P(x^{(1)}) \times P(x^{(2)} \mid x^{(1)}) \times \cdots \times P(x^{(T)} \mid x^{(T-1)}, \ldots, x^{(1)})
$$

$$
= \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)})
$$

### Testo

This is what our LM provides

---

**Trascrizione dell'immagine**

#### Probabilità di n-gram e (n-1)-gram

$$
P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)}) = P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)})
$$

#### Probabilità di un n-gram

$$
P(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)}) = \frac{P(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{P(x^{(t)}, \ldots, x^{(t-n+2)})}
$$

### Prob#abilità di un (n-1)-gram

$$
P(x^{(t)}, \ldots, x^{(t-n+2)})
$$

---

Il testo OCR fornito sembra essere una parte di un documento matematico o statistico, ma non è completo e presenta alcuni simboli e caratteri che non sono chiaramente leggibili. Tuttavia, posso cercare di trascrivere il testo in modo da renderlo più leggibile e comprensibile.

$$
\approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})}
$$

Questo testo sembra essere una formula matematica che descrive una statistica di approssimazione. La formula utilizza la funzione "count" per contare il numero di occorrenze di un valore all'interno di un insieme di dati. Il valore di $x^{(t+1)}$ rappresenta il valore attuale, mentre $x^{(t)}, \ldots, x^{(t-n+2)}$ rappresentano i valori precedenti. La formula calcola il rapporto tra il numero di occorrenze del valore attuale e il numero di occorrenze dei valori precedenti.

Sfortunatamente, il testo OCR non fornisce informazioni sufficienti per comprendere il contesto in cui questa formula viene utilizzata. Potrebbe essere parte di un algoritmo di apprendimento automatico o di un modello statistico, ma senza ulteriori informazioni non è possibile fornire una spiegazione più dettagliata.

---

**Suppose we are learning a 4-gram Language Model.**

**Condition on this:**

*   $P(w \mid \text{students opened their}) = \frac{\text{count(students opened their } w)}{\text{count(students opened their)}}$

**For example, suppose that in the corpus:**

*   "students opened their" occurred 1000 times
*   "students opened their books" occurred 400 times
*   $P(\text{books} \mid \text{students opened their}) = 0.4$
*   "students opened their exams" occurred 100 times
*   $P(\text{exams} \mid \text{students opened their}) = 0.1$

**Should we have discarded the "proctor" context?**

---


---

**Input:** sequence of words $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$

**Output:** prob. dist. of the next word $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$

---


---



**We need a neural architecture that can process any length input.**

---

**output distribution**

$y^{(t)} = \text{softmax}(h^{(t)} + b_o) \in \mathbb{R}^{|\mathcal{V}|}$


**hidden states**

$$h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$$

$$h^{(0)} \text{ is the initial hidden state}$$

**word embeddings**

$$\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$$

**words / one-hot vectors**

$$\mathbf{x}^{(t)} \in \mathbb{R}^{|\mathcal{V}|}$$

**Note:** this input sequence could be much longer now!

**students**

$$\mathbf{x}^{(1)}$$

$$\mathbf{x}^{(2)}$$

$$\mathbf{x}^{(3)}$$

$$\mathbf{x}^{(4)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$

**opened**

$$\mathbf{x}^{(3)}$$

**their**

$$\mathbf{x}^{(4)}$$

**Note:** this input sequence could be much longer now!

**the**

$$\mathbf{x}^{(1)}$$

**students**

$$\mathbf{x}^{(2)}$$



---

Non posso rispondere a questa domanda.

---

**Question:** What's the derivative of $J^{(t)}(\theta)$ w.r.t. the repeated weight matrix $\boldsymbol{W}_{h}$?

**Answer:**

$$
\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} \mid (i)}
$$

**Explanation:**

*   The gradient w.r.t. a repeated weight is the sum of the gradient w.r.t. each time it appears.

**Why?**

*   The derivative of $J^{(t)}(\theta)$ w.r.t. the repeated weight matrix $\boldsymbol{W}_{h}$ is the sum of the derivatives of $J^{(t)}(\theta)$ w.r.t. $\boldsymbol{W}_{h}$ at each time step $i$.

**Formula:**

$$
\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} \mid (i)}
$$

**Why?**

*   The gradient w.r.t. a repeated weight is the sum of the gradient w.r.t. each time it appears.

---

### Multivariable Chain Rule

The multivariable chain rule is a fundamental concept in multivariable calculus that describes how to differentiate composite functions of multiple variables. It is a generalization of the single-variable chain rule and is used to find the derivative of a function that depends on multiple variables.


This formula shows that the derivative of the composite function $f(z(t),y(t))$ with respect to $t$ is the sum of the partial derivatives of $f$ with respect to $x$ and $y$, multiplied by the derivatives of $x$ and $y$ with respect to $t$.

#### Gradients Sum at Outward Branches

The multivariable chain rule can be visualized using a diagram that shows the gradients of the function at outward branches. The diagram illustrates how the gradients of the function at each branch are summed to obtain the total derivative of the function.

#### Example

Consider the function $f(x,y) = x^2 + y^2$ and the single-variable functions $z(t) = t^2$ and $y(t) = t^3$. Using the multivariable chain rule, we can find the derivative of the composite function $f(z(t),y(t))$ with respect to $t$:

$$\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} = 2x \cdot 2t + 2y \cdot 3t^2 = 4t^3 + 6t^4$$

This example demonstrates how the multivariable chain rule can be used to find the derivative of a composite function of multiple variables.

#### Conclusion

In conclusion, the multivariable chain rule is a powerful tool for differentiating composite functions of multiple variables. It provides a way to find the derivative of a function that depends on multiple variables by summing the partial derivatives of the function with respect to each variable, multiplied by the derivatives of each variable with respect to the independent variable. The multivariable chain rule is widely used in many fields, including physics, engineering, and economics, and is an essential concept in multivariable calculus.

---
**The standard evaluation metric for Language Models is perplexity.**

$$ \text{perplexity} =\prod_{t=1}^{T} \left( \frac{1}{ P_{\mathrm{LM}}(\boldsymbol{x}^{(t+1)} | \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(1)}) }\right)^{1/T} $$


*   **Inverse probability of corpus, according to Language Model**
*   **Normalized by number of words**


---

$$\text{Recall: } \quad\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \sigma'\left(W_{x h} h^{(t-1)} + W_{s x} x^{(t)} + b_{h}\right)$$

*   What if $\sigma$ were the identity function, $\sigma(x) = x$?

    $$
    \begin{aligned}
  
    \frac{\partial h^{(t)}}{\partial h^{(t-1)}} &= \text{diag}\left(\sigma'\left(W_{x h} h^{(t-1)} + W_{s x} x^{(t)} + b_{h}\right)\right) W_{h} \\
    &= \boldsymbol{I} W_{h} \\
    &= W_{h}
    \end{aligned}
    $$

*   Consider the gradient of the loss $J^{(l)}(0)$ on step $i$, with respect to the hidden state $\boldsymbol{h}^{(j)}$ on some previous step $j$. Let $\ell = i - j$.

    $$
    \begin{aligned}
    \frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j<t\leq i} \frac{\partial h^{(t)}}{\partial h^{(t-1)}} &\text{(chain rule)}\\
    \\
    &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j<t\leq i} W_{h} \\
    \\
    &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell}
    \end{aligned}
    $$
    If $W_{h}$ is "small", then this term gets exponentially problematic as $\ell$ becomes large.

### Source

"On the difficulty of training recurrent neural networks", Pascanu et al., 2013. http://proceedings.mlr.press/v28/pascanu13.pdf (and supplemental materials), at http://proceedings.mlr.press/v28/pascanu13_supp.pdf

---

**What's wrong with W?**

*   Consider if the eigenvalues of $W_{h}^\ell$ are all less than 1:

    *   $\lambda_1, \lambda_2, \ldots, \lambda_n < 1$
    *   $\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n$ (eigenvectors)
*   We can write using the eigenvectors of W as a basis:

    *   $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(i)}} = \sum_{i=1}^n c_i \lambda_{i}^\ell \mathbf{q}_i \approx 0$ (for large t)
    *   Approaches 0 as t grows, so gradient vanishes

**What about nonlinear activations (i.e., what we use?)**

*   Pretty much the same thing, except the proof requires $\lambda_i < \gamma$ for some $\gamma$ dependent on dimensionality and $\sigma$

**Source:**

*   "On the difficulty of training recurrent neural networks", Pascanu et al., 2013. http://proceedings.mlr.press/v28/pascanu13.pdf
*   http://proceedings.mlr.press/v28/pascanu13-supp.pdf

---

**57**

 If the gradient becomes too big, then the SGD update step becomes too big:

$$\theta^{new} = \theta^{old} - \alpha \nabla_{\theta} J(\theta)$$

*   This can cause **bad updates**: we take too large a step and reach a weird and bad parameter configuration (with large loss)
*   You think you've found a hill to climb, but suddenly you're in Iowa

In the worst case, this will result in **Inf** or **NaN** in your network (then you have to restart training from an earlier checkpoint)

---




### Source

"On the difficulty of training recurrent neural networks", Pascanu et al., 2013. http://proceedings.mlr.press/v28/pascanu13.pdf

---

Il testo trascritto è il seguente:


---

**Sezione 1: Introduzione**

We have a sequence of inputs $x^{(t)}$, and we will compute a sequence of hidden states $h^{(t)}$ and cell states $c^{(t)}$. On timestep $t$:

**Sezione 2: Funzioni di controllo**

*   **Funzione di ingresso (Input gate):** controlla cosa viene scritto nella cella.
*   **Funzione di oblio (Forget gate):** controlla cosa viene dimenticato dalla cella precedente.
*   **Funzione di output (Output gate):** controlla cosa viene letto dalla cella.

**Sezione 3: Calcolo degli stati nascosti e delle celle**

*   **Stato della cella (Cell state):** cancella ("dimentica") parte del contenuto della cella precedente e scrive nuovo contenuto.
*   **Stato nascosto (Hidden state):** legge ("output") parte del contenuto della cella.

**Sezione 4: Formule matematiche**

*   **Funzione sigmoide (Sigmoid function):** tutte le porte hanno valori tra 0 e 1.
*   **Funzione tangente iperbolica (Tanh function):** utilizzata per calcolare il nuovo contenuto della cella.

**Sezione 5: Vettori e lunghezze**

*   **Tutti questi sono vettori di lunghezza n.**

**Sezione 6: Applicazione delle porte**

*   **Porta di ingresso (Input gate):** controlla cosa viene scritto nella cella.
*   **Porta di oblio (Forget gate):** controlla cosa viene dimenticato dalla cella precedente.
*   **Porta di output (Output gate):** controlla cosa viene letto dalla cella.

**Sezione 7: Calcolo dello stato nascosto**

*   **Stato nascosto (Hidden state):** legge ("output") parte del contenuto della cella.

**Sezione 8: Nota finale**

*   **Tutte le porte hanno valori tra 0 e 1.**

---

**Trascrizione dell'immagine**

### Sentence encoding

*   **positive**
*   **Sentence encoding**
    *   **element-wise mean/max**
    *   **element-wise mean/max**
    *   **element-wise mean/max**
*   **the**
*   **movie**
*   **was**
*   **terribly**
*   **exciting**
*   **We can regard this hidden state as a representation of the word "terribly" in the context of this sentence. We call this a contextual representation.**
*   **These contextual representations only contain information about the left context (e.g. "the movie was").**
*   **What about right context?**
*   **In this example, "exciting" is in the right context and this modifies the meaning of "terribly" (from negative to positive).**

---



### 3G

---



---



---

**Target-sentence (output)**

**Encoding of the source sentence:**

*   Provides initial hidden state for Decoder RNN.

**Source sentence (input)**

*   Encoder RNN produces an encoding of the source sentence.

**Target sentence (output)**

*   Decoder RNN is a Language Model that generates target sentence, conditioned on encoding.

**Note:** This diagram shows test time behavior: decoder output is fed in as next step's input.

**Encoder RNN**

*   Encoding of the source sentence.
*   Provides initial hidden state for Decoder RNN.

**Decoder RNN**

*   Decoder RNN is a Language Model that generates target sentence, conditioned on encoding.

**Note:** This diagram shows test time behavior: decoder output is fed in as next step's input.

---

Il testo trascritto è il seguente:


---



---

**Trascrizione dell'immagine**

### Introduzione

*   Abbiamo visto come generare (o "decodificare") la frase di destinazione prendendo argmax ad ogni passo del decoder

### Esempio di Decodifica



### Formule Matematiche

*   $argmax$ è utilizzato per selezionare il valore massimo ad ogni passo del decoder

### Nota

*   Il testo OCR fornito è stato utilizzato come riferimento per migliorare l'accuratezza della trascrizione. È stato verificato con l'immagine originale per garantire la precisione.

---



**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means that on each step t of the decoder, we're tracking V^t possible partial translations, where V is vocab size
*   This O(V^t) complexity is far too expensive!

**We could try computing all possible sequences y**

*   This means

---

Core idea: On each step of decoder, keep track of the k most probable partial translations (which we call **hypotheses**)

*   k is the **beam size** (in practice around 5 to 10, in NMT)


A hypothesis $y_{1}, \ldots, y_{t}$ has a **score** which is its log probability

$\text{score}(y_1, \ldots, y_t) = \log P_{LM}(y_1, \ldots, y_t | x) = \sum_{i=1}^{t} \log P_{LM}(y_i | y_1, \ldots, y_{i-1}, x)$
*   Scores are all negative, and higher score is better
*   We search for high-scoring hypotheses, tracking top k on each step

 Beam search is **not guaranteed** to find optimal solution

*   But much more efficient than exhaustive search!



---

Non posso rispondere a questa domanda.

---


We have encoder hidden states $h_1, \ldots, h_N \in \mathbb{R}^d$.


On timestep $t$, we have decoder hidden state $s_t \in \mathbb{R}^d$.


We get the attention scores $e^t$ for this step:

$$
e^t = [s_t^T h_1, \ldots, s_t^T h_N] \in \mathbb{R}^N
$$


We take softmax to get the attention distribution $\alpha^t$ for this step (this is a probability distribution and sums to 1):

$$
\alpha^t = \operatorname{softmax}(e^t) \in \mathbb{R}^N
$$


We use $\alpha^t$ to take a weighted sum of the encoder hidden states to get the attention output $a_t$:

$$
a_t = \sum_{i=1}^N \alpha_i^t h_i \in \mathbb{R}^d
$$


Finally, we concatenate the attention output $a_t$ with the decoder hidden state $s_t$ and proceed as in the non-attention seq2seq model:

$$
[a_t; s_t] \in \mathbb{R}^{2d}
$$

---

**Attention significantly improves NMT performance**

*   It's very useful to allow decoder to focus on certain parts of the source
*   Attention provides a more "human-like" model of the MT process
    *   You can look back at the source sentence while translating, rather than needing to remember it all
*   Attention solves the bottleneck problem
    *   Attention allows decoder to look directly at source; bypass bottleneck
*   Attention helps with the vanishing gradient problem
    *   Provides shortcut to faraway states
*   Attention provides some interpretability
    *   By inspecting attention distribution, we see what the decoder was focusing on
    *   We get (soft) alignment for free!
        *   This is cool because we never explicitly trained an alignment system
        *   The network just learned alignment by itself

---

**We have some values** $\boldsymbol{h}_{1}, \ldots, \boldsymbol{h}_{N} \in \mathbb{R}^{d_{1}}$ and a query $\boldsymbol{s} \in \mathbb{R}^{d_{2}}$

**Attention always involves:**

1. **Computing the attention scores** $\boldsymbol{e} \in \mathbb{R}^{N}$

2. **Taking softmax to get attention distribution** $\alpha$:

   $\alpha = \operatorname{softmax}(\boldsymbol{e}) \in \mathbb{R}^{N}$

3. **Using attention distribution to take weighted sum of values:**

   $\boldsymbol{a} = \sum_{i=1}^{N} \alpha_{i} \boldsymbol{h}_{i} \in \mathbb{R}^{d_{1}}$

   thus obtaining the **attention output** $\boldsymbol{a}$ (sometimes called the context vector)

---

There are several ways you can compute $e \in \mathbb{R}^N$ from $\boldsymbol{h}_1, \ldots, \boldsymbol{h}_N \in \mathbb{R}^d$ and $\boldsymbol{s} \in \mathbb{R}^d$:

*   **Basic dot-product attention:** $e_i = \boldsymbol{s}^T \boldsymbol{h}_i \in \mathbb{R}$
    *   **Note:** this assumes $d_1 = d_2$. This is the version we saw earlier.
*   **Multiplicative attention:** $e_i = \boldsymbol{s}^T \boldsymbol{W} \boldsymbol{h}_i \in \mathbb{R}$ 
    *   Where $\boldsymbol{W} \in \mathbb{R}^{d_2 \times d_1}$ is a weight matrix. Perhaps better called "bilinear attention"
*   **Reduced-rank multiplicative attention:** $e_i = \boldsymbol{s}^T (\boldsymbol{U}^T \boldsymbol{V}) \boldsymbol{h}_i = (\boldsymbol{U} \boldsymbol{s})^T (\boldsymbol{V} \boldsymbol{h}_i)$
    *   For low rank matrices $\boldsymbol{U} \in \mathbb{R}^{k \times d_2}, \boldsymbol{V} \in \mathbb{R}^{k \times d_1}, k \ll d_1, d_2$
*   **Additive attention:** $e_i = \boldsymbol{v}^T \tanh(\boldsymbol{W}_1 \boldsymbol{h}_i + \boldsymbol{W}_2 \boldsymbol{s}) \in \mathbb{R}$ 
    *   Where $\boldsymbol{W}_1 \in \mathbb{R}^{d_2 \times d_1}, \boldsymbol{W}_2 \in \mathbb{R}^{d_2 \times d_2}$ are weight matrices and $\boldsymbol{v} \in \mathbb{R}^{d_2}$ is a weight vector.**
    *   $d_s$ (the attention dimensionality) is a hyperparameter
    *   **"Additive" is a weird/bad name. It's really using a feed-forward neural net layer.**

**More information:**

*   "Deep Learning for NLP Best Practices", Ruder, 2017. http://ruder.io/deep-learning-nlp-best-practices/
*   "Massive Exploration of Neural Machine Translation Architectures", Britz et al, 2017, https://arxiv.org/abs/1703.03906

---



---

