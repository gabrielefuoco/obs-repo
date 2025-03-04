
# GloVe: Un Nuovo Modello di Regressione Log-Bilineare Globale

## Miglioramenti Rispetto a Modelli Precedenti

* **Fattorizzazione matriciale globale (es. LSA):** Scarsa performance nelle analogie di parole; contribuzione sproporzionata delle parole frequenti.
* **Metodi di finestra di contesto locale (poco profondi):** Non sfruttano le statistiche globali di co-occorrenza.

## Risultati di GloVe

* Supera significativamente altri modelli.
* Performance decrescente per dimensioni > 200.
* Finestre asimmetriche (sinistra) migliori per compiti sintattici.
* Finestre simmetriche migliori per compiti semantici.
* Corpus più ampio migliore per compiti sintattici; Wikipedia migliore di Gigaword5 per compiti semantici.
* Performance di Word2Vec peggiora con molti campioni negativi (>10).
* Più veloce e accurato di altri modelli a parità di condizioni.


## Valutazione dei Vettori di Parole: Intrinseca vs. Estrinseca

* **Intrinseca:** Valutazione su sottocompiti specifici; calcolo veloce; utile per comprendere il sistema; utilità incerta senza correlazione con compiti reali.
* **Estrinseca:** Valutazione su compiti reali; calcolo lento; difficoltà nell'isolare problemi di sottosistema; miglioramento dell'accuratezza indica miglioramento del sottosistema.


## Analogie di Vettori di Parole

* **Formula:**  $d=\arg\max_{i} \frac{(x_{b}-x_{a}+x_{c})^T x_{i}}{\|x_{b}-x_{a}+x_{c}\|}$ dove $a:b::c:? \to d$
* **Valutazione:** Distanza coseno dopo l'addizione per catturare analogie semantiche e sintattiche.
* **Esempio:** `uomo:donna :: re:?` (king-man = queen-woman)
* **Esclusione parole di input:** Problema con informazioni non lineari.


## Similarità del Significato

* **Valutazione:** Confronto della prossimità nello spazio di rappresentazione con riferimenti umani (esempio di tabella con valori di similarità).
* **GloVe:** Performance migliore di CBOW anche a dimensionalità doppia.


## Polisemia delle Parole

* **Problema:** La maggior parte delle parole (soprattutto comuni e antiche) ha molti significati.
* **Esempio:** "Pike" (punto, ferrovia, strada).
* **Soluzione:** La rappresentazione nello spazio deve catturare tutti i significati.


---

# I. Rappresentazione delle Parole

* **Ambiguità semantica:** Un singolo vettore per parola non cattura tutti i significati.
* **Soluzione:** Clustering dei contesti di occorrenza di ogni parola.
* **Metodo:** K-means su rappresentazioni dense ottenute tramite compressione LSA (Latent Semantic Analysis).
* **Iterazione:** Aggiornamento dei pesi dei cluster ad ogni iterazione.
* **Risultato:** Molteplici vettori di embedding per parola, uno per contesto.


# II. Named Entity Recognition (NER)

* **Obiettivo:** Identificare e classificare nomi propri in un testo.
* **Pre-processing:** Gestione parole composte e acronimi (forma canonica).
* **Approccio:** Classificazione binaria o multiclasse.
* **Esempio:** Vettore di contesto `X_window = [x_museums, x_in, x_paris, x_are, x_amazing]^T` per la parola target "Paris".
* **Applicazioni:** Analisi del sentiment e altri compiti.


# III. Addestramento con Cross-Entropy Loss

* **Obiettivo:** Massimizzare la probabilità della classe corretta *y* (o minimizzare la probabilità logaritmica negativa).
* **Cross-entropia:** $H(p,q) = -\sum_{c=1}^{C} p(c)\log q(c)$
    * *p*: Distribuzione di probabilità vera (ground truth).
    * *q*: Probabilità calcolata dal modello.
* **Caso One-hot encoding:** La cross-entropia si riduce a −log *p*(y<sub>i</sub>∣x<sub>i</sub>).
* **Funzione di costo:** Binary cross-entropy per classificazione binaria.

# IV.  (Sezione incompleta nel testo originale)

---

# Classificazione Neurale

## I. Classificatore Softmax

* **Formula:**  $p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$
* **Limite:** Confine di decisione lineare.


## II. Classificatore di Rete Neurale

* **Apprendimento:** Sia i pesi *W* che le rappresentazioni distribuite delle parole vengono appresi.
* **Mapping:** Vettori one-hot mappati in uno spazio vettoriale intermedio tramite uno strato di embedding:  `x = Le`.
* **Reti profonde:**  Permettono un classificatore non lineare.
* **Softmax:** Generalizzazione del regressore logistico a più classi. Massimizza l'allineamento di x con la matrice dei pesi per la classe y, normalizzando per ottenere una probabilità.  *w* è parte di $\theta$.


## III. Classificazione di Rete Neurale (Dettagli)

* **Softmax:** Calcola la probabilità di appartenenza ad ogni classe.
    * **Formula:** $p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$
    * **Passaggi:**
        1. **Prodotto scalare:** $W_y \cdot x = \sum_i W_{yi} x_i = f_y$
        2. **Softmax:** $p(y|x) = \frac{\exp(f_y)}{\sum_c \exp(f_c)} = \text{softmax}(f_y)$
        3. **Selezione della classe:** Viene selezionata la classe con la probabilità massima.
    * **Ottimizzazione:** Si massimizza $p(y|x)$ o si minimizza $-\log p(y|x) = -\log \left( \frac{\exp(f_y)}{\sum_c \exp(f_c)} \right)$ per ogni esempio di training (x, y).


## IV. Classificatore Lineare Neurale

* **Regressore Logistico:** Utilizza una funzione di attivazione sigmoide.
* **Trasformazione Input:** Gli input *x* sono trasformati tramite parametri appresi (pesi e bias).
* **Funzione di Trasformazione:** *f*(x) (es. sigmoide, logistica, tangente iperbolica, ReLU).
* **Combinazione con Vettore di Pesi:** Il risultato della trasformazione (*h*) è combinato con un vettore di pesi, seguito da una funzione logistica per ottenere le probabilità finali.
* **Input:** Istanza di training (parola centrale + contesto).


## V. Reti Neurali

* **Modello Multistrato:** Utilizza funzioni di regressione logistica composte per catturare relazioni non lineari.
* **Strati Intermedi:** Aggiungono capacità di gestione delle non-linearità.
    * **Strati vicini all'input:** Relazioni sintattiche (contesto vicino).
    * **Strati lontani dall'input:** Relazioni semantiche (parole lontane nel contesto).
* **Funzionamento:**
    * **Formulazione:** $𝑧 = 𝑊𝑥 + 𝑏$; $𝑎 = 𝑓(𝑧)$
    * **Applicazione elemento per elemento:** $𝑓([𝑧₁, 𝑧₂, 𝑧₃]) = [𝑓(𝑧₁), 𝑓(𝑧₂), 𝑓(𝑧₃)]$
    * **Esempio:** $𝑎₁ = 𝑓(𝑊_{11}𝑥_{1} + 𝑊_{12}𝑥_{2} + 𝑊_{13}𝑥_{3} + 𝑏_{1})$


## VI. Tecniche di Miglioramento

* **Regolarizzazione:** (Lasso, Ridge, Elastic Net) Riduce la complessità del modello ed evita l'overfitting.
* **Dropout:** Regolarizzazione che riduce la co-adattamento delle features disattivando casualmente neuroni durante l'addestramento.
* **Vettorizzazione:** Utilizzo di trasformazioni matriciali per efficienza computazionale.


## VII. Inizializzazione dei Parametri

* **Pesi:** Inizializzati a piccoli valori casuali (evitando zeri) per evitare simmetrie.
* **Bias:**
    * **Strati nascosti:** Inizializzati a 0.
    * **Strato di output:** Inizializzati al valore ottimale se i pesi fossero 0 (es., target medio o inverso della sigmoide del target medio).
* **Distribuzione Uniforme:** ~ Uniforme(-r, r) (con *r* opportunamente scelto, o rimossa la necessità con la normalizzazione per layer).


## VIII. Inizializzazione di Xavier

* **Obiettivo:** Migliorare l'efficacia dell'addestramento delle reti neurali.
* **Metodo:** Definizione della varianza dei pesi in base a *fan-in* e *fan-out*.
    * **Fan-in (n<sub>in</sub>):** Dimensione dello strato precedente.
    * **Fan-out (n<sub>out</sub>):** Dimensione dello strato successivo.


---

## Formula di Varianza e Benefici

La formula per la varianza di $W_i$ è:

$$Var(W_{i}) = \frac{2}{n_{in} + n_{out}}$$

dove:

* $n_{in}$ rappresenta il numero di elementi in input.
* $n_{out}$ rappresenta il numero di elementi in output.

Questa formula presenta due principali benefici:

* **Media dei pesi intorno a 0:**  La formula contribuisce a mantenere la media dei pesi centrata intorno a zero.

* **Evitare range troppo piccoli nella distribuzione uniforme:**  L'utilizzo di questa formula aiuta a prevenire la creazione di range troppo ristretti nella distribuzione uniforme dei pesi, migliorando la robustezza e la generalizzazione del modello.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
