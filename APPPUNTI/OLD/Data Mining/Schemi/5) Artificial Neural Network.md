**Reti Neurali Artificiali**

I. **Concetto Fondamentale:** Modello ispirato al sistema nervoso umano, composto da nodi (neuroni) interconnessi tramite collegamenti (sinapsi) con pesi associati che rappresentano la forza della connessione.  
I nodi sommano gli input, e se superano una soglia, si attivano producendo un output (eccitatorio o inibitorio).

   - A. **Componenti:**
      - Nodi (neuroni)
      - Collegamenti (assoni/dendriti)
      - Pesi (forza sinapsi)
      - Soglie di attivazione

   - B. **Progettazione:**
      - Scelta del numero e tipo di unità
      - Determinazione della struttura
      - Codifica degli esempi di addestramento (input/output)
      - Inizializzazione e addestramento (determinazione dei pesi)

   - C. **Caratteristiche:**
      - Rappresentazione dati tramite numerose feature, anche a valori reali.
      - Funzione obiettivo a valori reali.
      - Possibilità di dati rumorosi.
      - Tempi di addestramento lunghi.
      - Valutazione veloce della rete addestrata.
      - Non è necessaria la comprensione della semantica della funzione attesa.

   D. **Tipi di Modelli:**
      1. **Biologico:** Imitazione di sistemi neurali biologici (es. vista, udito).  Connettività maggiore ma tempi di commutazione più lenti (ms vs ns) rispetto ai neuroni biologici.
      2. **Guidato dalle applicazioni:** Architettura condizionata dalle esigenze applicative.


II. **Perceptron (Rosenblatt, 1962):**  Rete neurale semplice a singolo strato.

   A. **Architettura:**
      - Nodi di ingresso (uno per attributo $x_i$)
      - Un singolo nodo di uscita
      - Collegamenti pesati $w_i$

   B. **Calcolo dell'output:**
      $$y = \begin{cases} +1 & \text{se } w^T x + b > 0 \\ −1 & \text{altrimenti} \end{cases}$$  o in forma vettoriale:  $$y = sign(w^T x + b)$$

   C. **Funzione di attivazione:**  `sign()` (non lineare, fondamentale per evitare una semplice funzione lineare).

   D. **Addestramento:**  I pesi $w_i$ e il bias $b$ vengono addestrati per risolvere problemi di classificazione binaria.


**Addestramento del Perceptron**

* **Obiettivo:** Apprendere i parametri ottimali  $w$ (pesi) e $b$ (bias) per la classificazione lineare.

    * **Inizializzazione:**
        * Caricamento del training set {(x, y)}.
        * Impostazione del contatore delle iterazioni k = 0.
        * Inizializzazione dei pesi $w(0)$ con valori casuali.
    * **Iterazioni (fino a errore medio < γ):**
        * Calcolo dell'errore medio: $\frac{\sum_{i=1}^n|y_{i}-f(\tilde{w}^{(k)},x_{i})|}{n}$
        * Per ogni esempio (x, y):
            * Calcolo dell'output stimato $f(\tilde{w}^{(k)},x_{i})$.
            * Aggiornamento dei pesi per ogni $w_j$:  $w_j^{(k+1)} = w_j^{(k)} + \lambda (y_{i} - f(\tilde{w}^{(k)},x_{i})) x_j$
        * Incremento del contatore k = k + 1.
    * **Parametri:**
        * $\lambda$: learning rate (0 < λ < 1).
        * $x_j$: valore del j-esimo attributo.
        * $w(k)_j$: peso della connessione j-esima dopo la k-esima iterazione.

* **Aggiornamento dei Pesi:** Regola di discesa del gradiente stocastica.

    * Errore = 0: nessun aggiornamento.
    * Errore > 0: incremento del peso.
    * Errore < 0: decremento del peso.
    * L'entità dell'aggiornamento è proporzionale a λ, all'errore e a $x_j$.

* **Iperpiano di Separazione:** $w \cdot x + b = 0$

* **Convergenza:**

    * Il perceptron converge solo se le classi sono linearmente separabili.
    * **Limitazione:** Non può risolvere problemi con separazione non lineare.
    * **Teorema di Convergenza (Rosemblatt):** Garantisce la convergenza per classi linearmente separabili (nessun minimo locale).

* **Funzioni Booleane:**

    * Rappresentazione di funzioni booleane primitive linearmente separabili (AND, OR).
    * Non può rappresentare funzioni non linearmente separabili (XOR).


**Reti Neurali Multilivello**

I. **Architettura e Funzionamento:**

*   **Complessità:** Più complessa del percettrone, capace di risolvere problemi non lineari.
*   **Strati:** Organizzazione in strati (layer) che elaborano sequenzialmente le features di input, rappresentando diversi livelli di astrazione.

II. **Strati Principali:**

*   **Input Layer:**  Un nodo per ogni attributo di input (numerico, binario o categoriale).
*   **Hidden Layers:** Strati intermedi che elaborano i segnali dai livelli precedenti, producendo valori di attivazione per il livello successivo.
*   **Output Layer:** Strato finale che produce le previsioni: un nodo per la classificazione binaria, o $\frac{k}{\log_{2}(k)}$ nodi per problemi multiclasse.

III. **Tipi di Reti:**

*   **Feedforward:** Propagazione dei segnali solo in avanti (dall'input all'output).  Migliora la capacità di rappresentare confini decisionali complessi rispetto ai percettroni grazie agli strati nascosti.

IV. **Esempio: Problema XOR:**

*   Insolubile da un singolo percettrone.
*   Risolvibile da una rete con uno strato nascosto di due nodi: ogni nodo crea un iperpiano, l'output li combina.

V. **Apprendimento di Caratteristiche:**

*   **Gerarchia di astrazione:** Gli strati nascosti catturano caratteristiche a diversi livelli di astrazione (semplici nel primo strato, complesse negli strati successivi).
*   **Formula di attivazione:**  Il valore di attivazione del nodo i-esimo dell'l-esimo livello è:
	*   $a_i^l = f(z_i^l) = f(\sum_j w_{ij}^l a_j^{l-1} + b_i^l)$


**Funzioni di Attivazione nelle Reti Neurali**

* **Funzioni di Attivazione Alternative alla Funzione Segno:**
    * **Funzione Gradino:**  `gradino(x) = { 1 se x > t; 0 altrimenti }`
    * **Funzione Sigmoidea:** `sigmoid(x) = σ(x) = 1 / (1 + e⁻ˣ)`
        * Derivabile.
        * Non usata negli hidden layer.
        * Utile per classificazione binaria.
    * **Funzione Tangente Iperbolica:** `tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`
        * Aggiornamenti dei pesi più elevati rispetto alla sigmoidea.
        * Dati centrati verso 0, facilitando l'apprendimento.
        * Usata nel livello di output.
    * **Funzione ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`
        * Derivata: ` = { 1 se x > 0; 0 se x < 0 }`
        * Non derivabile in 0.
    * **Funzione Softmax:** $f(x) = \frac{e^{(zᵢ)}}{Σ e^{(zk)}}$
        * Normalizza gli output (somma degli output = 1).
        * Usata nel livello di output per classificazione multiclasse.


* **Problemi nell'Addestramento:**
    * **Vanishing Gradient Problem:** Diminuzione della grandezza degli aggiornamenti dei pesi, soprattutto con funzioni sigmoidee, causando convergenza lenta o blocco della rete.


* **Backpropagation e Addestramento:**
    * **Funzione di Perdita:** Misura la differenza tra output predetto (ŷ) e atteso (y).  
      `E(w,b) = Σᵢ₌₁ⁿ Loss(yᵢ,ŷᵢ)`
    * **Errore Quadratico Medio (MSE):** `Loss(yᵢ,ŷᵢ) = (yᵢ - ŷᵢ)²`
    * **Minimizzazione di E(w,b):** Obiettivo dell'addestramento.
    * **Non Linearità e Minimi Locali:** La non linearità delle funzioni di attivazione rende E(w,b) non convessa, con possibili minimi locali.



**Metodo di Discesa del Gradiente e Backpropagation**

I. **Discesa del Gradiente:**

   *   Scopo: Trovare un minimo locale della funzione di perdita E.
   *   Aggiornamento dei pesi e bias:
      *   $w_{ij}^l \leftarrow w_{ij}^l - \lambda \frac{\delta E}{\delta w_{ij}^l}$
      *   $b_{i}^l \leftarrow b_{i}^l - \lambda \frac{\delta E}{\delta b_{i}^l}$
   *   Calcolo del gradiente: Richiede la tecnica di backpropagation per i pesi dei nodi nascosti.

II. **Backpropagation:**

   *   **Derivata parziale della funzione di perdita:**
      *   Decomposizione della derivata: $\frac{\delta E}{\delta w_{j}^l} = \sum_{k=1}^n \frac{\delta Loss(y_{k},\hat{y_{k}})}{\delta w_{j}^l}$ (somma delle derivate delle singole perdite)
      *   Regola della catena per la derivata parziale rispetto a $w_{ij}^l$:
         *   $\frac{\partial Loss}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times \frac{\partial a_i^l}{\partial z_i^l} \times \frac{\partial z_i^l}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times \frac{\partial a_i^l}{\partial z_i^l} x_j^{l-1}$
         *   Con funzione di attivazione sigmoidale: $\frac{\partial Loss}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times a_i^l (1-a_i^l) \times x_j^{l-1}$
      *   Derivata parziale rispetto a $b_i^l$: $\frac{\partial Loss}{\partial b_i^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times a_i^l (1-a_i^l)$

   *   **Calcolo di δ (delta):**
      *   Livello di output (L) con errore quadratico: $\delta^L = \frac{\partial Loss}{\partial a_i^L} = 2(a_i^L - y_i)$
      *   Livelli nascosti:
         *   $\frac{\partial Loss}{\partial a_j^l} = \sum_i (\frac{\partial Loss}{\partial a_i^{l+1}} \times \frac{\partial a_i^{l+1}}{\partial z_i^{l+1}} \times \frac{\partial z_i^{l+1}}{\partial a_j^l} )$
         *   $\delta_j^l = \sum_i \left(\delta_i^{l+1} \times a_i^{l+1}(1-a_i^{l+1}) \times w_{ij}^{l+1}\right)$  (calcolo ricorsivo a ritroso)


III. **Riepilogo:** Il metodo combina la discesa del gradiente per minimizzare la funzione di perdita con la backpropagation per calcolare efficientemente il gradiente, propagando l'errore dagli output agli strati nascosti della rete neurale.  La backpropagation sfrutta la regola della catena per calcolare le derivate parziali necessarie all'aggiornamento dei pesi e dei bias.


**Schema Riassuntivo: Reti Neurali Artificiali (ANN)**

I. **Algoritmo di Addestramento**

   A. Inizializzazione:
      1. Definizione del dataset di training: $D_{train} = \{(x_k, y_k) | k = 1, 2, ..., n\}$
      2. Inizializzazione dei pesi e bias: $(w^{(0)}, b^{(0)})$ con valori randomici.
      3. Inizializzazione del contatore di iterazioni: $c = 0$.

   B. Iterazione finché convergenza: $(w^{(c+1)}, b^{(c+1)}) \approx (w^{(c)}, b^{(c)})$
      1. Per ogni record $(x_k, y_k) \in D_{train}$:
         a. Calcolo dei valori di attivazione: $(a_i^l)_k$
         b. Calcolo dei valori di errore: $(\delta_i^l)_k$ (backpropagation)
         c. Calcolo della perdita e della sua derivata: $(Loss)_k, (Loss')_k$
      2. Calcolo del gradiente rispetto ai pesi: $\frac{\partial E}{\partial w_{ij}^l} = \sum_k (\frac{\partial Loss}{\partial w_{ij}^l})_k$
      3. Calcolo del gradiente rispetto ai bias: $\frac{\partial E}{\partial b_i^l} = \sum_k (\frac{\partial Loss}{\partial b_i^l})_k$
      4. Aggiornamento dei pesi e bias tramite discesa del gradiente: $(w^{(c+1)}, b^{(c+1)})$
      5. Incremento del contatore: $c = c + 1$


II. **Caratteristiche delle ANN**

   A. Approssimatori Universali (multi-strato)
   B. Problematiche:
      1. Overfitting
      2. Minimi locali
      3. Tempo di costruzione del modello (lento) / test (veloce)
      4. Gestione attributi ridondanti (vantaggio)


III. **Limitazioni delle ANN**

   A. Sensibilità al rumore
   B. Gestione attributi mancanti (difficoltà)



