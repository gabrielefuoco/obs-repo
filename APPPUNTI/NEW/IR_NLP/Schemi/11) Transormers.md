
##### Reti Ricorrenti Stacked (RNN)

* **Problema principale:** Codifica lenta e non parallelizzabile di sequenze lunghe.
* Complessità computazionale: O(Lunghezza della sequenza) operazioni non parallelizzabili.
* Limiti dell'addestramento su dataset grandi a causa della dipendenza sequenziale degli stati nascosti.

##### Meccanismo di Attenzione

* **Funzionamento:** Ogni parola (query) accede selettivamente alle informazioni da tutte le altre parole (valori) nella sequenza.
* Differenza dalle RNN: Il numero di operazioni non parallelizzabili non cresce con la lunghezza della sequenza.
* Distanza di interazione massima: O(1).
* Generalizzabile con più layer.
* Richiede un numero quadratico di confronti tra stati.

##### Self-Attention

* **Analogia con la ricerca in un archivio chiave-valore:** Una "ricerca fuzzy" dove la query corrisponde "softly" a tutte le chiavi, con pesi tra 0 e 1.
* **Calcolo:**
- **Trasformazione degli embedding:** Ogni embedding di parola ($x_i$) viene trasformato in query ($q_i$), chiavi ($k_i$) e valori ($v_i$) usando matrici di peso ($Q, K, V \in \mathbb{R}^{d \times d}$):
$$ \begin{aligned} \mathbf{q}_{i} &= Q \mathbf{x}_{i} \\ \mathbf{k}_{i} &= K \mathbf{x}_{i} \\ \mathbf{v}_{i} &= V \mathbf{x}_{i} \end{aligned} $$
- **Calcolo delle similarità e normalizzazione:** Similarità a coppie tra chiavi e query ($e_{ij} = \mathbf{q}_{i}^{\top} \mathbf{k}_{j}$), normalizzate con softmax:
$$ \alpha_{i j} = \frac{\exp\left(e_{i j}\right)}{\sum_{j^{\prime}} \exp\left(e_{i j^{\prime}}\right)} $$
- **Output:** Somma pesata dei valori:
$$ \mathbf{o}_{i} = \sum_{j} \alpha_{i j} \mathbf{v}_{j} $$
* **Caratteristiche:** Ogni stato "attende" a tutti gli altri stati (self-attention). Le similarità sono calcolate a coppie tramite prodotto scalare. $w_i$ rappresenta la i-esima parola, $x_i$ il suo embedding ($E \in \mathbb{R}^{d \times |V|}$).

##### Self-Attention: Meccanismo Base

* Confronto parola target (query) con tutte le altre (chiavi).
* Calcolo probabilità con softmax e somma pesata.

### Barriere e Soluzioni della Self-Attention

##### Ordine delle Parole:

* **Barriera:** La self-attention non considera intrinsecamente l'ordine delle parole.
* **Soluzione:** Aggiungere rappresentazioni posizionali agli input ($ \tilde{x}_i = x_i + p_i $). Utilizzo comune della somma, ma è possibile anche la concatenazione. Rappresentazioni sinusoidali considerano la posizione relativa.
##### Non Linearità:

* **Barriera:** Mancanza di non linearità, solo medie pesate.
* **Soluzione:** Applicare una rete feedforward a ciascun output della self-attention.
##### Futuro:

* **Barriera:** "Guardare al futuro" in sequenze temporali (es. traduzione automatica).
* **Soluzione:** Mascheramento: impostare artificialmente a 0 i pesi dell'attenzione futuri.

##### Componenti della Self-Attention come Blocco Costruttivo

**Rappresentazioni Posizionali:** Specificano l'ordine della sequenza.
**Non Linearità:** Applicate all'output, spesso come rete feedforward.
**Mascheramento:** Parallelizza le operazioni, impedendo "perdite" di informazioni dal futuro al passato.

##### Connessioni Residuali

* Migliorano l'addestramento.
* Formula: $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$ invece di $X^{(i)} = \text{Layer}(X^{(i-1)})$.
* Gradiente elevato (1) attraverso la connessione residuale.
* Bias verso la funzione identità.

##### Normalizzazione per Layer (Layer Normalization)

* **Idea:** Ridurre la variazione non informativa normalizzando a media unitaria e deviazione standard unitaria per layer.
* **Formulazione Matematica:**
	* Media: $\mu = \frac{1}{d} \sum_{j=1}^d x_j$
	* Deviazione standard: $\sigma = \sqrt{\frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2}$
	* Output: $ \text{output} = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta $
	* **Processo:** Normalizzazione tramite media e varianza scalari, modulazione tramite guadagno ($\gamma$) e bias ($\beta$) appresi. Normalizzazione dimensione per dimensione.
	* **Successo:** Potrebbe essere dovuto alla normalizzazione dei gradienti. Stabilizza l'apprendimento per layer.

##### Standardizzazione dei Vettori

* **Pre-normalizzazione:** Metodo preferito per l'efficienza dell'addestramento. (Blocco "add & norm" nel Transformer)

##### Sequence-Stacked Attention

* **Definizione delle Variabili:**
	* $X = [x_1; x_2; ...; x_n] \in \mathbb{R}^{n \times d}$: concatenazione dei vettori di input (n = numero elementi, d = dimensione vettore).
	* $X_K, X_Q, X_V \in \mathbb{R}^{n \times d}$: matrici trasformate da X per chiavi (K), query (Q) e valori (V).
	* **Calcolo dell'Output:** `output = softmax(XQ(XK)^T)XV ∈ ℝ^(n × d)`
* **Passaggi del Calcolo:**
	* Prodotto scalare query-key: `XQ(XK)^T`
	* Applicazione della softmax: `softmax(XQ(XK)^T)`
	* Media pesata: `softmax(XQ(XK)^T)XV`

##### Multi-Head Attention

* **Concetto:** Combinazione di profondità della rete e molteplicità delle teste (più matrici query-valore). Ogni testa lavora in parallelo.
* **Vantaggi:**
	* Maggiore interpretazione delle relazioni tra parole (polisemia).
	* Controllo sulla tipologia di relazione tra le parole tramite layer.
	* Iperparametro: numero di teste (per ogni layer).
* **Calcolo con più teste:**
	* Trasformazione con matrici Q e K.
	* Applicazione della softmax.
	* Combinazione con somma pesata. Estensione naturale da matrici a tensori (spessore = numero teste).
	* Costo computazionale non aumenta significativamente.
	* Riformulazione delle matrici: `XQ ∈ ℝ^(n × d)` riformulato in `ℝ^(n × h × d/h)` (stesso per XK, XV), dove h è il numero di teste. Trasposizione in `ℝ^(h × n × d/h)` (asse testa come asse batch).
	* **Calcolo dei Prodotti Scalari Query-Key:** `XQ(XK)^T` (tutti i set di coppie di punteggi di attenzione).
	* **Calcolo della Media Ponderata:** `softmax((XQ(XK)^T)/√d) XV` (fattore di scala √d per stabilizzare il gradiente).
	* **Output:** Matrice `n × d`.
	* **Risoluzione del problema della polisemia:** Definizione di più "teste" di attenzione tramite molteplici matrici Q, K, V (`Qℓ, Kℓ, Vℓ ∈ ℝ^(d × d/h)`, dove ℓ varia da 1 a h).

##### Multi-Head Self-Attention

* **Meccanismo:** Ogni testa di attenzione esegue l'attenzione indipendentemente: `outputₗ = softmax(XQₗKₗᵀXᵀ) × XVₗ`, dove `outputₗ ∈ ℝⁿˣᵈ/ʰ`. Gli output di tutte le teste vengono poi combinati: `output = [output₁; ...; outputʰ]Y`, con `Y ∈ ℝᵈˣᵈ`.
* **Indipendenza delle teste:** Ogni testa può focalizzarsi su aspetti diversi dell'input.
* **Attenzione Scalata con Prodotto Scalare:** Per migliorare la stabilità dell'addestramento, si scala il prodotto scalare: `outputₑ = softmax(XQₑKₑᵀXᵀ/√(d/h)) × XVₑ`.
* **Problema:** Prodotti scalari grandi con alta dimensionalità (`d`) portano a gradienti piccoli.
* **Soluzione:** La divisione per `√(d/h)` stabilizza i gradienti.

##### Transformer Decoder

* **Struttura del Blocco di Decodifica:** Ogni blocco contiene:
* **Self-Attention (masked):** Focalizzazione su diverse parti dell'input, impedendo di "vedere" parole future durante la generazione sequenziale.
* **Add & Norm**: Aggiunta dell'input all'output e normalizzazione.
* **Feed-Forward Network**: Trasformazione dell'input.
* **Add & Norm**: Aggiunta dell'input all'output e normalizzazione.
* **Funzionamento:** L'input passa attraverso una pila di blocchi di decodifica, ripetendo le operazioni di self-attention, Add & Norm e feed-forward. Il mascheramento nella self-attention è opzionale per compiti come la classificazione.

##### Transformer Encoder-Decoder

* **Architettura:** Utilizza un Encoder e un Decoder.
* **Encoder:** Transformer standard (bidirezionale).
* **Decoder:** Transformer modificato con cross-attention.
* **Tipi di Attenzione:**
	* **Multi-head self-attention** (*encoder*).
	* **Masked multi-head self-attention** (*decoder*).
	* **Cross-attention** (*decoder sull'output dell'encoder*). L'output passa attraverso un ulteriore Add & Norm.
	* **Cross-Attention:** Simile alla self-attention, ma le query provengono dal decoder e le chiavi/valori dall'encoder. Utilizzata nella traduzione automatica (*seq2seq*).

##### Ottimizzazioni del Decoder

* **Connessioni Residuali:** Aggiungono l'input al risultato di un layer, migliorando il flusso del gradiente.
* **Normalizzazione per Layer:** Normalizza i vettori nascosti, stabilizzando l'addestramento. Spesso combinate come "Add & Norm".

##### Architettura Transformer:

* **Encoder:** Genera vettori di output $h_1, \dots, h_n \in \mathbb{R}^d$.
* **Decoder:** Riceve vettori di input $z_1, \dots, z_n \in \mathbb{R}^d$.
* **Meccanismo Attenzione:**
	* **Chiavi (k):** $k_i = Kh_j$ (estratte dall'encoder).
	* **Valori (v):** $v_i = Vh_j$ (estratte dall'encoder).
	* **Query (q):** $q_i = Qz_i$ (estratte dal decoder).
	* **Output finale dell'encoder:** `h` (rappresenta la codifica finale).

##### Pre-processing:

* **Tokenizzazione Subword:** Unica fase di pre-processing rilevante.

