### Operazioni Point-to-Point

Le operazioni point-to-point sono operazioni di pre-processing che agiscono su ogni singolo punto di un'immagine. La formula generale è:

$$G(x) = h(f(x))$$

dove:

* **x:** rappresenta la coordinata del punto nell'immagine.
* **f:** rappresenta l'immagine di input.
* **h:** rappresenta la funzione di trasformazione applicata al punto.

Possiamo lavorare sia con il valore associato alla coordinata **x** che con il valore stesso dell'immagine **f(x)**.

### Operazioni di Filtro

Le operazioni di filtro sono un'estensione delle operazioni point-to-point. Data una coordinata **x**, definiamo un intorno (un "vicinato") attorno a quel punto e applichiamo una trasformazione a tutti i punti nell'intorno. Il valore in output sarà una trasformazione su tutto il vicinato.

La formula generale per le operazioni di filtro è:

$$G(x) = h(f(\text{neighbour}(x)))$$

Le operazioni di filtro hanno un corrispettivo nell'analisi del segnale. Le immagini possono essere interpretate come segnali bidimensionali (x e y). Su queste due dimensioni, abbiamo la funzione di intensità (o valore).

### Convoluzione

La convoluzione è un'operazione di filtro che consiste nell'applicare una funzione **f** (il filtro) all'immagine. Questa operazione è spesso utilizzata per il filtraggio spaziale lineare.

La formula per il filtraggio spaziale lineare è:

$$\sum_{k=-a}^a \sum_{l=-b}^b f[k+a,l+b] \times I[i+k,j+l]$$

##### Cosa significa applicare questa funzione su una qualsiasi immagine?

Applicare questa funzione ad un'immagine significa scorrere il filtro **f** su tutta l'immagine, calcolando la somma ponderata dei valori dei pixel nell'intorno del filtro per ogni posizione. Il risultato è un'immagine filtrata, dove i valori dei pixel sono stati modificati in base al filtro applicato.
[[3.Filters.pdf#page=8|3.Filters, p.8]]
==spiegazione da inserire==

## Gestione dei Bordi nelle Operazioni sulle Immagini

Quando si applicano operazioni su immagini, come filtri o trasformazioni, è necessario gestire i bordi dell'immagine. Esistono diverse strategie per farlo:

* **Ignorare i bordi:** Si può semplicemente ignorare il calcolo dei valori dei pixel che si trovano ai bordi dell'immagine. Questo può portare a risultati non desiderati, soprattutto se l'operazione richiede informazioni dai pixel vicini.
* **Estensione con zeri:** Si può estendere l'immagine con una cornice di zeri attorno ai bordi. Questo permette di calcolare i valori dei pixel ai bordi, ma può introdurre artefatti nell'immagine.

##### Quali sono le potenzialità di queste operazioni?

Le potenzialità di queste operazioni dipendono dall'applicazione specifica. In generale, la scelta del metodo di gestione dei bordi influenza il risultato finale dell'operazione.

##### Vantaggi:

* **Trasformazione dei valori:** Le operazioni sui bordi permettono di trasformare i valori dei pixel in modo controllato, ad esempio per calcolare la media dei pixel vicini.
* **Semantica delle immagini:** La gestione dei bordi può influenzare la semantica dell'immagine, ad esempio, estendendo l'immagine con zeri si può creare un effetto di "sfocatura" ai bordi.

#### Smoothing mediante filtraggio spaziale

* **Filtro Media (Average Filter):**
	* Sostituisce l'intensità del pixel con il valore medio dei suoi vicini.
	* **Smoothing:** Attenua le transizioni brusche (sharp) d'intensità.

##### Applicazione:

Applicare un filtro che moltiplica per $k = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$ (calcola la media dei pixel) può essere utilizzato per sfumare l'immagine (filtro di smoothing o average). Questo attenua le transizioni brusche.

## Convoluzione 2D

```python
def convolve2d(image, kernel):
    """
    Questa funzione prende un'immagine e un kernel e restituisce la loro convoluzione.

    :param image: un array numpy di dimensione [altezza_immagine, larghezza_immagine].
    :param kernel: un array numpy di dimensione [altezza_kernel, larghezza_kernel].
    :return: un array numpy di dimensione [altezza_immagine, larghezza_immagine] (output della convoluzione).
    """

    kernel_height, kernel_width = kernel.shape

    a = kernel_width // 2
    b = kernel_height // 2

    # Flip the kernel

    kernel = np.flipud(np.fliplr(kernel))
    # convolution output

    output = np.zeros_like(image)

    # Add zero padding to the input image

    image_padded = np.zeros((image.shape[0] + 2*a, image.shape[1] + 2*b))
    image_padded[a:-a, b:-b] = image

    # Loop over every pixel of the image

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image

            output[y, x] = (kernel * image_padded[y: y+kernel_width, x: x+kernel_height]).sum()

    return output
```

##### Note:

* Questo approccio di doppio `for` non si utilizza in pratica. Si usano i prodotti tra matrici.
* Il prodotto con un kernel 3x3 non fa in modo che si notino molte differenze.
* Un filtro 5x5, come $k = \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1& 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 &  1 & 1 \\ 1 & 1 & 1& 1& 1   \\ 1 & 1 & 1& 1 & 1  \end{bmatrix}$, già migliora la cosa.
* In questo caso, i bordi vengono riempiti di nero.

## Filtro Gaussiano

##### Definizione:

La funzione del filtro gaussiano è definita come:

$$f(i,j)=\frac{1}{2 \pi \sigma^2 }e^{- \frac{i^2+j^2}{2 \sigma^2}}$$

##### Caratteristiche:

* I pesi decadono con la distanza dal centro.
* Riduce l'effetto di blurring (sfocatura) durante l'operazione di smoothing.
* I coefficienti sono inversamente proporzionali alla distanza dal pixel centrale.
* Con maschere piccole non vi sono grandi differenze.
* Questa funzione è espressa rispetto ad un punto medio e l'intensità diminuisce in maniera proporzionale esponenziale quadratica rispetto alla distanza dalla media.

##### Interpretazione:

Il filtro gaussiano rappresenta il peso che possiamo assegnare alla matrice.

##### Parametri:

* **$\sigma$**: rappresenta la rapidità con cui facciamo degradare i pesi (la forma della campana). Definisce le dimensioni del filtro.

##### Esempi:

* **$\sigma = 1$**: il filtro diventa un filtro 3x3 con forma kernel $k = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$.
* **$\sigma = 3$**: il filtro diventa un filtro 5x5.
* **$\sigma = 5$**: il filtro diventa un filtro 20x20.

##### Osservazioni:

* Passato un certo range, i valori del filtro diventano poco significativi.

## Filtraggio Spaziale Lineare

##### Formula:

$$h[i,j]= \sum_{k=-a}^a \sum_{l=-b}^b f[k+a,l+b] \times I[i+k,j+l]$$

##### Importanza:

* **Fondamentale!**
* **Migliora l'immagine:**
	* Denoise (riduzione del rumore)
	* Ridimensionamento
	* Aumento del contrasto
	* ...
* **Estrae informazioni dall'immagine:**
	* Texture
	* Bordi (edges)
	* Punti distintivi
	* ...
* **Trova patterns:**
	* Template matching (ricerca di modelli)

## Esempi di Filtri

### 1) Filtro Identità

![[2) Filters-20241001190420784.png|241]]
* Restituisce l'immagine originale senza modifiche.

### 2) Filtro di Traslazione a Destra

![[2) Filters-20241001190551085.png|241]]
* Sposta l'immagine di un pixel verso destra.

### 3) Filtro per Bordi Verticali

![[2) Filters-20241001190601398.png|350]]

* **Struttura:** La parte centrale del filtro è composta da zeri.
* **Funzionamento:** Si ottiene sottraendo la prima colonna dell'immagine all'ultima.
* **Effetto:** Evidenzia i cambiamenti di intensità verticale.
* **Interpretazione:** Fornisce un'interpretazione semantica del contenuto dell'immagine basata sulle variazioni di intensità.
* **Perdita di informazione:** Il valore assoluto della differenza viene utilizzato, quindi si perde il senso della direzione (se il bordo è più scuro a sinistra o a destra).
**Domanda:** Perché trovo contorni neri o bianchi?
* **Risposta:**
	* La differenza tra un valore positivo e uno negativo risulta in un valore nero.
	* La differenza tra un valore negativo e uno positivo risulta in un valore bianco.

### 4) Filtro per Bordi Orizzontali

![[2) Filters-20241001214837166.png|405]]
* Simile al filtro per bordi verticali, ma evidenzia i cambiamenti di intensità orizzontale.

### 5) Filtro per Migliorare il Contorno

![[2) Filters-20241001214853302.png|330]]
* Combina due filtri: uno per aumentare il contrasto e uno per lo smoothing.
* Il risultato è un'immagine con contorni più netti.
* Il filtro rimuove la media, evidenziando i bordi.
* Aumenta l'intensità dei pixel che si trovano sui bordi.

## Convoluzione

##### Definizione:

$$(f * g)(x) = \int_{-\infty}^{\infty} f(u) g(x-u) \, du$$

##### Interpretazione:

* La convoluzione deriva dalla teoria del segnale.
* Un segnale può essere potenzialmente infinito, quindi anche il filtro può esserlo.

##### Caso bidimensionale discreto:

* La convoluzione è un'operazione che prende due funzioni e restituisce una terza funzione che rappresenta la loro sovrapposizione.
* Nel caso bidimensionale, la convoluzione viene applicata a due immagini:
* Un'immagine di input.
* Un filtro.
* Il filtro viene spostato su tutta l'immagine, calcolando la somma ponderata dei valori dei pixel nell'intorno del filtro per ogni posizione.
* Il risultato è un'immagine filtrata, dove i valori dei pixel sono stati modificati in base al filtro applicato.

### Convoluzione e Correlazione

- **Convoluzione** $(f * I)(x,y) = \sum_{i,j=-\infty}^\infty f(i,j) I(x-i,y-j)$
- $(x,y)$ = Segnale filtrato
- $f(i,j)$ = Filtro
- $I(x-i,y-j)$ = Segnale di input
- **Correlazione** $(f * I)(x,y) = \sum_{i,j=-\infty}^\infty f(i,j) I(x-i,y-j)$
* **Nessuna differenza se il filtro è simmetrico.**

##### Spiegazione del mirroring:

Facciamo il mirroring del filtro sia sull'asse X che sull'asse Y. In pratica, capovolgiamo l'immagine del filtro. L'elemento più in alto a sinistra del filtro viene moltiplicato con l'elemento più in basso a destra dell'immagine.

##### Motivo del mirroring:

Questo perché la convoluzione è un'operazione di tipo "scorrevole" e il mirroring del filtro assicura che l'operazione sia simmetrica rispetto all'asse X e Y. In altre parole, il filtro viene applicato in modo uniforme su tutta l'immagine, indipendentemente dalla direzione in cui viene spostato.

##### Correlazione:

L'operazione di correlazione prende gli elementi nell'ordine in cui si trovano. Questo argomento non è trattato nel corso.

==skip fino a filtri separabili==
## Filtri Separabili

Un filtro è separabile se lo stesso effetto può essere ottenuto applicando in sequenza due filtri più semplici.
![[Pasted image 20240930181445.png]]

**Esempio:** (filtro di edge)

Quando applico il filtro, eseguo due operazioni: una sull'asse Y e una sull'asse X. Queste due operazioni sono separate, quindi è come se le applicassi in sequenza. Il vettore verticale, immaginato solo sull'asse X, è un'operazione di media pesata. Questo filtro fa la differenza dei valori medi: calcola la media lungo X e fa la differenza delle medie.

##### Semantica specifica:

I filtri che godono di queste proprietà sono detti separabili (ad esempio, smoothing).

##### Vantaggio:

Siccome sono separabili, possiamo applicare un'operazione sulle righe e una sulle colonne, quindi è più veloce.

## Filtri e Intensità

##### I filtri lavorano sulle intensità.

##### Filtri passa-basso e passa-alto:

* **Filtro passa-alto:** fa "passare" le componenti ad alta frequenza e riduce o elimina le componenti a bassa frequenza.
* **Filtro passa-basso:** fa "passare" le componenti a bassa frequenza e riduce o elimina le componenti ad alta frequenza.

##### Tagliano le intensità:

Ad esempio, lo smoothing è un passa-basso perché taglia le frequenze alte.

* **Filtro passa-basso (es., average filter):**
	* La somma dei coefficienti vale 1 -> regioni uniformi preservate e non uniformi tendono ad uniformarsi.
	* Offusca sia i bordi che il rumore.
* **Filtro passa-alto:**
	* La somma dei coefficienti è 0 -> la risposta sulle componenti a bassa frequenza è prossima a zero.

##### Col passa-alto evidenziamo i punti di cambio della frequenza.

Siccome si parla di discontinuità, usiamo la derivata, che misura la variazione che la funzione esprime al limite.

## Edge Detection

* **Come si possono identificare le discontinuità?**
	* Si calcolano le derivate.
	* Nei punti di discontinuità le derivate sono grandi.
* **Come si calcolano le derivate di un segnale discreto?**
	* Calcolando le differenze finite.

##### La variazione si calcola con le derivate: nei punti di discontinuità le derivate sono grandi, dunque c'è un possibile edge.

![[2) Filters-20241001215021438.png|440]]
la definizione che c'è di derivata sul segnale discreto è la seconda, la def di derivata seconda sul segnale discreto è la terza

cosa calcolano queste derivate?
## Derivate di immagini digitali

![[2) Filters-20241001215049063.png|505]]
##### Scenario:

* Iniziamo con un segnale di intensità alta (6).
* Scendiamo a 1 e manteniamo un andamento "smooth" (liscio).
* Saliamo bruscamente a 6.

##### Analisi della Derivata:

* **Edge:** Quando la derivata passa da 0 a 5, si identifica un edge (bordo).
* **0-Crossing:** Quando la derivata passa da 5 a -5, si verifica uno 0-crossing (cambio rapido).
* Gli 0-crossing indicano variazioni significative di intensità.
* Le piccole variazioni di intensità sono poco significative.
* La derivata seconda registra le variazioni significative tramite il meccanismo dello 0-crossing.

##### Derivata Seconda:

* La derivata seconda calcola le variazioni della derivata prima.
* Quando la derivata prima sale ripidamente, la derivata seconda identifica gli edge.

##### Applicazione con Filtri:

* Possiamo utilizzare i filtri per calcolare le derivate.
* Il filtro di Sobel calcola la derivata prima.

## Filtro sobel

- **Orizzontale**:
$$
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 \\2 \\1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0 & -1\end{bmatrix}
$$
- **Verticale**:
$$
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\1 & 2 & 1\end{bmatrix}
\begin{bmatrix}
1 \\0 \\-1
\end{bmatrix}
\cdot
\begin{bmatrix}
1 &2 &1
\end{bmatrix}
$$

![[2) Filters-20241001215112251.png|509]]

##### Calcolo del Gradiente:

* A partire dalle due derivate parziali (rispetto a x e y) di un'immagine, possiamo calcolare il gradiente.
* Il gradiente è un vettore che indica la direzione e l'ampiezza della massima variazione di intensità dell'immagine.
![[2) Filters-20241001215952259.png]]

* La direzione del gradiente è data dalla tangente iperbolica del rapporto tra le due derivate parziali.
* L'ampiezza del gradiente è data dalla radice quadrata della somma dei quadrati delle due derivate parziali.

##### Interpretazione del Gradiente:

* **Direzione:** La direzione del gradiente indica l'orientamento del bordo.
* **Ampiezza:** L'ampiezza del gradiente indica la forza del bordo. Un'ampiezza alta indica un bordo netto e ben definito, mentre un'ampiezza bassa indica un bordo sfumato o poco definito.

![[2) Filters-20241001215940910.png]]
- Con le derivate seconde otteniamo il filtro laplaciano, che è la somma delle due derivate parziali **seconde**, una rispetto a x e l'altra rispetto a y.
## Laplaciano di immagini digitali

* **Filtro isotropico:** invariante rispetto alle rotazioni.
* **Laplaciano può essere utilizzato per image sharpening.**
* **Una proprietà desiderabile è che la risposta del filtro sia indipendente dalla direzione delle discontinuità (isotropia).**

##### Il filtro laplaciano:

* **Ci mostra la direzione.**
	* $w = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$
* **Applicare questo filtro equivale a fare la somma delle due derivate seconde, in x e y.**

##### Effetto dello zero-crossing (variazione molto ripida di intensità):

* **Abbiamo una striscia bianca e poi una striscia nera.**
* **Questo perché il filtro laplaciano, essendo una derivata seconda, evidenzia le variazioni di intensità più brusche.**
	* Quando si passa da un'area di alta intensità (bianca) ad un'area di bassa intensità (nera), la derivata seconda assume un valore molto alto, creando una striscia bianca.
	* Viceversa, quando si passa da un'area di bassa intensità ad un'area di alta intensità, la derivata seconda assume un valore molto basso, creando una striscia nera.
	* Esempio:
![[2) Filters-20241001123217625.webp|148]]

## Definizione dei Contorni e Problemi con la Derivata

##### Definizione dei Contorni:

* I contorni "veri" sono quelli definiti dal Laplaciano.

##### Problemi con la Derivata:

* Le derivate sono calcolate sulla scala determinata dalla risoluzione dell'immagine. Questo significa che la precisione della derivata è limitata dalla risoluzione dell'immagine.
* Le immagini rumorose comportano un'alta instabilità della derivata. Il rumore introduce variazioni casuali nell'immagine, che vengono amplificate dalla derivata, rendendo difficile l'individuazione dei contorni reali.

##### In sintesi:

* Il Laplaciano è un operatore più robusto per la definizione dei contorni rispetto alla derivata.
* La derivata è sensibile alla risoluzione dell'immagine e al rumore, il che può portare a risultati imprecisi.

![[2) Filters-20241001123544559.webp|347]]
in questo esempio c'è molto rumore nella zona dei capelli
![[2) Filters-20241001215146241.png|374]]
che succede col laplaciano?
![[2) Filters-20241001215206671.png|390]]

in presenza di segnali instabili la caratteristica del gradiente viene persa, ovvero la capacità di definire i bordi, perchè il gradiente è confuso da piccole variazioni

come fare a risolvere? si usa il filtro di media(blurring)
![[2) Filters-20241001215224651.png|551]]
## Operazioni in Cascata e Recupero Informazioni

##### Operazioni in Cascata:

* Eseguire una serie di operazioni in cascata (una dopo l'altra) può permettere di recuperare informazioni utili da un'immagine.

##### Esempio: Filtro Gaussiano e Derivata:

* Applicare il filtro Gaussiano all'immagine originale produce un risultato errato per l'estrazione dei bordi: la "campana" del gradiente si trova all'inizio del bordo, non dove il bordo è realmente presente.
* Applicando la derivata all'immagine sfocata (blurred) con il filtro Gaussiano, si riesce a recuperare l'informazione sui bordi.

##### Spiegazione:

* Il filtro Gaussiano, essendo un filtro passa-basso, attenua il rumore e le piccole variazioni di intensità, ma allo stesso tempo sfuma i bordi.
* Applicando la derivata dopo il filtro Gaussiano, si riesce a recuperare l'informazione sui bordi, eliminando al contempo il rumore.

##### In sintesi:

* L'applicazione di una serie di operazioni in cascata può essere utile per ottenere informazioni specifiche da un'immagine.
* Il filtro Gaussiano e la derivata sono un esempio di come combinare filtri per ottenere risultati desiderati.

![[2) Filters-20241001215243027.png|526]]
Matematicamente, applicare i due filtri in cascata $S_{x}(\cdot G_{\sigma}\cdot I)$ equivalale ad applicare $(S_{x}\cdot G_{\sigma})\cdot I$

![[2) Filters-20241001215308187.png|471]]
I filtri sono fatti così
![[2) Filters-20241001215405724.png|632]]

## Trasformata di Fourier e Filtri

##### Trasformata di Fourier:

* La trasformata di Fourier permette di trasformare le immagini in due componenti: **ampiezza** e **fase**.
* L'ampiezza rappresenta la quantità di energia presente a ciascuna frequenza.
* La fase rappresenta lo sfasamento di ciascuna frequenza.

##### Filtri nella Trasformata di Fourier:

* I filtri possono essere interpretati come un'operazione di "pulizia" nella rappresentazione delle frequenze.
* Ad esempio, un filtro passa-basso elimina le alte frequenze, mentre un filtro passa-alto elimina le basse frequenze.

##### Derivata e Laplaciano come Filtri:

* La derivata e il Laplaciano sono filtri **passa-alto**.
* Sono espressi come combinazioni di frequenze.

![[2) Filters-20241001215435159.png|408]]

Le operazioni di filtraggio sono operazioni che lavorano sia sul dominio dell'intensità che delle frequenze
![[2) Filters-20241001215509121.png|428]]
Posso applicare o il filtro come convoluzione(intensità) o nelle frequenze e la convoluzione assume interpretazione di moltiplicazione punto a punto tra matrici(operazione efficente e parallelizzabile)

https://github.com/gmanco/cv_notebooks/blob/master/labs_lecture/lab02/4.Fourier_transform.ipynb

![[2) Filters-20241001125616233.webp|405]]
conservo i valori bassi (?)

altre trasformate utili
![[2) Filters-20241001220035817.png]]

## Estrazione dei Contorni: Gradienti e Canny Edge Detector

##### Informazioni sui Contorni:

* La prima informazione che ci interessa estrarre da un'immagine è quella dei contorni.
* Lo strumento principale per farlo sono i **gradienti**, di primo o secondo ordine (Laplaciani).

##### Problemi con i Gradienti:

* L'informazione del gradiente da sola non è sufficiente perché genera troppi artefatti, ovvero troppi dettagli all'interno dei bordi.
* Altri problemi:
* **Rumore:** risolvibile con il filtro Gaussiano.
* **Toni di grigio:** risolvibile con il *thresholding*.
* **Bordi di diverso spessore**.

##### Soluzione: Canny Edge Detector

* Per risolvere questi problemi, si utilizza il **Canny edge detector**.
* È un algoritmo semplice che estende il *gradient-based filtering*.

##### Caratteristiche del Canny Edge Detector:

* **Spessori uniformi:** ottenuto tramite la *non-maximal suppression*.
* **Rimozione di artefatti:** tramite il *double thresholding* e l'isteresi.

##### Funzionamento del Canny Edge Detector:

* Consiste nel prendere un'immagine, applicare il filtro Gaussiano al gradiente e poi eseguire una serie di operazioni.

##### Vantaggi del Canny Edge Detector:

* Risolve i problemi elencati sopra.
* Produce contorni con spessori uniformi e riduce gli artefatti.

![[2) Filters-20241001220103586.png|720]]
il gradiente, nell'espressione in termini di angolo, punta a valori ad alta intensità

l'algo calcola l'angolo nella direzione della normale, poi lo discretizza in una serie di zone (8), utilizzate per identificare i pixel massimali lungo la direzione del gradiente.

## Non maximal suppression

•La non-maximal suppression è un algoritmo che serve a eliminare i pixel non massimali lungo la direzione del gradiente, se i pixel adiacenti hanno un valore più alto
![[2) Filters-20241001215549309.png|299]]
- Le 8 zone corrispondono a 8 possibili combinazioni di triplette all interno dell immagine

![[2) Filters-20241001215605374.png|302]]
##### Funzionamento:

- Per ogni pixel, calcola la direzione del gradiente.
- Lungo la direzione del gradiente, controlla se esiste un pixel con intensità maggiore.
- Se esiste un pixel con intensità maggiore, il pixel corrente non è massimale e viene soppresso (impostato a nero).
- Ripeti i passaggi 1-3 per tutti i pixel dell'immagine.

##### Risultato:

L'algoritmo di non-maximal suppression produce un'immagine in cui i pixel massimali lungo la direzione del gradiente sono mantenuti, mentre i pixel non massimali vengono soppressi. Questo aiuta a identificare i bordi in modo più preciso, eliminando il rumore e i falsi positivi.

![[2) Filters-20241001215626821.png|484]]

## Level-wise thresholding, hysteresis

* **Strong/Weak/Irrelevant pixels:**
	* **Strong pixels:** intensità alta (contribuiscono sicuramente ai bordi)
	* **Weak pixels:** intensità non alta, ma neanche bassa. Li teniamo da parte.
	* **Irrelevant pixels:** intensità bassa, da rimuovere.
* **Utilizzo di due soglie:**
	* **High threshold:** per identificare strong pixels.
	* **Low threshold:** per identificare irrelevant pixels.
	* Tutti i pixel nel mezzo delle due soglie sono weak e verranno gestiti dal meccanismo dell'isteresi.

**Problema:** con questo meccanismo molti elementi vengono eliminati, non ottenendo una soluzione valida.

**Soluzione:** utilizzare due soglie, una alta e una bassa.

* **Strong pixels:** fanno parte dei bordi.
* **Weak pixels:** potrebbero far parte dei bordi. Per decidere, usiamo il meccanismo dell'isteresi:

## Hysteresis

* **Attrazione gravitazionale:** se un weak pixel ha uno strong pixel nel vicinato, diventa anche esso uno strong pixel. Altrimenti, diventa irrilevante e viene soppresso.

![[2) Filters-20241001215646672.png|415]]

![[2) Filters-20241001215657465.png|547]]

