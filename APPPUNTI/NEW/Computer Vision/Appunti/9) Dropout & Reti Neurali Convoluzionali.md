## Dropout

Il layer di Dropout ha un compito specifico: modificare il comportamento di un layer denso classico. In un layer denso, ogni input si propaga a tutti i neuroni del layer, influenzando l'output.
Il Dropout, invece, "*spegne*" alcuni di questi neuroni in modo casuale durante l'addestramento. In pratica, i neuroni successivi non considerano l'output dei neuroni spenti.

Immaginate un'immagine divisa in patch. Durante l'addestramento, alcuni neuroni potrebbero essere inattivi per un certo patch, mentre altri saranno attivi. Questo processo è casuale e varia ad ogni epoca di addestramento.

#### Obiettivo

L'obiettivo principale del Dropout è **migliorare la capacità di generalizzazione della rete**. Spegnendo casualmente i neuroni, si evita che la rete si "specializzi" eccessivamente su specifici pattern presenti nei dati di training. In altre parole, si impedisce l'overfitting. Senza Dropout, ogni neurone potrebbe diventare troppo dipendente da un ristretto insieme di input. Con il Dropout, invece, i neuroni sono "costretti" ad imparare rappresentazioni più robuste e generali, utili anche con dati mai visti prima.

#### Implementazione

Durante la fase di addestramento, il Dropout si implementa "spegnendo" i neuroni in base ad una probabilità predefinita, chiamata **tasso di Dropout**. Questo parametro, tipicamente compreso tra il 10% e il 50%, determina la percentuale di neuroni disattivati ad ogni iterazione. Durante la fase di test, invece, il layer di Dropout si comporta come una **funzione identità**: tutti gli input vengono propagati all'output senza modifiche. Questo perché, durante il test, l'obiettivo è valutare le prestazioni della rete su dati sconosciuti, e non è necessario "forzare" la generalizzazione.

#### Utilizzo

È fondamentale impostare correttamente la modalità di esecuzione del modello (training o valutazione) per garantire il corretto funzionamento del Dropout. Fortunatamente, framework come PyTorch semplificano questo processo con metodi specifici come `.train()` e `.eval()`. Il Dropout si utilizza principalmente nella parte finale della rete, in particolare nei layer densi responsabili della classificazione. Ad esempio, in architetture come la VGG, l'utilizzo del Dropout dopo i layer convoluzionali e prima dei layer densi ha dimostrato di migliorare significativamente le prestazioni.

#### Implementazione Tecnica

Sebbene si parli di "spegnere" i neuroni, l'implementazione pratica del Dropout non prevede la rimozione fisica di elementi dalla rete. Si tratta, invece, di **mascherare** l'output dei neuroni selezionati, azzerando i loro contributi durante la propagazione. Esistono diverse tecniche per implementare il Dropout. Una soluzione comune è utilizzare un layer dedicato che applica una maschera casuale all'input. In alternativa, è possibile integrare la funzionalità del Dropout direttamente all'interno dei layer esistenti.

### Vantaggi dell'utilizzo di un Layer Dedicato per il Dropout

Utilizzare un layer dedicato per il Dropout offre maggiore flessibilità rispetto all'integrazione diretta nei layer esistenti. Ad esempio, permette di:

* **Combinare** il Dropout con altri tipi di layer.
* **Posizionare** il Dropout in punti specifici della rete.
* **Sperimentare** con diversi tassi di Dropout.

## Architettura delle Reti Neurali Convoluzionali

Come abbiamo visto, la rete può essere sintetizzata in questo modo: abbiamo una serie di convoluzioni a scala, ognuna delle quali comprime la risoluzione della feature map, ad esempio da 18x28 a 14x14, 10x10, 5x5, fino ad ottenere un vettore da dare in pasto ad un'unità di classificazione. Alla fine di questo processo otterremo un tensore di dimensione pari al numero di classi totali, in questo caso 120, sulle quali calcolare la massimizzazione. In pratica, ogni volta riduciamo la risoluzione della feature map, non la dimensione, la riduciamo. Vedremo in seguito da cosa dipende questa riduzione.

Potreste chiedervi come passiamo da una feature map di 5x5 a 120 componenti, che è il nostro formato finale. Partiamo dalla risoluzione iniziale, 32x32. Applichiamo un filtro 5x5. I parametri che determinano la dimensione del filtro sono importanti, ma per il momento concentriamoci sul fatto che applicando il filtro 5x5 otteniamo una feature map 28x28. Questi sono calcoli che vedremo in dettaglio più avanti, la cosa importante è che il kernel comprime l'informazione restituendoci un certo numero di feature map, che in questo caso rappresentano il nostro tipo di informazione.

Comprimendo, e quindi se ogni filtro sintetizza una porzione 5x5 dell'input, il rischio è di perdere informazioni. Ad esempio, se abbiamo un kernel 5x5 che riconosce linee oblique, ma per qualche motivo non riconosce le linee oblique non centrate nel centro della maschera 5x5, in questo caso perderemmo l'informazione. Un meccanismo adottato per gestire più casistiche è quello di ridurre la dimensione del frame e aumentare la profondità del filtro. Invece di addestrare un unico kernel, ne addestriamo 6. Ognuno riconoscerà pattern diversi, perché di fatto questi kernel devono essere addestrati a riconoscere dei pattern. Quali siano questi pattern non ci interessa, perché sarà il processo di addestramento a delinearli. La cosa fondamentale è che ogni kernel si specializza per individuare uno o più pattern.

Aumentando il numero di filtri, 6, 16, 16, 16, 16, compensiamo la specializzazione di un kernel aumentando il loro numero. In generale, questo è un pattern che esamineremo spesso: più si scende in profondità, più piccola sarà la risoluzione della feature map (in questo caso abbiamo 28x28, 14x14, 12x12, 5x5) e più sarà grande il numero di filtri che vogliamo apprendere. Quindi, aumentando la profondità, diminuisce la risoluzione.

Il problema è sempre il costo computazionale. Dobbiamo applicare più filtri perché ci sono operazioni di convoluzione. Come si riduce questo numero di operazioni? Ci sono varie strategie architetturali, ne vedremo alcune, ma di fatto per ridurre il numero di operazioni o si riduce il numero di filtri o si riduce la dimensione dell'immagine sulla quale applicarli. In questo caso si adotta proprio questa tecnica, cioè si riduce la risoluzione, che diventa sempre più piccola, fino ad arrivare a una risoluzione 5x5.

Il problema è che la convoluzione ha dei parametri che determinano la dimensione del filtro. Un parametro efficiente è un iperparametro, stabilito in fase di training. Come? Facendo delle prove. Non ci sono regole fisse. Quello che si sa è che tipicamente un principio che funziona è questo: se si riduce la stride size aiuta ad avere più feature map e viceversa. Sempre per bilanciare da un lato l'esigenza di avere più possibilità di apprendere pattern, più pattern possibile, e dall'altro diminuire il numero di operazioni. Però, un numero o una regola precisa non c'è. Bisogna andare per tentativi ed è appunto un iperparametro, con un insieme di parametri che costituiscono la rete in termini di input, numero di layer e così via, ma che non sono parametri addestrati, ma vengono definiti a priori.

## Pooling Layers e Funzioni di Attivazione

### Pooling Layers

I **pooling layers** sono un altro tipo di blocco utilizzato nelle reti neurali convoluzionali per ridurre il numero di locazioni (feature map) e quindi la complessità computazionale.

Un esempio comune è il **max pooling**, che calcola il massimo valore all'interno di una finestra di dimensioni definite (ad esempio, 4x4). Questo processo seleziona il pixel con il valore massimo all'interno della finestra, presumibilmente quello con il contenuto informativo più rilevante.

Esistono anche **pooling layers addestrabili**, che hanno parametri che vengono aggiornati durante l'addestramento della rete. In questi casi, i kernel del pooling hanno dei pesi.

Sebbene esistano diverse tipologie di pooling, il **max pooling** è il più diffuso per la sua semplicità e interpretazione intuitiva.

##### Interpretazione del Max Pooling:

Il max pooling seleziona il pixel con il valore massimo all'interno di una finestra, presumibilmente quello con la caratteristica più prominente. Questo processo contribuisce a:

* **Ridurre la dimensione delle feature map:** semplificando la rete e riducendo il numero di parametri.
* **Rendere la rete più robusta alle variazioni di posizione:** poiché il max pooling seleziona la caratteristica più prominente, la sua posizione esatta diventa meno importante.

### Funzioni di Attivazione

Le **funzioni di attivazione** sono funzioni non lineari, tipicamente differenziabili, che vengono applicate dopo ogni strato convoluzionale.

##### Scopo delle Funzioni di Attivazione:

* **Introduzione di non-linearità:** le funzioni di attivazione permettono di modellare relazioni non lineari tra i dati, rendendo le reti neurali più potenti.
* **Miglioramento dell'addestramento:** le funzioni di attivazione possono limitare il range di valori in uscita, facilitando l'addestramento della rete.

##### Esempi di Funzioni di Attivazione:

* **ReLU (Rectified Linear Unit):** restituisce il massimo tra l'input e 0. È una funzione semplice e computazionalmente efficiente.
* **Funzione Logistica (Sigmoide):** comprime lo spazio di output tra 0 e 1. Viene utilizzata quando si necessita di un output che sia assimilabile a una probabilità.
* **Tangente Iperbolica (Tanh):** simile alla funzione logistica, ma comprime lo spazio di output tra -1 e 1.

##### ReLU:

La ReLU è una funzione di attivazione molto popolare per la sua semplicità e efficienza computazionale. La sua formula è:

```
ReLU(x) = max(0, x)
```

La ReLU introduce una soglia a 0, restituendo solo valori positivi. Questo aiuta a limitare il range di valori in uscita e a migliorare l'addestramento della rete.

##### Funzione Logistica:

La funzione logistica, o sigmoide, è una funzione che comprime lo spazio di output tra 0 e 1. La sua formula è:

```
Sigmoid(x) = 1 / (1 + exp(-x))
```

La funzione logistica è spesso utilizzata quando si necessita di un output che sia assimilabile a una probabilità.

##### Tangente Iperbolica:

La tangente iperbolica è una funzione simile alla funzione logistica, ma comprime lo spazio di output tra -1 e 1. La sua formula è:

```
Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

La tangente iperbolica è spesso utilizzata quando si desidera un output che sia centrato attorno a 0.

## Convoluzione e Parametri

### Schema Generale di una Rete Neurale Convoluzionale

Le reti neurali convoluzionali tipicamente seguono questo schema: convoluzione, attivazione, layer di pooling. Ognuno ha lo scopo di estrarre una *feature map* che rappresenta una qualche caratteristica, una proprietà. Combinando le *feature map*, quindi sui layer densi (layer lineari o *fully connected*), l'output sarà quello che noi interpreteremo come una funzione di probabilità.

### Convoluzione e Parametri

In questa lezione, analizzeremo il processo di convoluzione e i parametri che lo governano.

##### Esempio:

Immaginiamo di avere un'immagine e di voler applicare un *kernel* ad essa. Il risultato sarà una *feature map*. Il processo di convoluzione consiste nell'iterare su una finestra dell'immagine, applicando il *kernel* e calcolando l'output. Questo output viene memorizzato come elemento nella *feature map*. Il processo viene ripetuto su tutte le dimensioni dell'immagine.

##### Parametri:

Esistono alcuni parametri, o iperparametri, che determinano il modo in cui il *kernel* viene applicato all'immagine. Tra questi, troviamo il *padding*.

### Padding

Il *padding* è un'operazione che consiste nell'aggiungere dei valori, tipicamente zeri, ai bordi dell'immagine. Questo permette di applicare il *kernel* anche ai pixel che si trovano ai bordi dell'immagine, senza perdere informazioni.

**"No padding"**: in questo caso, il *kernel* viene applicato solo ai pixel che si trovano all'interno dell'immagine. Questo significa che i pixel ai bordi non vengono considerati.

**"Padding"**: in questo caso, vengono aggiunti dei valori di *padding* ai bordi dell'immagine. Questo permette di applicare il *kernel* anche ai pixel che si trovano ai bordi, considerando i valori di *padding* come se fossero pixel reali.

##### Esempio:

Consideriamo un *kernel* 3x3. Se non utilizziamo il *padding*, il *kernel* può essere applicato solo ai pixel che si trovano all'interno dell'immagine, perdendo informazioni sui pixel ai bordi. Utilizzando il *padding*, invece, possiamo applicare il *kernel* anche ai pixel ai bordi, considerando i valori di *padding* come se fossero pixel reali.

##### In sintesi:

Il *padding* è un'operazione che permette di applicare il *kernel* anche ai pixel ai bordi dell'immagine, senza perdere informazioni. Questo è importante per preservare le informazioni ai bordi dell'immagine e per evitare che la dimensione dell'immagine diminuisca dopo ogni convoluzione.

## Stride e Dilation

### Stride

Il parametro "Stride" determina di quanto la maschera si sposta durante le iterazioni di convoluzione. In pratica, indica il numero di pixel che si saltano tra un'applicazione del kernel e la successiva.

* **Stride = 1:** Il kernel viene applicato a blocchi di pixel adiacenti, con una sovrapposizione tra le applicazioni successive.
* **Stride > 1:** Il kernel viene applicato a blocchi di pixel non adiacenti, saltando un numero di pixel pari allo stride.
* **Stride = dimensione del kernel:** Il kernel viene applicato a patch distinte dell'immagine, senza sovrapposizioni.

### Dilation

La "Dilation" determina quali pixel vengono utilizzati nell'applicazione del kernel all'interno della maschera.

* **Dilation = 1:** Vengono considerati solo i pixel adiacenti al kernel.
* **Dilation > 1:** Vengono considerati pixel non adiacenti, saltando un numero di pixel pari alla dilation.

##### Vantaggi della Dilation:

* **Riduzione delle dimensioni dell'immagine:** La dilation permette di lavorare su un'immagine più piccola, riducendo il numero di operazioni necessarie.
* **Estrazione di feature su una superficie più ampia:** La dilation consente di calcolare le feature su una porzione più ampia dell'immagine, pur mantenendo un numero di operazioni inferiore.

##### Compromesso:

La dilation rappresenta un compromesso tra l'utilizzo di un kernel più grande (che cattura più informazioni) e la riduzione del numero di operazioni necessarie.

### Applicazioni

* **LeNet:** Nei primi livelli della rete LeNet, vengono utilizzati kernel più grandi per estrarre feature di alto livello su porzioni ampie dell'immagine.
* **VGG:** Anche nella VGG, i primi livelli utilizzano kernel più grandi per lo stesso scopo.
* **ResNet:** La dimensione del kernel è un fattore chiave nelle architetture ResNet.

### Principio della Dilation

Il principio della dilation si basa sull'idea che, per riconoscere un pattern, non è necessario considerare tutti i pixel. Ad esempio, per riconoscere una linea, è sufficiente considerare la maggior parte dei punti della linea.

##### In sintesi:

* Stride e Dilation sono parametri importanti nella convoluzione che influenzano il modo in cui il kernel viene applicato all'immagine.
* La scelta di questi parametri dipende dal tipo di feature che si desidera estrarre e dal compromesso tra accuratezza e complessità computazionale.

