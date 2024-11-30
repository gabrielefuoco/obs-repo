
Il progetto consiste nell'identificare e tracciare la posizione di un pallone da calcio in video.  Ciò richiede la costruzione di un modello di object detection/tracking che rilevi la posizione della palla frame per frame, producendo infine un video con la traccia della posizione della palla sovrapposta.

## Dataset

Il progetto utilizza due dataset:

* **Train:** 4 clip Full HD (1920x1080).
* **Test:** 2 clip Full HD (1920x1080).

Ogni clip è accompagnata da un file XML contenente le annotazioni sulla posizione della palla in ogni frame.  Alcuni frame potrebbero non contenere il pallone. Le annotazioni includono le coordinate della palla per ogni frame.  Le annotazioni sono strutturate come mostrato nell'esempio seguente:

```xml
<annotations>
…
<track id="0" label="ball" source="manual">
<points frame="115" outside="0" occluded="0" keyframe="1" points="7.02,790.87" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
<points frame="116" outside="0" occluded="0" keyframe="1" points="24.90,785.80" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
<points frame="117" outside="0" occluded="0" keyframe="1" points="41.90,780.70" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
…
</annotations>
```

Riferimento: *T. D’Orazio, M.Leo, N. Mosca, P.Spagnolo, P.L.Mazzeo. A Semi-Automatic System for Ground Truth Generation of Soccer Video Sequences. 6th IEEE International Conference on Advanced Video and Signal Surveillance, Genoa, Italy September 2-4 2009.*


## Obiettivi del Progetto

* Costruire un modello che rilevi la posizione della palla in ogni frame.
* Generare la traccia della palla nella clip (sequenza delle posizioni nei diversi frame).
* Produrre un video per le 2 clip di test che evidenzi la posizione della palla riconosciuta dal modello.
* La scelta del modello, della backbone e dei parametri è oggetto di valutazione.

Link al dataset: [https://drive.google.com/file/d/1IEt-TPw1pwtI6ieq1TYZgfzVpm0kkS5c/view?usp=drive_link](https://drive.google.com/file/d/1IEt-TPw1pwtI6ieq1TYZgfzVpm0kkS5c/view?usp=drive_link)


## Suggerimenti

* Utilizzo di piattaforme come Colab/Azure con salvataggio dello stato del modello.
* Test su porzioni del dataset per la stima degli iperparametri.
* Eventuale "arricchimento" del dataset di training.
* Personalizzazione del modello.
* Valutazione di modelli semplici vs. modelli complessi.
* Confronto tra loss standard e loss custom.
* Riferimento utile per il salvataggio e il caricamento dei modelli PyTorch: [https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
* [[saving]]


## Suggerimenti (2)

* Analisi approfondita del dataset: tipo di immagini, possibilità di aggiungere immagini simili.
* Scelta del modello: studio della letteratura, valutazione di diverse architetture e parametri.
* Implementazione di strategie di error-correction in fase di processing o post-processing.
* Definizione di metodi per la generazione della traccia della palla: confronto tra algoritmi di object tracking e object detection + post-processing.
* Analisi degli errori e tuning dei modelli tramite analisi della loss function.


## Requisiti di Consegna

* Definire e addestrare un modello di object detection/tracking.
* Presentazione delle scelte progettuali (modello, loss function, parametri utilizzati), analisi qualitativa delle clip di test annotate.
* Consegna del notebook (e/o file sorgente) per il training e il test, della presentazione e del dump del modello.
* Generazione di due file JSON (videoID_matricola.json), uno per ciascuna clip di test (clip 5 e 6), contenenti le posizioni rilevate della palla.



# PROMPT

Sei un esperto in computer vision e deep learning.  Ti viene richiesto di elaborare un piano dettagliato per un progetto di tracciamento di un pallone da calcio in video.  Il progetto prevede l'utilizzo di un modello di object detection/tracking per identificare la posizione del pallone frame per frame in video Full HD (1920x1080).  Sono disponibili due dataset: uno di training (4 clip) e uno di test (2 clip).  Ogni clip è annotata con file XML contenenti le coordinate (x, y) della palla per ogni frame; alcuni frame potrebbero non contenere la palla.  Un esempio di struttura XML delle annotazioni è il seguente:

```xml
<annotations>
…
<track id="0" label="ball" source="manual">
<points frame="115" outside="0" occluded="0" keyframe="1" points="7.02,790.87" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
<points frame="116" outside="0" occluded="0" keyframe="1" points="24.90,785.80" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
<points frame="117" outside="0" occluded="0" keyframe="1" points="41.90,780.70" z_order="0">
<attribute name="used_in_game">1</attribute>
</points>
…
</annotations>
```

Il progetto richiede:

1. **Scelta del modello:**  Descrivi dettagliatamente il modello di object detection/tracking proposto, giustificando la scelta in base alle caratteristiche del dataset e agli obiettivi del progetto.  Considera diverse architetture (YOLO, Faster R-CNN, DeepSORT, ecc.) e le relative prestazioni.  Specifica la backbone utilizzata.
2. **Strategia di training:** Descrivi il processo di training, inclusi i parametri di ottimizzazione (optimizer, learning rate, batch size, ecc.), la funzione di loss utilizzata (e la sua eventuale personalizzazione), e le tecniche di data augmentation impiegate.
3. **Strategia di tracking:** Spiega come verrà gestito il tracking del pallone tra i frame, considerando la possibilità di occlusioni o di assenza del pallone in alcuni frame.
4. **Valutazione delle prestazioni:** Descrivi le metriche utilizzate per valutare le prestazioni del modello sul dataset di test (precision, recall, F1-score, Mean Average Precision (mAP), ecc.).
5. **Generazione del video:** Spiega come verrà generato il video finale con la traccia della posizione del pallone sovrapposta.
6. **Gestione del progetto:** Descrivi l'ambiente di sviluppo (Colab, Azure, ecc.) e le strategie per la gestione del codice e dei dati.

Considera i seguenti punti:

* Test su sottoinsiemi del dataset per la tuning degli iperparametri.
* Eventuale arricchimento del dataset di training.
* Strategie di error correction.
* Confronto tra modelli semplici e complessi.
* Utilizzo di loss function standard o custom.

Il risultato finale deve includere un piano dettagliato e ben argomentato, pronto per l'implementazione.  

# TODO

**1. Scelta del Modello**

**1.1. Analisi Preliminare e Requisiti:**

*   **Dataset:** Abbiamo un dataset relativamente piccolo (4 clip di training, 2 di test), annotato con coordinate precise del pallone. 
*   **Obiettivo:** Tracciare la posizione del pallone in tempo reale in video Full HD. La precisione è importante, ma anche la velocità di inferenza è cruciale per un'applicazione di tracking video.
*   **Vincoli:** Dobbiamo gestire occlusioni e assenze temporanee del pallone.

**1.2. Modelli Candidati:**

*   **Object Detection:**
    *   **YOLOv8:** Veloce e preciso, adatto per applicazioni real-time. Ha diverse varianti (nano, small, medium, large, extra large) che permettono di bilanciare velocità e accuratezza.
    *   **Faster R-CNN:** Più lento di YOLO, ma potenzialmente più preciso, soprattutto in scenari complessi. Potrebbe essere utile come termine di paragone per valutare le prestazioni di YOLO.
    *   **SSD (Single Shot MultiBox Detector):** Un compromesso tra velocità e accuratezza, ma generalmente meno performante di YOLOv8.
*   **Object Tracking:**
    *   **DeepSORT:**  Un algoritmo di tracking robusto che combina un detector (come YOLO) con un algoritmo di associazione dati (Hungarian Algorithm) e una rete di re-identificazione (ReID) per gestire occlusioni e perdite temporanee del target.
    *   **Simple Online and Realtime Tracking (SORT):** Un algoritmo di tracking più semplice e veloce di DeepSORT, ma meno robusto in caso di occlusioni prolungate.

**1.3. Modello Proposto:**

Considerando i requisiti e le caratteristiche del dataset, propongo un approccio ibrido:

*   **Detector:** **YOLOv8 (versione medium o large)** come detector principale. Inizieremo con la versione medium per valutare la velocità di inferenza e passeremo alla versione large se necessario per migliorare l'accuratezza. La backbone sarà quella di default di YOLOv8 (CSPDarknet53 o un suo derivato).
*   **Tracker:** **DeepSORT**. DeepSORT offre un buon bilanciamento tra accuratezza e robustezza, ed è in grado di gestire occlusioni e perdite temporanee del target.

**Giustificazione:**

*   **YOLOv8** offre un ottimo compromesso tra velocità e accuratezza, fondamentale per il tracking in tempo reale. La sua architettura "single-stage" lo rende più veloce di Faster R-CNN, che è "two-stage".
*   **DeepSORT** è più robusto di SORT in caso di occlusioni, grazie alla rete di re-identificazione. Questo è importante nel contesto di una partita di calcio, dove il pallone può essere temporaneamente nascosto da giocatori o altri oggetti.
*   L'approccio ibrido ci permette di sfruttare i punti di forza di entrambi i modelli: la velocità di YOLOv8 per la detection e la robustezza di DeepSORT per il tracking.

**2. Strategia di Training**

**2.1. Preparazione del Dataset:**

*   **Divisione Train/Validation:** Divideremo ulteriormente il dataset di training in training (80%) e validation (20%) per la tuning degli iperparametri e la valutazione durante il training.
*   **Conversione Annotazioni:** Convertiremo le annotazioni XML in un formato compatibile con YOLOv8 (es. formato YOLO: `classe x_centro y_centro larghezza altezza`).
*   **Arricchimento del Dataset:** Arricchiremo il dataset con tecniche di data augmentation (vedi sotto) o, se possibile, annotando nuove clip video.
* **Utilizzare** [[ROBOFLOW]] per annotare nuove clip video

**2.2. Data Augmentation:**

Tecniche di data augmentation che applicheremo:

*   **Geometriche:**
    *   Random Resizing e Cropping
    *   Random Horizontal/Vertical Flip
    *   Rotazioni casuali (con attenzione a non alterare troppo l'aspetto del pallone)
*   **Fotometriche:**
    *   Variazioni di luminosità, contrasto, saturazione e hue
    *   Aggiunta di rumore gaussiano
*   **Specifiche per il Calcio:**
    *   Simulazione di motion blur (per rendere il modello più robusto ai movimenti veloci del pallone)

**2.3. Parametri di Ottimizzazione:**

*   **Optimizer:** AdamW (Adam con weight decay regularization) per una buona convergenza e generalizzazione.
*   **Learning Rate:** Inizieremo con un learning rate di 0.001 e utilizzeremo uno scheduler (es. ReduceLROnPlateau) per ridurlo dinamicamente se la loss non migliora.
*   **Batch Size:**  Un batch size di 16 o 32 dovrebbe essere un buon compromesso tra velocità di training e stabilità. Potrebbe essere necessario adattarlo in base alla memoria GPU disponibile.
*   **Epochs:**  Traineremo il modello per un numero sufficiente di epoche (es. 50-100), monitorando la loss sul validation set per evitare overfitting. Utilizzeremo early stopping per interrompere il training se la loss sul validation set non migliora per un certo numero di epoche consecutive.

**2.4. Funzione di Loss:**

*   **Loss di Detection (YOLOv8):** Utilizzeremo la loss di default di YOLOv8, che è una combinazione di:
    *   **Binary Cross-Entropy Loss** per la classificazione (presenza/assenza del pallone).
    *   **CIoU Loss** (Complete IoU) per la regressione delle bounding box (posizione e dimensioni).
*   **Loss di Tracking (DeepSORT):** DeepSORT non ha una loss function specifica, ma utilizza una metrica di distanza (Mahalanobis distance e cosine distance) per associare le detection ai track.

**2.5. Tuning degli Iperparametri:**

*   Utilizzeremo il validation set per la tuning degli iperparametri (learning rate, batch size, data augmentation, ecc.).
*   Proveremo diverse combinazioni di iperparametri e valuteremo le prestazioni utilizzando le metriche descritte nella sezione 4.
*   Potremmo utilizzare tecniche di ottimizzazione automatica degli iperparametri (es. Optuna, Hyperopt) per trovare la configurazione ottimale.

**3. Strategia di Tracking**

**3.1. Integrazione Detector-Tracker:**

*   Ogni frame del video verrà passato al detector YOLOv8 per ottenere le bounding box del pallone.
*   Le detection verranno passate al tracker DeepSORT, che si occuperà di:
    *   **Predizione:** Stimare la posizione del pallone nel frame successivo in base ai frame precedenti.
    *   **Associazione Dati:** Associare le detection correnti ai track esistenti utilizzando l'Hungarian Algorithm e le metriche di distanza (Mahalanobis e cosine).
    *   **Aggiornamento:** Aggiornare lo stato dei track (posizione, velocità, aspetto) in base alle detection associate.
    *   **Gestione di Nuovi Track:** Creare nuovi track per le detection non associate a track esistenti.
    *   **Gestione di Track Persi:** Eliminare i track che non vengono associati a detection per un certo numero di frame consecutivi.

**3.2. Gestione Occlusioni e Assenze:**

*   **Occlusioni:** DeepSORT è in grado di gestire occlusioni temporanee grazie alla rete di re-identificazione, che permette di riassociare un track anche dopo che il pallone è stato nascosto per alcuni frame.
*   **Assenze:** Se il pallone non viene rilevato per alcuni frame, il tracker continuerà a predire la sua posizione in base ai frame precedenti. Se l'assenza è prolungata, il track verrà eliminato.
*   **Interpolazione:**  Se il pallone è assente per un breve periodo, potremmo interpolare la sua posizione tra l'ultimo frame in cui è stato rilevato e il primo frame in cui viene rilevato di nuovo.

**3.3. Error Correction (Opzionale):**

*   **Vincoli Fisici:** Potremmo applicare dei vincoli fisici al movimento del pallone (es. velocità massima, accelerazione massima) per filtrare detection errate o predizioni implausibili.
*   **Smoothing:** Potremmo applicare un filtro di smoothing (es. Kalman filter, moving average) alle coordinate del pallone per ridurre il rumore e rendere la traiettoria più fluida.

**4. Valutazione delle Prestazioni**

**4.1. Metriche:**

*   **Object Detection:**
    *   **Precision:** Percentuale di detection corrette sul totale delle detection.
    *   **Recall:** Percentuale di detection corrette sul totale degli oggetti presenti.
    *   **F1-score:** Media armonica di precision e recall.
    *   **Mean Average Precision (mAP):** Media delle precisioni medie calcolate a diversi livelli di recall. Valuteremo mAP con diverse soglie di IoU (Intersection over Union) per valutare la precisione della localizzazione del pallone.
    *   **Frames Per Second (FPS):** Velocità di elaborazione dei frame, fondamentale per valutare le prestazioni in tempo reale.
*   **Object Tracking:**
    *   **Multi-Object Tracking Accuracy (MOTA):** Metrica che combina detection accuracy, false positives e false negatives.
    *   **Multi-Object Tracking Precision (MOTP):** Misura la precisione della localizzazione delle bounding box.
    *   **Mostly Tracked (MT):** Percentuale di track correttamente tracciati per almeno l'80% della loro durata.
    *   **Mostly Lost (ML):** Percentuale di track correttamente tracciati per meno del 20% della loro durata.
    *   **ID Switches (IDS):** Numero di volte in cui un track cambia identità.

**4.2. Valutazione sul Dataset di Test:**

*   Valuteremo le prestazioni del modello sul dataset di test *dopo* aver completato la fase di training e tuning degli iperparametri sul dataset di training/validation.
*   Calcoleremo le metriche sopra descritte per valutare sia le prestazioni del detector che del tracker.
*   Confronteremo le prestazioni del modello proposto con quelle di modelli più semplici (es. SORT) o più complessi (es. Faster R-CNN + DeepSORT) per valutare l'efficacia della nostra scelta.

**5. Generazione del Video**

**5.1. Sovrapposizione della Traccia:**

*   Dopo aver ottenuto le coordinate del pallone per ogni frame dal tracker, le sovrapporremo al video originale.
*   Disegneremo un cerchio o una bounding box attorno al pallone per evidenziarne la posizione.
*   Potremmo anche visualizzare l'ID del track e/o la velocità del pallone.
*   Utilizzeremo librerie come OpenCV o MoviePy per manipolare i video e sovrapporre gli elementi grafici.

**5.2. Salvataggio del Video:**

*   Salveremo il video finale con la traccia del pallone sovrapposta in un formato video standard (es. MP4).
*   Potremmo anche salvare un file di testo o CSV con le coordinate del pallone per ogni frame, per ulteriori analisi.

**6. Gestione del Progetto**

**6.1. Ambiente di Sviluppo:**

*   **Google Colab Pro:**  Offre GPU gratuite (o a pagamento con Colab Pro) per il training e l'inferenza, ed è facilmente accessibile.
*   **Alternative:** Se necessario, potremmo utilizzare servizi cloud come Azure Machine Learning o AWS SageMaker, che offrono maggiori risorse di calcolo e strumenti di gestione del progetto.

**6.2. Gestione del Codice:**

*   **Controllo Versione:** Utilizzeremo Git per il controllo versione del codice, con repository su GitHub o GitLab.
*   **Struttura del Codice:** Organizzeremo il codice in moduli separati per il detector, il tracker, la valutazione e la visualizzazione.
*   **Documentazione:** Documenteremo il codice con commenti e docstring, e creeremo un file README con istruzioni per l'installazione, l'esecuzione e la valutazione del progetto.

**6.3. Gestione dei Dati:**

*   **Archiviazione:** Memorizzeremo i dataset (video e annotazioni) su Google Drive o un servizio di cloud storage equivalente.
*   **Versionamento:** Se il dataset viene modificato o arricchito, terremo traccia delle diverse versioni.
*   **Script di Preprocessing:**  Automatizzeremo la conversione delle annotazioni e la preparazione del dataset con script Python.

**6.4. Workflow:**

1. **Setup:** Configurazione dell'ambiente di sviluppo, installazione delle librerie necessarie (PyTorch, YOLOv8, DeepSORT, OpenCV, ecc.).
2. **Preprocessing:** Conversione delle annotazioni, divisione del dataset, data augmentation.
3. **Training:** Training del modello YOLOv8, tuning degli iperparametri, valutazione sul validation set.
4. **Tracking:** Integrazione del detector YOLOv8 con il tracker DeepSORT, test su sottoinsiemi del dataset.
5. **Valutazione:** Valutazione delle prestazioni del sistema completo sul dataset di test.
6. **Generazione Video:** Creazione del video finale con la traccia del pallone sovrapposta.
7. **Documentazione:** Documentazione del progetto e dei risultati.

**7. Confronto tra Modelli e Strategie (Opzionale)**

*   **Modelli Semplici vs. Complessi:**  Confronteremo le prestazioni di YOLOv8 + DeepSORT con quelle di modelli più semplici (es. SORT) o più complessi (es. Faster R-CNN + DeepSORT) per valutare il trade-off tra velocità e accuratezza.
*   **Loss Function Standard vs. Custom:** Se necessario, potremmo sperimentare con loss function custom per migliorare le prestazioni del modello in scenari specifici (es. loss function che penalizza maggiormente gli errori di localizzazione in prossimità dei bordi del campo).

**8. Considerazioni Aggiuntive**

*   **Real-time Performance:**  Valuteremo attentamente le prestazioni in tempo reale del sistema, misurando gli FPS e ottimizzando il modello e il codice per garantire un tracking fluido.
*   **Robustezza a Condizioni Diverse:**  Se possibile, testeremo il sistema su video con condizioni di illuminazione, angolazioni di ripresa e qualità dell'immagine diverse per valutarne la robustezza.
*   **Generalizzazione:**  Se il sistema dovrà essere utilizzato su video provenienti da diverse fonti (es. diverse telecamere, diversi campionati), valuteremo la necessità di un dataset di training più ampio e diversificato.
