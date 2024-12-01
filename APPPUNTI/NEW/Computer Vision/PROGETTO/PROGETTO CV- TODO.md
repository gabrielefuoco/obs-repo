Riferimento utile per il salvataggio e il caricamento dei modelli PyTorch: 
- [Link](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
* [[saving]] **Da formattare**

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
