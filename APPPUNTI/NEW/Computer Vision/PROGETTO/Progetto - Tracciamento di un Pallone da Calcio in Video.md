
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

