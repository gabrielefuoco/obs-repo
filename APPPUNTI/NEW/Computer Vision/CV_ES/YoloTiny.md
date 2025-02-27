# YOLO tiny

**TASK**: utilizzare la versione tiny di YOLO classificare il dataset <https://github.com/experiencor/kangaroo>
HINT per la fase di preprocesing del dataset vedi anche <https://towardsdatascience.com/implementing-yolo-on-a-custom-dataset-20101473ce53>

## Implementazione

Si può utilizzare una qualsiasi implementazione di YOLO-tiny pretrained. Ad esempio [qui](https://github.com/ultralytics/yolov3) è presente l'implementazione, una guida e come modificare la rete per il task di classificazione.

[Qui](https://github.com/ultralytics/yolov3/blob/master/tutorial.ipynb) è presente un tutorial su come eseguire il codice su Colab

I pesi del modello tiny richiedono circa 34MB

## Modifiche alla rete e training

La rete deve essere modificata con un numero di classi compatibile con quelle del dataset

[Info](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

## Valutazione dei risultati

Valutare i risultati in termini di mAP

