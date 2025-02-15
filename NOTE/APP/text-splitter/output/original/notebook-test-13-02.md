# Notebook di Test



# Utils


```python
!pip install ultralytics
!pip install pyyaml
!pip install gdown

```


```python
import os

# Definizione del percorso della cartella principale
base_path = "/kaggle/working"

# Definizione delle sottocartelle da creare
subfolders = ["Test", "Test/weights"]

# Creazione delle cartelle
for folder in subfolders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Cartella creata: {path}")

```

    Cartella creata: /kaggle/working/Test
    Cartella creata: /kaggle/working/Test/weights
    

# Download dei pesi e dei video di test

Per facilitare il test, si è scelto di caricare i pesi e i video di test su Google Drive. Di seguito il codice per scaricarli.

### Download dei pesi


```python
import gdown


file_id = "1Bd4Lwa3GExgKyaCob1pKzkTxhaEGGrmN"
destination_path = "/kaggle/working/CV-1/training_run/weights/best.pt"

gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)

print(f"✅ File scaricato in {destination_path}")

```

### Download Test Video


```python
import gdown

# ID-5
file_id = "1Fl5cKRCpQfggebGWQxmsDrMt557eXhsE"
destination_path = "/kaggle/working/CV-1/ID-5.json"

gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)
#------------------------------------------------------------------------------------------------------------#
# ID-6
file_id = "1oUYxLB9iy96Hd1smx65gwUbQfy2vV0zl"
destination_path = "/kaggle/working/CV-1/ID-6.json"

gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)

print(f"✅ File scaricati.")


```

# Configurazione del modello per la predizione

Si è optato per una soglia di confidenza leggermente più bassa poichè molte detection verranno corrette durante la fase di post processing.

Inoltre, è stato impostato un parametro per aumentare a 4 il numero massimo di detection possibile: durante la fase di post processing un metodo si occuperà di gestire le detection multiple, scegliendo non quelle con confidenza maggiore (come farebbe yolo), ma quelle che distano meno dall'ultima detection valida.



```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    conf: float
    iou: float
    classes: list
    max_det: int
    device: str
    stream: bool
    half: bool
    verbose: bool
    save: bool
    imgsz: (int,int)
    augment: bool=True

@dataclass
class ProcessingConfig:
    min_conf_threshold: float

def get_model_config(device: str) -> ModelConfig:
    return ModelConfig(
        conf=0.6,
        iou=0.5,
        imgsz=(1088,1920),
        classes=[0],
        max_det=4,
        device=device,
        stream=True,
        half=False,
        verbose=False,
        save=False
    )

def get_processing_config() -> ProcessingConfig:
    return ProcessingConfig(
        min_conf_threshold=0.6,
)
```

# Estrazione dei frame con OpenCV


```python
import cv2
import os

def open_video_capture(input_video):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video {input_video} not found")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError("Cannot open input video")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total_frames

def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError("Cannot create output video")
    return out

def release_resources(cap, out):
    if out is not None:
        out.release()
    if cap is not None:
        cap.release()
```

# Metodi per il disegno della traiettoria


```python
import cv2
import numpy as np

def draw_ball(frame, detection):
    if detection["x"] != -1:
        center = (int(detection["x"]), int(detection["y"]))
        # Se la detection è interpolata, usa il rosso (BGR: (0, 0, 255))
        if detection.get("interpolated", False):
            fill_color = (0, 0, 255)      # rosso pieno
            border_color = (0, 0, 255)    # rosso per il contorno
        else:
            fill_color = (0, 255, 0)      # verde pieno
            border_color = (255, 255, 255)  # bordo bianco
        cv2.circle(frame, center, 7, fill_color, -1)
        cv2.circle(frame, center, 8, border_color, 1)
    return frame

def draw_trajectory(frame, trajectory_points, interpolated=False):
    if len(trajectory_points) < 2:
        return  # Non disegnare nulla se non ci sono abbastanza punti

    # 'trajectory_points' è una lista di tuple (x, y)
    points = np.array(trajectory_points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))  # Rimodelliamo in (num_points, 1, 2)

    # Se la detection corrente è interpolata, la traiettoria sarà rossa, altrimenti verde
    color = (0, 0, 255) if interpolated else (0, 255, 0)
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

```

# Metodi per gestire la creazione del file Json


```python
import json

class DataHandler:
    def __init__(self):
        self.data = {}
        
    def add_frame(self, frame_number, detections):
        # Inizializziamo un dizionario vuoto per il frame
        frame_data = []
        
        for detection in detections:
            # Aggiungiamo i dati di ogni oggetto rilevato nel frame
            frame_data.append({
                "x": detection["x"],
                "y": detection["y"],
                "conf": detection["conf"],
                "interpolated": detection.get("interpolated", False)
            })
        
        # Salviamo tutte le detections per il frame corrente
        self.data[f"{frame_number:05d}.png"] = frame_data


    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4)
```

# Script per il test

Il seguente script si occupa di testare il modello.



```python
import os
import torch
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO


"""def clean_memory():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()"""

def process_video(input_video, output_path, model_path, json_path, device='cuda'):
    try:
        device = device if torch.cuda.is_available() else 'cpu'
        
        model = YOLO(model_path).to(device)
        """layers = list(model.model.model.children())
        p5_layer = layers[22]  
        for param in p5_layer.parameters():
            param.requires_grad = False
        model.model.model[22] = p5_layer
        model.fuse()"""
        
        cap, fps, width, height, total_frames = open_video_capture(input_video)
        out = create_video_writer(output_path, fps, width, height)
        
        model_config = get_model_config(device)
        
        data = DataHandler()
        det_count = 0
        
        pbar = tqdm(total=total_frames, desc='Processing video')
        
        with torch.inference_mode():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                
                # Processa il frame con YOLO
                results = model.predict(source=frame, **model_config.__dict__)
                
                for res in results:
                    proc_frame = res.orig_img.copy()
                    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    proc_detections = process_detection(res, model_config.conf)
                    
                    if len(res.boxes) > 1:
                        print(f"Frame {frame_num}: Multiple detections found ({len(res.boxes)})")
                    
                    if not proc_detections:
                        proc_detections.append({"x": -1, "y": -1, "conf": 0.0, "box": (-1, -1, -1, -1), "interpolated": False})
                    
                    det_count += len(proc_detections)
                    
                    out.write(proc_frame)
                    data.add_frame(frame_num, proc_detections)
                    pbar.update(1)
                
        print(f"✅ Completed. Detection rate: {det_count/total_frames*100:.1f}%")
        data.save(json_path)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise
    
    finally:
        release_resources(cap, out)
        #clean_memory()
        cv2.destroyAllWindows()
        if os.path.exists("runs"):
            shutil.rmtree("runs")

def process_detection(result, min_conf):
    detections = []
    
    if not result.boxes:
        return detections
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    if len(confs) == 0:
        return detections
    
    for i in range(len(confs)):
        if confs[i] >= min_conf:
            x1, y1, x2, y2 = boxes[i].astype(int)

            y1 *= (1080/1088) #Normalizzazione delle cooridinate y (imgsize deve essere multiplo di 32)
            y2 *= (1080/1088)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            det = {
                "x": float(cx),
                "y": float(cy),
                "conf": float(confs[i]),
                "box": (x1, y1, x2, y2),
                "interpolated": False
            }
            detections.append(det)
    
    return detections
```


```python
model_path = f'/kaggle/working/CV-1/training_run/weights/last.pt'

for i in range(5,7):
    input_video = f'/kaggle/input/test-set/ID-{i}.avi'
    output_path = f'/kaggle/working/CV-1/ID-{i}_detected.mp4'
    json_output_path = f'/kaggle/working/CV-1/ID-{i}_detected.json'
    
    process_video(input_video, output_path, model_path, json_output_path)
```

    YOLO11s summary (fused): 238 layers, 9,413,187 parameters, 0 gradients, 21.3 GFLOPs
    

    Processing video:  17%|█▋        | 523/3002 [00:46<03:41, 11.17it/s]

    Frame 520: Multiple detections found (2)
    Frame 522: Multiple detections found (2)
    

    Processing video:  17%|█▋        | 525/3002 [00:46<03:44, 11.02it/s]

    Frame 523: Multiple detections found (2)
    Frame 524: Multiple detections found (2)
    

    Processing video:  18%|█▊        | 531/3002 [00:47<03:43, 11.07it/s]

    Frame 528: Multiple detections found (2)
    Frame 529: Multiple detections found (2)
    Frame 530: Multiple detections found (2)
    

    Processing video:  19%|█▊        | 557/3002 [00:49<03:40, 11.11it/s]

    Frame 555: Multiple detections found (2)
    

    Processing video:  19%|█▉        | 577/3002 [00:51<03:37, 11.14it/s]

    Frame 574: Multiple detections found (2)
    

    Processing video:  20%|█▉        | 599/3002 [00:53<03:33, 11.23it/s]

    Frame 597: Multiple detections found (2)
    

    Processing video:  97%|█████████▋| 2909/3002 [04:17<00:08, 11.17it/s]

    Frame 2907: Multiple detections found (2)
    

    Processing video:  99%|█████████▊| 2959/3002 [04:21<00:03, 11.37it/s]

    Frame 2956: Multiple detections found (2)
    Frame 2957: Multiple detections found (2)
    Frame 2958: Multiple detections found (2)
    

    Processing video:  99%|█████████▊| 2961/3002 [04:21<00:03, 11.39it/s]

    Frame 2959: Multiple detections found (2)
    Frame 2960: Multiple detections found (2)
    Frame 2961: Multiple detections found (2)
    

    Processing video: 100%|█████████▉| 2999/3002 [04:25<00:00, 11.32it/s]

    ✅ Completed. Detection rate: 100.5%
    

    
    

# Post-Processing

Il post processing avviene applicando in pipeline al file json diversi metodi per la correzione degli outlier e per l'interpolazione dei buchi tra le detection. 
Altri metodi utili riguardano l'interpolazione delle traiettorie d'uscita della palla (punto debole del modello) e la validazione e smoothing delle interpolazioni effettuate in precedenza.

### Funzioni di Supporto


```python
import json
import os
import math
import numpy as np



def select_best_detection(detections, prev_det, next_det):
    """
    Seleziona la detection migliore (quella con il punteggio più basso, basato sulla distanza dai frame adiacenti)
    e, in caso di parità, quella con maggiore confidence.
    """
    min_score = float('inf')
    best_idx = 0
    for i, det in enumerate(detections):
        score = 0
        if prev_det:
            dx = det['x'] - prev_det['x']
            dy = det['y'] - prev_det['y']
            score += math.hypot(dx, dy)
        if next_det:
            dx = det['x'] - next_det['x']
            dy = det['y'] - next_det['y']
            score += math.hypot(dx, dy)
        if not prev_det and not next_det:
            score = 1 - det['conf']  # Se non esistono frame adiacenti validi, sceglie quella con confidence più alta
        if score < min_score or (score == min_score and det['conf'] > detections[best_idx]['conf']):
            min_score = score
            best_idx = i
    return best_idx

def find_previous_single_detection(sorted_keys, data, current_idx):
    """
    Cerca, a ritroso nell’array ordinato, il primo frame che contiene una sola detection.
    Restituisce la detection (singola) oppure None.
    """
    for i in range(current_idx - 1, -1, -1):
        frame_key = sorted_keys[i]
        detections = data[frame_key]
        if len(detections) == 1:
            return detections[0]
    return None

def find_next_single_detection(sorted_keys, data, current_idx):
    """
    Cerca, in avanti nell’array ordinato, il primo frame che contiene una sola detection.
    Restituisce la detection (singola) oppure None.
    """
    for i in range(current_idx + 1, len(sorted_keys)):
        frame_key = sorted_keys[i]
        detections = data[frame_key]
        if len(detections) == 1:
            return detections[0]
    return None

```

# Classe per Elaborazione in Pipeline

I metodi principali in questa classe sono:
- xxx


```python

class JsonProcessor:
    def __init__(self, input_file_path):
        self.input_file_path = input_file_path
        with open(input_file_path, 'r') as f:
            self.data = json.load(f)
        # Ordina le chiavi in base al numero contenuto nel nome
        self.sorted_keys = sorted(self.data.keys(), key=lambda x: int(x.split('.')[0]))

    def process_and_filter(self):
        """
        Rimozione detection multiple:
            Per ogni frame con più di una detection, seleziona quella migliore, ossia quella
            più vicina (in termini di distanza euclidea) ai frame adiacenti che contengono detection singola.
        """
        for idx, frame_key in enumerate(self.sorted_keys):
            detections = self.data[frame_key]
            if len(detections) <= 1:
                continue

            prev_det = find_previous_single_detection(self.sorted_keys, self.data, idx)
            next_det = find_next_single_detection(self.sorted_keys, self.data, idx)
            best_idx = select_best_detection(detections, prev_det, next_det)

            self.data[frame_key] = [detections[best_idx]]

    def correct_outliers_k(self, k, threshold=50):
        """
        Interpolazione/rimozione k outlier:
           Gestisce due casi:
           - Finestre di k outlier (tra due frame validi) → interpolazione lineare.
           - Finestre di k frame tra due frame invalidi (-1) → imposta tutti a -1.
        Le modifiche vengono effettuate in memoria.
        
        NOTA: Se uno dei frame intermedi è già un gap (coordinate -1), la finestra non viene trattata
        come outlier, in modo da lasciare la gestione dei gap alla funzione 'correct_k_gaps'.
        """
        outlier_windows = []
        invalid_windows = []
        n = len(self.sorted_keys)

        # Passata 1: rilevamento finestre
        for i in range(n - (k + 1)):
            start_frame = self.sorted_keys[i]
            end_frame = self.sorted_keys[i + k + 1]
            start_det = self.data[start_frame][0]
            end_det = self.data[end_frame][0]

            # Caso 1: entrambi validi
            if (start_det['x'] != -1 and start_det['y'] != -1 and
                end_det['x'] != -1 and end_det['y'] != -1):
                dx = (end_det['x'] - start_det['x']) / (k + 1)
                dy = (end_det['y'] - start_det['y']) / (k + 1)
                valid_outlier = True
                for j in range(1, k + 1):
                    current_det = self.data[self.sorted_keys[i + j]][0]
                    # Se uno dei frame intermedi è già un gap, non trattare la finestra come outlier.
                    if current_det['x'] == -1 and current_det['y'] == -1:
                        valid_outlier = False
                        break
                    expected_x = start_det['x'] + dx * j
                    expected_y = start_det['y'] + dy * j
                    distance = math.hypot(current_det['x'] - expected_x, current_det['y'] - expected_y)
                    if distance < threshold or current_det.get('interpolated', False):
                        valid_outlier = False
                        break
                if valid_outlier:
                    outlier_windows.append((i, i + k + 1))
            # Caso 2: entrambi invalidi (-1)
            elif (start_det['x'] == -1 and start_det['y'] == -1 and
                  end_det['x'] == -1 and end_det['y'] == -1):
                invalid_windows.append((i, i + k + 1))

        modified = False
        # Passata 2: correzione finestre outlier
        for start, end in outlier_windows:
            start_frame = self.sorted_keys[start]
            end_frame = self.sorted_keys[end]
            start_det = self.data[start_frame][0]
            end_det = self.data[end_frame][0]
            dx = (end_det['x'] - start_det['x']) / (k + 1)
            dy = (end_det['y'] - start_det['y']) / (k + 1)
            for j in range(1, k + 1):
                current_key = self.sorted_keys[start + j]
                current_det = self.data[current_key][0]
                new_x = start_det['x'] + dx * j
                new_y = start_det['y'] + dy * j
                current_det['x'] = new_x
                current_det['y'] = new_y
                current_det['interpolated'] = True
                modified = True

        # Correzione finestre invalide
        for start, end in invalid_windows:
            for j in range(start + 1, end):
                current_key = self.sorted_keys[j]
                current_det = self.data[current_key][0]
                # Se il frame non è già invalido, lo aggiorno e imposto il flag interpolated.
                if current_det['x'] != -1 or current_det['y'] != -1:
                    current_det['x'] = -1
                    current_det['y'] = -1
                    current_det['conf'] = 0
                    current_det['interpolated'] = True
                    modified = True


    def correct_k_gaps(self, k):
        """
        Interpolazione gap di lunghezza k:
           Interpola gap costituiti esattamente da k frame (tutti invalidi) tra due frame validi,
           utilizzando interpolazione lineare.
        """
        n = len(self.sorted_keys)
        modified = False
        i = 0
        while i < n - (k + 1):
            start_frame = self.sorted_keys[i]
            end_frame = self.sorted_keys[i + k + 1]
            start_det = self.data[start_frame][0]
            end_det = self.data[end_frame][0]

            # Se gli estremi non sono validi, salta
            if (start_det['x'] == -1 or start_det['y'] == -1 or
                end_det['x'] == -1 or end_det['y'] == -1):
                i += 1
                continue

            # Verifica che tutti i frame intermedi siano invalidi
            gap_found = True
            for j in range(i + 1, i + k + 1):
                intermediate_det = self.data[self.sorted_keys[j]][0]
                if intermediate_det['x'] != -1 or intermediate_det['y'] != -1:
                    gap_found = False
                    break
            if not gap_found:
                i += 1
                continue

            # Calcola i parametri per l'interpolazione lineare
            start_x = start_det['x']
            start_y = start_det['y']
            end_x = end_det['x']
            end_y = end_det['y']
            delta_x = (end_x - start_x) / (k + 1)
            delta_y = (end_y - start_y) / (k + 1)

            for m in range(1, k + 1):
                current_key = self.sorted_keys[i + m]
                current_det = self.data[current_key][0]
                new_x = start_x + delta_x * m
                new_y = start_y + delta_y * m
                current_det['x'] = new_x
                current_det['y'] = new_y
                current_det['interpolated'] = True
                modified = True
            i += k + 1  # Avanza oltre il gap processato


    
    def is_near_border(self, detection, border_distance, frame_width=None, frame_height=None):
        """
        Verifica se una detection è "vicina" a uno dei bordi del frame.
        Se frame_width e frame_height sono specificati, vengono controllati anche i bordi destro e inferiore.
        
        Parametri:
            detection: il dizionario della detection (con chiavi 'x' e 'y')
            border_distance: la distanza dal bordo sotto la quale il frame è considerato vicino al bordo
            frame_width: larghezza del frame (opzionale)
            frame_height: altezza del frame (opzionale)
            
        True se la detection è vicina a uno dei bordi, False altrimenti.
        """
        # Verifica che le coordinate siano valide
        if detection['x'] == -1 or detection['y'] == -1:
            return False
            
        # Controlla i bordi sinistro e superiore
        if detection['x'] <= border_distance or detection['y'] <= border_distance:
            return True
        # Se sono disponibili le dimensioni, controlla anche destro e inferiore
        if frame_width is not None and detection['x'] >= frame_width - border_distance:
            return True
        if frame_height is not None and detection['y'] >= frame_height - border_distance:
            return True
        return False
    

    def correct_exit_trajectories(self, border_distance=80, num_valid_neighbors=10, num_future_invalid=10, image_width=1920, image_height=1080):
        """
        Corregge le traiettorie di uscita interpolando con una parabola i frame invalidi (-1),
        se i frame precedenti sono validi e quelli successivi sono tutti -1.
        Parametri:
            border_distance: Distanza massima dai bordi per considerare un'uscita.
            num_valid_neighbors: Numero di frame precedenti validi richiesti.
            num_future_invalid: Numero di frame successivi che devono essere tutti -1.
        """
        n = len(self.sorted_keys)
        processed_exits = set()
        
        for i in range(n):
            frame_key = self.sorted_keys[i]
            detection = self.data[frame_key][0]
    
            if frame_key in processed_exits:
                continue
    
            if not self.is_near_border(detection, border_distance, image_width, image_height):
                continue
            
            valid_trail = []
            if i >= num_valid_neighbors:
                for j in range(i - num_valid_neighbors, i):
                    prev_key = self.sorted_keys[j]
                    prev_det = self.data[prev_key][0]
                    if prev_det['x'] != -1 and prev_det['y'] != -1:
                        valid_trail.append((prev_det['x'], prev_det['y']))
                    else:
                        break
            
            if len(valid_trail) < num_valid_neighbors:
                continue
    
            # Verifica la direzione del movimento
            dx = valid_trail[-1][0] - valid_trail[0][0]  # variazione in x
            dy = valid_trail[-1][1] - valid_trail[0][1]  # variazione in y
            
            # Determina quale bordo è stato raggiunto
            is_left_border = detection['x'] <= border_distance
            is_right_border = detection['x'] >= image_width - border_distance
            is_top_border = detection['y'] <= border_distance
            is_bottom_border = detection['y'] >= image_height - border_distance
            
            # Verifica che la direzione del movimento sia coerente con il bordo raggiunto
            valid_direction = False
            if is_left_border and dx < 0:  # si muove verso sinistra
                valid_direction = True
            elif is_right_border and dx > 0:  # si muove verso destra
                valid_direction = True
            elif is_top_border and dy < 0:  # si muove verso l'alto
                valid_direction = True
            elif is_bottom_border and dy > 0:  # si muove verso il basso
                valid_direction = True
                
            if not valid_direction:
                continue
    
            if i + num_future_invalid >= n:
                continue
            
            invalid_future = True
            future_invalid_count = 0
            for j in range(i + 1, min(i + num_future_invalid + 1, n)):
                future_key = self.sorted_keys[j]
                future_det = self.data[future_key][0]
                if future_det['x'] != -1 or future_det['y'] != -1:
                    invalid_future = False
                    break
                future_invalid_count += 1
            
            if not invalid_future:
                continue
    
            x_vals = [pt[0] for pt in valid_trail]
            y_vals = [pt[1] for pt in valid_trail]
    
            coeffs = np.polyfit(x_vals, y_vals, 2)
            a, b, c = coeffs
    
            reached_border = False
            for j in range(i + 1, i + future_invalid_count + 1):
                if reached_border:
                    break
                    
                future_key = self.sorted_keys[j]
                future_x_unclamped = x_vals[-1] + (j - i) * (x_vals[-1] - x_vals[-2])
                future_x = max(0, min(future_x_unclamped, image_width))
                
                future_y_unclamped = a * future_x ** 2 + b * future_x + c
                future_y = max(0, min(future_y_unclamped, image_height))
                
                # Controlla se ha raggiunto uno dei bordi
                if (future_x <= 0 or future_x >= image_width or 
                    future_y <= 0 or future_y >= image_height):
                    reached_border = True
                
                self.data[future_key][0]['x'] = future_x
                self.data[future_key][0]['y'] = future_y
                self.data[future_key][0]['interpolated'] = True
                processed_exits.add(future_key)
    
            i += future_invalid_count
    

    def validate_interpolated_trajectories(self, num_neighbors=5):
        """
        Valida e corregge le traiettorie interpolate confrontandole con i punti validi vicini.
        Per ogni sequenza di punti interpolati, verifica la coerenza con la traiettoria dei
        punti validi precedenti e successivi, aggiustando sia x che y.
    
        Param num_neighbors: Numero di punti validi da considerare prima e dopo la sequenza interpolata
        """
        n = len(self.sorted_keys)
        i = 0
        
        while i < n:
            frame_key = self.sorted_keys[i]
            detection = self.data[frame_key][0]
            
            # Se non è un punto interpolato o è invalido, passa al successivo
            if not detection.get('interpolated', False) or detection['x'] == -1 or detection['y'] == -1:
                i += 1
                continue
                
            # Trova l'inizio e la fine della sequenza interpolata
            start_interp = i
            while (i < n and self.data[self.sorted_keys[i]][0].get('interpolated', False) and 
                   self.data[self.sorted_keys[i]][0]['x'] != -1):
                i += 1
            end_interp = i
            
            # Raccogli i punti validi precedenti
            prev_points = []
            j = start_interp - 1
            count = 0
            while j >= 0 and count < num_neighbors:
                prev_det = self.data[self.sorted_keys[j]][0]
                if (not prev_det.get('interpolated', False) and 
                    prev_det['x'] != -1 and prev_det['y'] != -1):
                    prev_points.insert(0, (j - start_interp, prev_det['x'], prev_det['y']))
                    count += 1
                j -= 1
                
            # Raccogli i punti validi successivi
            next_points = []
            j = end_interp
            count = 0
            while j < n and count < num_neighbors:
                next_det = self.data[self.sorted_keys[j]][0]
                if (not next_det.get('interpolated', False) and 
                    next_det['x'] != -1 and next_det['y'] != -1):
                    next_points.append((j - start_interp, next_det['x'], next_det['y']))
                    count += 1
                j += 1
                
            # Se non abbiamo abbastanza punti validi, continua
            if len(prev_points) < num_neighbors or len(next_points) < num_neighbors:
                continue
                
            # Calcola le parabole usando i punti validi
            all_points = prev_points + next_points
            t_vals = [pt[0] for pt in all_points]  # Indici temporali relativi
            x_vals = [pt[1] for pt in all_points]  # Coordinate x
            y_vals = [pt[2] for pt in all_points]  # Coordinate y
            
            # Calcola i coefficienti per x(t) e y(t)
            x_coeffs = np.polyfit(t_vals, x_vals, 2)  # Parabola per x nel tempo
            y_coeffs = np.polyfit(t_vals, y_vals, 2)  # Parabola per y nel tempo
            ax, bx, cx = x_coeffs
            ay, by, cy = y_coeffs
            
            # Correggi i punti interpolati
            for j in range(start_interp, end_interp):
                current_key = self.sorted_keys[j]
                current_det = self.data[current_key][0]
                
                # Calcola il tempo relativo per questo punto
                t = j - start_interp
                
                # Calcola i punti sulle parabole
                x_expected = ax * t**2 + bx * t + cx
                y_expected = ay * t**2 + by * t + cy
                
                # Media con i punti esistenti
                current_det['x'] = (current_det['x'] + x_expected) / 2
                current_det['y'] = (current_det['y'] + y_expected) / 2

        
    def save(self, suffix='_final'):
        """
        Salva il JSON processato in un nuovo file.
        """
        base_path = os.path.splitext(self.input_file_path)[0]
        output_path = f"{base_path}{suffix}.json"
        
        output_dict = {}
        for key in self.sorted_keys:
            # Ogni frame ha una lista contenente una detection; estraiamo l'oggetto detection
            detection = self.data[key][0]
            output_dict[key] = detection
        
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
        #print(f"[DEBUG] File salvato: {output_path}")
        return output_path



```


```python
for i in range (5,7):
    input_file_path = f"/kaggle/working/CV-1/ID-{i}_detected.json"
    
    processor = JsonProcessor(input_file_path)
    # Pipeline di metodi da applicare
    
    processor.process_and_filter()
    processor.correct_outliers_k(k=4, threshold=87)
    processor.correct_exit_trajectories(border_distance=86, num_valid_neighbors=7, num_future_invalid=2)
    processor.correct_k_gaps(k=1)
    processor.correct_exit_trajectories(border_distance=93, num_valid_neighbors=8, num_future_invalid=16)
    processor.correct_k_gaps(k=4)
    processor.correct_outliers_k(k=5, threshold=120)
    processor.correct_k_gaps(k=9)
    processor.validate_interpolated_trajectories(num_neighbors=20)
    processor.correct_exit_trajectories(border_distance=90, num_valid_neighbors=19, num_future_invalid=6)
    processor.correct_exit_trajectories(border_distance=93, num_valid_neighbors=10, num_future_invalid=14)
    processor.correct_k_gaps(k=2)
    
    output_file = processor.save(suffix='_final')
    
    print("File salvato in:", output_file)


```

    File salvato in: /kaggle/working/CV-1/ID-5_detected_final.json
    

    <ipython-input-13-a30210dbd4ef>:342: RankWarning: Polyfit may be poorly conditioned
      coeffs = np.polyfit(x_vals, y_vals, 2)
    <ipython-input-13-a30210dbd4ef>:342: RankWarning: Polyfit may be poorly conditioned
      coeffs = np.polyfit(x_vals, y_vals, 2)
    

# MSE


```python
import os
import json
import numpy as np

def calculate_trajectory_mse(gt_path, pred_path):
    try:
        # Carica i file JSON
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r') as f:
            pred_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Errore nel caricamento dei file JSON: {e}")
        return None

    # Ottieni tutti i frame presenti nei due file
    all_frames = set(gt_data.keys()).union(set(pred_data.keys()))

    squared_errors = []
    valid_points = 0

    for frame in all_frames:
        gt_values = gt_data.get(frame, {"x": -1, "y": -1})
        pred_values = pred_data.get(frame, {"x": -1, "y": -1})

        # Controlla che i valori siano nel formato corretto
        if not isinstance(gt_values, dict) or not isinstance(pred_values, dict):
            continue

        gt_x = gt_values.get("x", -1)
        gt_y = gt_values.get("y", -1)
        pred_x = pred_values.get("x", -1)
        pred_y = pred_values.get("y", -1)

        # Ignora i frame con valori mancanti (-1, -1)
        #if gt_x == -1 or gt_y == -1 or pred_x == -1 or pred_y == -1:
            #continue

        
        squared_error = (gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2
        squared_errors.append(squared_error)
        valid_points += 1

    total_frames = len(all_frames)

    if valid_points == 0:
        print("⚠️ Nessun punto valido trovato per il calcolo dell'MSE!")
        return {"mse": float('nan'), "rmse": float('nan'), "valid_points": 0, "total_frames": total_frames}

    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return {
        "mse": mse,
        "rmse": rmse,
        "valid_points": valid_points,
        "total_frames": total_frames
    }
```


```python
pred="/kaggle/working/CV-1/ID-5_detected_final.json"
gt="/kaggle/working/CV-1/ID-5.json"
result=calculate_trajectory_mse(gt,pred)
print(result["mse"])
```

    85955.08688863015
    

# Tuning - MSE


```python
import random
import copy
import os
from multiprocessing import Pool, cpu_count
import functools
import uuid

def process_iteration(args, gt_file, pred_file, processor, methods):
    # Genera una sequenza casuale
    seq_length = random.randint(1, 12)
    sequence = []
    
    for _ in range(seq_length):
        method = random.choice(methods)
        params = []
        for param_def in method['params']:
            param_name, param_range = param_def
            params.append(random.choice(param_range))
        sequence.append((method['name'], params))

    # Clona il processor per evitare side effects
    temp_processor = copy.deepcopy(processor)
    try:
        for method_call in sequence:
            method_name, params = method_call
            method = getattr(temp_processor, method_name)
            method(*params)
    except Exception as e:
        return (None, float('inf'))

    # Genera un nome file unico per evitare conflitti
    processed_file = temp_processor.save(suffix=f'_processed_{uuid.uuid4().hex}')

    # Calcola l'MSE
    mse_result = calculate_trajectory_mse(gt_file, processed_file)
    
    # Estrai l'MSE
    if isinstance(mse_result, dict):
        mse = mse_result.get('mse', float('inf'))
    else:
        mse = mse_result

    os.remove(processed_file)
    return (sequence, mse)

def optimize_processing_sequence(gt_file, pred_file, processor, iterations=10000):
    methods = [
        {
            'name': 'correct_outliers_k',
            'params': [
                ('k', range(1, 11)),
                ('threshold', range(30, 121))
            ]
        },
        {
            'name': 'correct_k_gaps',
            'params': [('k', range(1, 11))]
        },
        {
            'name': 'correct_exit_trajectories',
            'params': [
                ('border_distance', range(60, 121)),
                ('num_valid_neighbors', range(1, 21)),
                ('num_future_invalid', range(1, 21))
            ]
        },
        {
            'name': 'validate_interpolated_trajectories',
            'params': [('num_neighbors', range(2, 21))]
        }
    ]

    # Prepara i parametri per il parallel processing
    worker = functools.partial(
        process_iteration,
        gt_file=gt_file,
        pred_file=pred_file,
        processor=processor,
        methods=methods
    )

    best_mse = float('inf')
    best_sequence = []

    # Usa tutti i core disponibili
    with Pool(processes=cpu_count()) as pool:
        results = pool.imap_unordered(worker, range(iterations))
        for seq, mse in results:
            if mse < best_mse and seq is not None:
                best_mse = mse
                best_sequence = seq

    # Formatta l'output
    formatted_sequence = []
    for method_call in best_sequence:
        method_name, params = method_call
        method = next(m for m in methods if m['name'] == method_name)
        param_names = [p[0] for p in method['params']]
        formatted_params = {name: value for name, value in zip(param_names, params)}
        formatted_sequence.append(
            f"processor.{method_name}({', '.join(f'{k}={v}' for k, v in formatted_params.items())})"
        )
    
    return formatted_sequence, best_mse
```


```python
gt_file = "/kaggle/working/CV-1/ID-5.json"
pred_file = "/kaggle/working/CV-1/ID-5_detected_final.json"
processor = JsonProcessor(input_file_path)

best_sequence, best_mse = optimize_processing_sequence(gt_file, pred_file, processor, iterations=1000)

# Stampa i risultati
print("Migliore MSE trovato:", best_mse)
print("Sequenza ottimale:")
for step in best_sequence:
    print(step)
```

# Draw Trajectory on Video

Questo script usa i metodi definiti prima per disegnare la traiettoria sul video. 
Per risultare più comprensibile, la traiettoria avrà una persistenza di 5 secondi prima di vernir cancellata.



```python
import json
import cv2
import os
import numpy as np
from tqdm import tqdm


def update_trajectory(trajectory, frame_num, detection, max_age):
    """
    Aggiunge il punto della detection alla traiettoria se valido.
    """
    if detection["x"] != -1:
        trajectory.append((frame_num, (int(detection["x"]), int(detection["y"]))))

def filter_trajectory(trajectory, current_frame, max_age):
    """
    Filtra la traiettoria mantenendo solo i punti che hanno un'età inferiore a max_age.
    Restituisce una lista di coordinate (x, y).
    """
    return [pt for f, pt in trajectory if (current_frame - f) <= max_age]

def draw_video_trajectory(input_video_path, output_video_path, json_file_path):
    try:
        # Apertura video e creazione del writer
        cap, fps, width, height, total_frames = open_video_capture(input_video_path)
        out = create_video_writer(output_video_path, fps, width, height)

        # Caricamento delle detections dal file JSON
        with open(json_file_path, "r") as f:
            detections = json.load(f)

        # Inizializzazione della traiettoria e contatore delle detections valide
        trajectory = []
        det_count = 0
        # Imposta la durata del tracciamento in frame (5 secondi)
        track_duration = fps * 5

        pbar = tqdm(total=total_frames, desc="Processing video")

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # La chiave per il frame corrente è "num_frame.png"
            key = f"{frame_index:05d}.png"
            detection = detections.get(key, {"x": -1, "y": -1})

            # Aggiorna la traiettoria se la detection è valida
            update_trajectory(trajectory, frame_index, detection, track_duration)
            if detection["x"] != -1 and detection["y"] != -1:
                det_count += 1

            # Disegna la palla sul frame corrente
            draw_ball(frame, detection)

            # Filtra la traiettoria per includere solo i punti recenti
            filtered_traj = filter_trajectory(trajectory, frame_index, track_duration)
            if len(filtered_traj) >= 2:
                draw_trajectory(frame, filtered_traj, interpolated=detection.get("interpolated", False))

            out.write(frame)
            frame_index += 1
            pbar.update(1)

        pbar.close()
        print(f"✅ Completed")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

    finally:
        release_resources(cap, out)
        cv2.destroyAllWindows()

```

    Processing video: 100%|█████████▉| 2999/3002 [00:44<00:00, 66.99it/s]
    

    ✅ Completed. Detection rate: 15.7%
    


```python
for i in range (5,7):
    input_video_path = f"/kaggle/input/test-set/ID-{i}.avi"
    output_video_path = f"/kaggle/working/CV-1/ID-{i}_processed.mp4"
    json_file_path = f"/kaggle/working/CV-1/ID-{i}_detected_final.json"
    draw_video_trajectory(input_video_path, output_video_path, json_file_path)
```
