# Notebook Training

Questo notebook si occupa del training del modello.

Ѐ stato eseguito su Kaggle, usando una singola GPU P-100.

Il modello di partenza è yolo11s, scelto per la sua versatilità e leggerezza.

### Installazione dei pacchetti necessari


```python
!pip install -U ultralytics
!pip install gdown
!pip install pyyaml
!pip install roboflow
!pip install --upgrade --force-reinstall sympy


```

### Creazione sottocartelle
Utilizzate per riprendere il training in caso di mancata persistenza dei file su kaggle


```python
import os

# Definizione del percorso della cartella principale
base_path = "/kaggle/working/CV-1"

# Definizione delle sottocartelle da creare
subfolders = ["training_run", "training_run/weights"]

# Creazione delle cartelle
for folder in subfolders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Cartella creata: {path}")

```

### Download dei pesi

Per gestire interruzioni improvvise causate dai limiti temporali imposti dalla piattaforma utilizzata (Kaggle) si è scelto di effettuare dei backup su Google Drive, per poter riprendere l'addestramento dall'ultima epoca calcolata senza perdere informazioni.


```python
import gdown


file_id = "1i4RHBe9NVhOQmRcKa4gImse2K321Z9Ul"
destination_path = "/kaggle/working/CV-1/training_run/weights/last.pt"

gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)

print(f"✅ File scaricato in {destination_path}")

```

# Elaborazione dataset

Si è effettuata una fase di pre-processing e data agumentation usando la piattaforma roboflow. 

I valori utilizzati sono i seguenti:
- **Flip**: Horizontal
- **Rotation**: Between -4° and +4°
- **Saturation**: Between -32% and +32%
- **Brightness**: Between -10% and +10%
- **Exposure**: Between -10% and +10%
- **Blur**: Up to 1.1px
- **Noise**: Up to 0.3% of pixels

Inoltre, il dataset viene scaricato direttamente usando la piattaforma.


```python

from roboflow import Roboflow
import shutil
import yaml
import zipfile
import os
from pathlib import Path

# Download del dataset da Roboflow

rf = Roboflow(api_key="est75tMpvx4WJGSIQN4K")
project = rf.workspace("computervision-p7hdm").project("cv-lnm6k")
version = project.version(1)
dataset = version.download("yolov11")


# Percorso del dataset scaricato
dataset_path = dataset.location  

# Definizione delle cartelle di destinazione
train_img_dir = os.path.join(dataset_path, 'images')
train_label_dir = os.path.join(dataset_path, 'labels')
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)

# Estrazione dei file ZIP se necessario
zip_path = os.path.join(dataset_path, "dataset.zip")
if os.path.exists(zip_path):
    print("Estrazione del dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)



# Creazione del file YAML per YOLOv11
data_yaml = {
    'train': train_img_dir,
    'val': "/kaggle/working/CV-1/valid/images",  
    'names': ["ball"]
    'nc': 1
}

with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f)

print("Dataset preparato con successo!")
print(f"Immagini salvate in: {train_img_dir}")
print(f"Etichette salvate in: {train_label_dir}")

```

# RIMUOVERE


```python
# Definisci il contenuto del file data.yaml

data_config = {
    "names": ["ball"],
    "nc": 1,
    "train": "/kaggle/working/CV-1/train/images",
    "val": "/kaggle/working/CV-1/valid/images"
}

# Percorso dove salvare il file data.yaml
yaml_path = "/kaggle/working/CV-1/data.yaml"

# Scrivi il contenuto nel file YAML
with open(yaml_path, "w") as file:
    yaml.dump(data_config, file, default_flow_style=False)

print(f"Il file {yaml_path} è stato aggiornato correttamente!")

```

## Pulizia GPU
Prima di avviare il training risulta utile terminare tutti i processi superflui, in modo da avere più memoria a disposizione.


```python
import torch
import gc
import os

def free_gpu():
    print("Liberazione memoria GPU in corso...")
    torch.cuda.empty_cache()  # Svuota la cache della GPU
    gc.collect()  # Forza il garbage collector a rilasciare la memoria

    # Se stai usando processi multipli (DDP), termina i processi zombie
    os.system("kill -9 $(pgrep -f 'python')")  # Forza la chiusura di processi Python
    print("Memoria GPU liberata con successo!")

free_gpu()

```

    Liberazione memoria GPU in corso...
    

# Training 

Si è scelto di usare come modello di partenza yolo11, in particolare la versione small(yolo11s).
Per ottimizzare la detection di un oggetto piccolo(palla), si è scelto di congelare una delle 3 teste d'attenzione di yolo, ovvero la P5. 
Questa testa corrisponde al livello 22, dettaglio ottenuto controllando il file yaml di yolo11. 
Il compito di questa testa era quello di elaborare oggetti più grandi durante la detection, compito ritenuto inutile per questo caso specifico.
Aver congelato il calcolo dei gradienti di questa testa d'attenzione ha permesso di ottenere una prestazione migliore, velocizzando il calcolo. Le risorse risparmiate sono state usate per aumentare la risoluzione delle immagini di addestramento a 1088 pixel.


```python
import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List
from ultralytics import YOLO

@dataclass
class TrainingConfig:
    """Configurazione ottimizzata per rilevamento single-class"""
    # Path specifici per Kaggle
    save_path: str = "/kaggle/working/CV-1"
    project_name: str = "training_run3"
    data_path: str = "/kaggle/working/CV-1/data.yaml"
    base_model: str = "yolo11s.pt"
    
    epochs: int = 150
    imgsz: int = 1088
    save_period: int = 10
    
    # Data augmentation minima (già preprocessato)
    hsv_h: float = 0.008
    hsv_s: float = 0.1
    hsv_v: float = 0.1
    fliplr: float = 0
    mosaic: float = 0.1
    mixup: float = 0.05
    
    # Ottimizzazione 
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Early stopping 
    patience: int = 15                
    close_mosaic: int = 0
    
    # Parametri specifici 
    conf: float = 0.35
    iou: float = 0.5
    max_det: int = 4
    nms: bool = True


class YOLOTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_path = Path(config.save_path) / config.project_name / "weights"
        self.device_setting = "cuda"
        self.batch_size = 16
        self.num_workers = 2
        
    
    
    def _find_last_checkpoint(self) -> tuple[str, bool]:
        try:
            if self.checkpoint_path.exists():
                checkpoints = sorted(self.checkpoint_path.glob("last.pt"), 
                                  key=lambda x: x.stat().st_mtime, 
                                  reverse=True)
                if checkpoints:
                    print(f"Riprendendo il training da: {checkpoints[0]}")
                    return str(checkpoints[0]), True
            
            print("Nessun checkpoint trovato, avvio nuovo training.")
            return self.config.base_model, False
            
        except Exception as e:
            print(f"Errore nel caricamento del checkpoint: {e}")
            print("Utilizzo del modello base come fallback.")
            return self.config.base_model, False
    
    def train(self):
        try:
            checkpoint, resume = self._find_last_checkpoint()
            model = YOLO(checkpoint)
            
            results = model.train(
                data=self.config.data_path,
                epochs=self.config.epochs,
                imgsz=self.config.imgsz,
                batch=self.batch_size,
                iou=self.config.iou,
                conf=self.config.conf,
                amp=True,
                augment=True,
                device=self.device_setting,  # Using the selected device (single GPU)
                project=self.config.save_path,
                name=self.config.project_name,
                freeze=[22],
                save_period=self.config.save_period,
                workers=self.num_workers,
                resume=resume,
                hsv_h=self.config.hsv_h,
                hsv_s=self.config.hsv_s,
                hsv_v=self.config.hsv_v,
                fliplr=self.config.fliplr,
                dropout=0.1,
                dfl= 2.0,
                mosaic=self.config.mosaic,
                mixup=self.config.mixup,
                optimizer=self.config.optimizer,
                lr0=self.config.lr0,
                lrf=self.config.lrf,
                max_det=self.config.max_det,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                warmup_epochs=self.config.warmup_epochs,
                warmup_momentum=self.config.warmup_momentum,
                warmup_bias_lr=self.config.warmup_bias_lr,
                patience=self.config.patience,
                close_mosaic=self.config.close_mosaic,
                half=True,
                tracker="bytetrack.yaml"
            )
            
            print("Training completato con successo!")
            return results
            
        except Exception as e:
            print(f"Errore durante il training: {e}")
            raise

def main():
    config = TrainingConfig()
    trainer = YOLOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

```

---
# aggiornamento training


```python
import yaml

# Definizione dei parametri
params = {
    "task": "detect",
    "mode": "train",
    "model": "/kaggle/working/CV-1/training_run/weights/last.pt",
    "data": "/kaggle/working/CV-1/data.yaml",
    "epochs": 150,
    "time": None,
    "patience": 25,
    "batch": 16,
    "imgsz": 1088,
    "save": True,
    "save_period": 5,
    "cache": False,
    "device": "cuda",
    "workers": 2,
    "project": "/kaggle/working/CV-1",
    "name": "training_run",
    "exist_ok": False,
    "pretrained": True,
    "optimizer": "SGD",
    "verbose": True,
    "seed": 0,
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 0,
    "resume": "/kaggle/working/CV-1/training_run/weights/last.pt",
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "freeze": [22],
    "multi_scale": False,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.1,
    "val": True,
    "split": "val",
    "save_json": False,
    "save_hybrid": False,
    "conf": 0.35,
    "iou": 0.5,
    "max_det": 4,
    "half": True,
    "dnn": False,
    "plots": True,
    "source": None,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": True,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "embed": None,
    "show": False,
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "line_width": None,
    "format": "torchscript",
    "keras": False,
    "optimize": False,
    "int8": False,
    "dynamic": False,
    "simplify": True,
    "opset": None,
    "workspace": None,
    "nms": False,
    "lr0": 0.0005,
    "lrf": 0.02,
    "momentum": 0.95,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.85,
    "warmup_bias_lr": 0.2,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 2.0,
    "pose": 12.0,
    "kobj": 1.0,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.2,
    "hsv_v": 0.2,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0,
    "bgr": 0.0,
    "mosaic": 0.2,
    "mixup": 0.1,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "crop_fraction": 1.0,
    "cfg": None,
    "tracker": "bytetrack.yaml",
    "save_dir": "/kaggle/working/CV-1/training_run"
}

# Percorso del file YAML
yaml_path = "/kaggle/working/CV-1/training_run/args.yaml"

# Scrittura del file YAML
with open(yaml_path, "w") as yaml_file:
    yaml.dump(params, yaml_file, default_flow_style=False)

print(f"File YAML aggiornato: {yaml_path}")

```


```python
import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, List
from ultralytics import YOLO

@dataclass
class TrainingConfig:
    """Configurazione per il fine tuning di un modello YOLO single-class"""
    # Path specifici
    save_path: str = "/kaggle/working/CV-1"
    project_name: str = "training_run"
    data_path: str = "/kaggle/working/CV-1/data.yaml"
    base_model: str = "yolo11s.pt"  # modello pre-addestrato

    # Parametri generali (default per training da zero)
    epochs: int = 150
    imgsz: int = 1088
    save_period: int = 10

    # Parametri per la data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.2
    hsv_v: float = 0.2
    fliplr: float = 0.1
    mosaic: float = 0.2
    mixup: float = 0.1

    # Parametri per l'ottimizzazione (default)
    optimizer: str = "SGD"
    lr0: float = 0.0005
    lrf: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    warmup_momentum: float = 0.85
    warmup_bias_lr: float = 0.2

    # Early stopping e altri parametri
    patience: int = 25                
    close_mosaic: int = 0

    # Parametri per il rilevamento
    conf: float = 0.35
    iou: float = 0.5
    max_det: int = 4
    nms: bool = True

    # Parametri specifici per il fine tuning
    finetune: bool = True                   # Abilita il fine tuning
    finetune_epochs: int = 50               # Meno epoche per fine tuning
    finetune_lr0: float = 0.0001            # Learning rate ridotto per fine tuning
    freeze_backbone: bool = True            # Congela il backbone nelle prime epoche
    freeze_layers: List[int] = field(default_factory=lambda: [i for i in range(22)])

class YOLOTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_path = Path(config.save_path) / config.project_name / "weights"
        self.device_setting = self._setup_device()
        self.batch_size = self._calculate_batch_size()
        self.num_workers = self._calculate_num_workers()
        
    def _setup_device(self) -> str:
        device_setting = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilizzo del dispositivo: {device_setting}")
        return device_setting
    
    def _calculate_batch_size(self) -> int:
        base_batch = 16
        return base_batch  
        
    def _calculate_num_workers(self) -> int:
        return max(2, os.cpu_count() // 2)
    
    def _find_last_checkpoint(self) -> tuple[str, bool]:
        try:
            if self.checkpoint_path.exists():
                checkpoints = sorted(self.checkpoint_path.glob("last.pt"), 
                                       key=lambda x: x.stat().st_mtime, 
                                       reverse=True)
                if checkpoints:
                    print(f"Riprendendo il training da: {checkpoints[0]}")
                    return str(checkpoints[0]), True
            
            print("Nessun checkpoint trovato, avvio nuovo training.")
            return self.config.base_model, False
            
        except Exception as e:
            print(f"Errore nel caricamento del checkpoint: {e}")
            print("Utilizzo del modello base come fallback.")
            return self.config.base_model, False

    def train(self):
        try:
            # Se il fine tuning è abilitato, si usano i parametri specifici
            if self.config.finetune:
                epochs = self.config.finetune_epochs
                lr0 = self.config.finetune_lr0
                freeze_layers = self.config.freeze_layers if self.config.freeze_backbone else []
                print(f"Modalità fine tuning abilitata: {epochs} epoche con lr0={lr0}")
            else:
                epochs = self.config.epochs
                lr0 = self.config.lr0
                freeze_layers = []  # addestramento da zero: nessun freeze
                print(f"Modalità training da zero: {epochs} epoche con lr0={lr0}")
            
            checkpoint, resume = self._find_last_checkpoint()
            model = YOLO(checkpoint)
            
            results = model.train(
                cfg="/kaggle/working/CV-1/training_run/args.yaml",
                data=self.config.data_path,
                epochs=epochs,
                imgsz=self.config.imgsz,
                batch=self.batch_size,
                iou=self.config.iou,
                conf=self.config.conf,
                amp=True,
                augment=True,
                device=self.device_setting,
                project=self.config.save_path,
                name=self.config.project_name,
                freeze=[22],
                save_period=self.config.save_period,
                workers=self.num_workers,
                resume=False,
                hsv_h=self.config.hsv_h,
                hsv_s=self.config.hsv_s,
                hsv_v=self.config.hsv_v,
                fliplr=self.config.fliplr,
                dropout=0.1,
                dfl=2.0,
                mosaic=self.config.mosaic,
                mixup=self.config.mixup,
                optimizer=self.config.optimizer,
                lr0=lr0,                      # Utilizzo del learning rate specifico per il fine tuning
                lrf=self.config.lrf,
                max_det=self.config.max_det,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                warmup_epochs=self.config.warmup_epochs,
                warmup_momentum=self.config.warmup_momentum,
                warmup_bias_lr=self.config.warmup_bias_lr,
                patience=self.config.patience,
                close_mosaic=self.config.close_mosaic,
                half=True,
                tracker="bytetrack.yaml"
            )
            
            print("Training completato con successo!")
            return results
            
        except Exception as e:
            print(f"Errore durante il training: {e}")
            raise

def main():
    config = TrainingConfig()
    trainer = YOLOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

```
