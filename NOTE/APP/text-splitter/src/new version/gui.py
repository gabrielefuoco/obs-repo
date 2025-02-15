import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
from typing import Dict, Tuple, List
import json
from datetime import datetime
from tkinterdnd2 import DND_FILES, TkinterDnD
import main  # Importa il modulo con la logica principale

class TooltipManager:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.tooltip = None
        self.widget = None
        self.id = None
        
    def show_tooltip(self, widget, text, event=None):
        def display():
            if self.tooltip or not widget.winfo_exists():
                return
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, justify='left',
                              background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()
            
        self.widget = widget
        self.id = widget.after(int(self.delay * 1000), display)
        
    def hide_tooltip(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class Application(TkinterDnD.Tk):
    """
    Application class represents the main GUI application for text processing with multiple prompts.
    Attributes:
        PADDING (dict): Padding configuration for widgets.
        INITIAL_SIZE (tuple): Initial size of the application window.
        CONFIG_FILE (str): Path to the configuration file.
    Methods:
        __init__(): Initializes the application.
        create_widgets(): Creates the main widgets of the application.
        create_folder_selection(): Creates the folder selection section.
        create_file_selection(): Creates the file selection section.
        create_prompt_selection(): Creates the prompt selection section.
        create_progress_indicator(): Creates the progress indicator.
        setup_drag_drop(): Configures drag and drop for files.
        handle_drop(event): Handles files dropped into the window.
        create_file_checkbox(name, var): Creates a checkbox for a file with preview.
        show_preview(file_path): Shows a preview of the file content.
        add_tooltip(widget, text): Adds a tooltip to a widget.
        toggle_all_files(state): Selects/deselects all files.
        toggle_all_prompts(state): Selects/deselects all prompts.
        filter_items(item_type): Filters files or prompts based on search.
        load_config(): Loads the saved configuration.
        save_config(): Saves the current configuration.
        update_progress(current, total, message=""): Updates the progress bar and message.
        run_processing(files, output, prompts, prompt_folder, split_method, order_mode, output_mode): Runs the processing in a separate thread.
        on_processing_success(): Callback for successful processing.
        on_processing_error(error): Callback for errors during processing.
        reset_ui(): Resets the user interface.
        on_close(): Handles window close event.
        create_processing_options(): Creates processing options.
        create_process_button(): Creates the process button.
        create_status_bar(): Creates the status bar.
        start_processing(): Starts the file processing.
        validate_inputs() -> bool: Validates inputs before processing.
        select_input_folder(): Selects the input folder and loads files automatically.
        select_output_folder(): Selects the output folder.
        select_prompt_folder(): Selects the prompt folder and loads prompts automatically.
        load_prompts(): Loads prompts from the selected folder.
        load_files(): Loads files from the input folder.
        update_status(message): Updates the status bar message.
    """
    PADDING = {'padx': 5, 'pady': 5}
    INITIAL_SIZE = (900, 750)
    CONFIG_FILE = 'app_config.json'
    
    def __init__(self):
        super().__init__()
        self.title("Elaboratore di Testi con Prompt Multipli")
        self.geometry(f"{self.INITIAL_SIZE[0]}x{self.INITIAL_SIZE[1]}")
        self.resizable(True, True)
        self.minsize(800, 600)  # Imposta dimensione minima
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Tooltip manager
        self.tooltip_manager = TooltipManager()
        
        # Variabili di stato
        self.processing_active = False
        self.all_prompts: Dict[str, str] = {}
        self.prompt_vars: Dict[str, tk.BooleanVar] = {}
        self.file_vars: Dict[str, Tuple[tk.BooleanVar, str]] = {}
        
        # Carica configurazione
        self.load_config()
        
        # Widget principali
        self.create_widgets()
        self.create_status_bar()
        self.setup_drag_drop()
        
        # Caricamento automatico all'avvio
        if Path(self.input_folder_var.get()).is_dir():
            self.load_files()
        if Path(self.prompt_folder_var.get()).is_dir():
            self.load_prompts()

    def create_widgets(self):
        self.create_folder_selection()
        self.create_file_selection()
        self.create_prompt_selection()
        self.create_processing_options()
        self.create_process_button()
        self.create_progress_indicator()

    def create_folder_selection(self):
        frame = ttk.LabelFrame(self, text="Selezione Cartelle")
        frame.pack(fill="x", **self.PADDING)
        frame.grid_columnconfigure(1, weight=1)  # Permette alle entry di espandersi

        # Cartella Input
        input_label = ttk.Label(frame, text="Cartella Input:")
        input_label.grid(row=0, column=0, sticky="w", **self.PADDING)
        self.input_folder_var = tk.StringVar(value=self.config.get('input_folder', ''))
        input_entry = ttk.Entry(frame, textvariable=self.input_folder_var, width=50)
        input_entry.grid(row=0, column=1, sticky="we", **self.PADDING)
        input_btn = ttk.Button(frame, text="Seleziona", command=self.select_input_folder)
        input_btn.grid(row=0, column=2, **self.PADDING)
        self.add_tooltip(input_label, "Seleziona la cartella contenente i file da elaborare")

        # Cartella Output
        output_label = ttk.Label(frame, text="Cartella Output:")
        output_label.grid(row=1, column=0, sticky="w", **self.PADDING)
        self.output_folder_var = tk.StringVar(value=self.config.get('output_folder', ''))
        output_entry = ttk.Entry(frame, textvariable=self.output_folder_var, width=50)
        output_entry.grid(row=1, column=1, sticky="we", **self.PADDING)
        output_btn = ttk.Button(frame, text="Seleziona", command=self.select_output_folder)
        output_btn.grid(row=1, column=2, **self.PADDING)
        self.add_tooltip(output_label, "Seleziona la cartella dove salvare i risultati")

        # Cartella Prompt
        prompt_label = ttk.Label(frame, text="Cartella Prompt:")
        prompt_label.grid(row=2, column=0, sticky="w", **self.PADDING)
        self.prompt_folder_var = tk.StringVar(value=self.config.get('prompt_folder', ''))
        prompt_entry = ttk.Entry(frame, textvariable=self.prompt_folder_var, width=50)
        prompt_entry.grid(row=2, column=1, sticky="we", **self.PADDING)
        prompt_btn = ttk.Button(frame, text="Seleziona", command=self.select_prompt_folder)
        prompt_btn.grid(row=2, column=2, **self.PADDING)
        self.add_tooltip(prompt_label, "Seleziona la cartella contenente i file prompt")

    def create_file_selection(self):
        frame = ttk.LabelFrame(self, text="Seleziona i File da Processare")
        frame.pack(fill="both", expand=True, **self.PADDING)
        
        # Campo di ricerca
        search_frame = ttk.Frame(frame)
        search_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(search_frame, text="Cerca File:").pack(side='left')
        self.file_search_var = tk.StringVar()
        self.file_search_var.trace('w', lambda *args: self.filter_items('files'))
        ttk.Entry(search_frame, textvariable=self.file_search_var).pack(side='left', fill='x', expand=True)
        
        # Pulsanti di selezione
        btn_frame = ttk.Frame(frame)
        ttk.Button(btn_frame, text="Seleziona Tutti", 
                  command=lambda: self.toggle_all_files(True)).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Deseleziona Tutti", 
                  command=lambda: self.toggle_all_files(False)).pack(side='left')
        btn_frame.pack(anchor='w', padx=5, pady=2)
        
        self.file_scroll_frame = ScrollableFrame(frame)
        self.file_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_prompt_selection(self):
        frame = ttk.LabelFrame(self, text="Seleziona i Prompt da Usare")
        frame.pack(fill="both", expand=True, **self.PADDING)
        
        # Campo di ricerca
        search_frame = ttk.Frame(frame)
        search_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(search_frame, text="Cerca Prompt:").pack(side='left')
        self.prompt_search_var = tk.StringVar()
        self.prompt_search_var.trace('w', lambda *args: self.filter_items('prompts'))
        ttk.Entry(search_frame, textvariable=self.prompt_search_var).pack(side='left', fill='x', expand=True)
        
        # Pulsanti di selezione
        btn_frame = ttk.Frame(frame)
        ttk.Button(btn_frame, text="Seleziona Tutti", 
                  command=lambda: self.toggle_all_prompts(True)).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Deseleziona Tutti", 
                  command=lambda: self.toggle_all_prompts(False)).pack(side='left')
        btn_frame.pack(anchor='w', padx=5, pady=2)
        
        self.prompt_scroll_frame = ScrollableFrame(frame)
        self.prompt_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_progress_indicator(self):
        """Crea gli indicatori di progresso"""
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x')
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(fill='x')

    def setup_drag_drop(self):
        """Configura il drag and drop per i file"""
        self.file_scroll_frame.scrollable_frame.drop_target_register(DND_FILES)
        self.file_scroll_frame.scrollable_frame.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        """Gestisce i file trascinati nella finestra"""
        files = event.data.split()
        for file in files:
            file_path = Path(file)
            if file_path.suffix.lower() in ['.md', '.txt']:
                var = tk.BooleanVar(value=True)
                name = file_path.name
                self.file_vars[name] = (var, str(file_path))
                self.create_file_checkbox(name, var)

    def create_file_checkbox(self, name, var):
        """Crea un checkbox per un file con preview"""
        frame = ttk.Frame(self.file_scroll_frame.scrollable_frame)
        
        chk = ttk.Checkbutton(frame, text=name, variable=var)
        chk.pack(side='left')
        
        preview_btn = ttk.Button(
            frame, 
            text="👁", 
            width=3,
            command=lambda: self.show_preview(self.file_vars[name][1])
        )
        preview_btn.pack(side='left', padx=2)
        
        frame.pack(anchor='w', fill='x')

    def show_preview(self, file_path):
        """Mostra l'anteprima del contenuto di un file"""
        try:
            preview_window = tk.Toplevel(self)
            preview_window.title("Anteprima")
            preview_window.geometry("600x400")
            
            text_widget = tk.Text(preview_window, wrap='word', width=60, height=20)
            text_widget.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Aggiungi scrollbar
            scrollbar = ttk.Scrollbar(preview_window, command=text_widget.yview)
            scrollbar.pack(side='right', fill='y')
            text_widget['yscrollcommand'] = scrollbar.set
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text_widget.insert('1.0', content)
            
            text_widget.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire il file:\n{str(e)}")

    def add_tooltip(self, widget, text):
        """Aggiunge un tooltip a un widget"""
        widget.bind('<Enter>', lambda e: self.tooltip_manager.show_tooltip(widget, text))
        widget.bind('<Leave>', lambda e: self.tooltip_manager.hide_tooltip())

    def toggle_all_files(self, state):
        """Seleziona/deseleziona tutti i file"""
        for var, _ in self.file_vars.values():
            var.set(state)

    def toggle_all_prompts(self, state):
        """Seleziona/deseleziona tutti i prompt"""
        for var in self.prompt_vars.values():
            var.set(state)

    def filter_items(self, item_type):
        """Filtra file o prompt in base alla ricerca"""
        search_text = self.file_search_var.get().lower() if item_type == 'files' else self.prompt_search_var.get().lower()
        items = self.file_vars if item_type == 'files' else self.prompt_vars
        frame = self.file_scroll_frame if item_type == 'files' else self.prompt_scroll_frame
        
        # Nascondi/mostra elementi in base alla ricerca
        for widget in frame.scrollable_frame.winfo_children():
            name = widget.winfo_children()[0]['text'].lower()
            if search_text in name:
                widget.pack(anchor='w', fill='x')
            else:
                widget.pack_forget()

    def load_config(self):
        """Carica la configurazione salvata"""
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def save_config(self):
        """Salva la configurazione corrente"""
        config = {
            'input_folder': self.input_folder_var.get(),
            'output_folder': self.output_folder_var.get(),
            'prompt_folder': self.prompt_folder_var.get()
        }
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def update_progress(self, current, total, message=""):
        """Aggiorna la barra di progresso e il messaggio"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.update_idletasks()

    def run_processing(self, files: List[str], output: str, prompts: List[str], 
                      prompt_folder: str, split_method: str, order_mode: str, output_mode: str):
        """Esegue l'elaborazione in un thread separato"""
        try:
            # Mappatura dei valori scelti nella GUI a quelli attesi dal modulo main
            if split_method == "paragrafi":
                split_method_mapped = "headers"
            else:
                split_method_mapped = split_method  # "frasi" o "intero" → NaturalTextSplitter

            if order_mode == "sequenziale":
                order_mode_mapped = "chunk"
            elif order_mode == "inverso":
                order_mode_mapped = "prompt"
            else:
                order_mode_mapped = order_mode

            if output_mode == "singolo":
                output_mode_mapped = "single"
            else:
                output_mode_mapped = output_mode

            total_files = len(files)
            # Carica tutti i prompt dalla cartella e filtra quelli selezionati
            prompt_folder_path = Path(prompt_folder)
            all_prompts = main.read_prompts(prompt_folder_path)
            selected_prompts_dict = {p: all_prompts[p] for p in prompts if p in all_prompts}
            output_folder_path = Path(output)
            main.ensure_folder(output_folder_path)

            # Processa i file uno per uno aggiornando la barra di progresso
            for idx, file in enumerate(files, start=1):
                self.after(0, self.update_progress, idx, total_files, f"Elaborazione per: {Path(file).name}")
                main.process_text_file(Path(file), selected_prompts_dict, output_folder_path, 
                                         split_method_mapped, order_mode_mapped, output_mode_mapped)
            self.after(0, self.on_processing_success)
            
        except Exception as e:
            self.after(0, self.on_processing_error, e)
        finally:
            self.after(0, self.reset_ui)

    def on_processing_success(self):
        """Callback per elaborazione completata con successo"""
        self.progress_var.set(100)
        self.progress_label.config(text="Elaborazione completata!")
        messagebox.showinfo("Completato", "Elaborazione completata con successo!")
        self.update_status("Pronto")

    def on_processing_error(self, error: Exception):
        """Callback per errori durante l'elaborazione"""
        self.progress_label.config(text="Errore durante l'elaborazione")
        messagebox.showerror("Errore", f"Si è verificato un errore:\n{str(error)}")
        self.update_status("Errore durante l'elaborazione")

    def reset_ui(self):
        """Ripristina l'interfaccia utente"""
        self.processing_active = False
        self.process_button.config(state=tk.NORMAL)
        self.save_config()  # Salva la configurazione al termine

    def on_close(self):
        """Gestisce la chiusura della finestra"""
        if self.processing_active:
            if messagebox.askokcancel("Uscita", "Elaborazione in corso! Uscire comunque?"):
                self.save_config()
                self.destroy()
        else:
            self.save_config()
            self.destroy()

    def create_processing_options(self):
        """Crea le opzioni di elaborazione"""
        frame = ttk.LabelFrame(self, text="Opzioni di Elaborazione")
        frame.pack(fill="x", **self.PADDING)
        frame.grid_columnconfigure(1, weight=1)  # Permette alle combo di espandersi
        
        # Metodo di divisione
        split_label = ttk.Label(frame, text="Metodo di Divisione:")
        split_label.grid(row=0, column=0, sticky="w", **self.PADDING)
        self.split_method = tk.StringVar(value="paragrafi")
        split_combo = ttk.Combobox(
            frame, 
            textvariable=self.split_method,
            values=["paragrafi", "frasi", "intero"],
            state="readonly"
        )
        split_combo.grid(row=0, column=1, sticky="we", **self.PADDING)
        self.add_tooltip(split_label, "Scegli come dividere il testo:\n- paragrafi: divide per paragrafi (MarkdownSplitter)\n- frasi: divide per frasi (NaturalTextSplitter)\n- intero: processa il testo intero (NaturalTextSplitter)")
        
        # Modalità di ordinamento
        order_label = ttk.Label(frame, text="Ordinamento:")
        order_label.grid(row=1, column=0, sticky="w", **self.PADDING)
        self.order_mode = tk.StringVar(value="sequenziale")
        order_combo = ttk.Combobox(
            frame,
            textvariable=self.order_mode,
            values=["sequenziale", "inverso"],
            state="readonly"
        )
        order_combo.grid(row=1, column=1, sticky="we", **self.PADDING)
        self.add_tooltip(order_label, "Ordine di elaborazione:\n- sequenziale: elabora chunk per chunk\n- inverso: elabora prompt per prompt")
        
        # Modalità di output
        output_label = ttk.Label(frame, text="Output:")
        output_label.grid(row=2, column=0, sticky="w", **self.PADDING)
        self.output_mode = tk.StringVar(value="singolo")
        output_combo = ttk.Combobox(
            frame,
            textvariable=self.output_mode,
            values=["singolo", "multiplo"],
            state="readonly"
        )
        output_combo.grid(row=2, column=1, sticky="we", **self.PADDING)
        self.add_tooltip(output_label, "Formato output:\n- singolo: un file per ogni input\n- multiplo: file separati per ogni prompt")

    def create_process_button(self):
        """Crea il pulsante di elaborazione"""
        self.process_button = ttk.Button(
            self,
            text="Avvia Elaborazione",
            command=self.start_processing
        )
        self.process_button.pack(**self.PADDING)

    def create_status_bar(self):
        """Crea la barra di stato"""
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_processing(self):
        """Avvia l'elaborazione dei file"""
        # Validazione input
        if not self.validate_inputs():
            return
        
        # Raccogli i file e i prompt selezionati
        selected_files = [
            path for name, (var, path) in self.file_vars.items()
            if var.get()
        ]
        
        selected_prompts = [
            name for name, var in self.prompt_vars.items()
            if var.get()
        ]
        
        if not selected_files:
            messagebox.showerror("Errore", "Seleziona almeno un file da elaborare")
            return
        
        if not selected_prompts:
            messagebox.showerror("Errore", "Seleziona almeno un prompt")
            return
        
        # Disabilita il pulsante durante l'elaborazione
        self.process_button.config(state=tk.DISABLED)
        self.processing_active = True
        self.update_status("Elaborazione in corso...")
        
        # Avvia l'elaborazione in un thread separato
        threading.Thread(
            target=self.run_processing,
            args=(
                selected_files,
                self.output_folder_var.get(),
                selected_prompts,
                self.prompt_folder_var.get(),
                self.split_method.get(),
                self.order_mode.get(),
                self.output_mode.get()
            ),
            daemon=True
        ).start()

    def validate_inputs(self) -> bool:
        """Valida gli input prima dell'elaborazione"""
        # Verifica cartelle
        input_folder = Path(self.input_folder_var.get())
        if not input_folder.is_dir():
            messagebox.showerror("Errore", "Cartella di input non valida")
            return False
        
        output_folder = Path(self.output_folder_var.get())
        if not output_folder.is_dir():
            messagebox.showerror("Errore", "Cartella di output non valida")
            return False
        
        prompt_folder = Path(self.prompt_folder_var.get())
        if not prompt_folder.is_dir():
            messagebox.showerror("Errore", "Cartella dei prompt non valida")
            return False
        
        # Verifica che le cartelle siano diverse
        folders = [input_folder, output_folder, prompt_folder]
        if len(set(folders)) != len(folders):
            messagebox.showerror("Errore", "Le cartelle devono essere diverse tra loro")
            return False
        
        return True

    def select_input_folder(self):
        """Seleziona la cartella di input e carica i file automaticamente"""
        if folder := filedialog.askdirectory(title="Seleziona Cartella Input"):
            self.input_folder_var.set(folder)
            self.load_files()

    def select_output_folder(self):
        """Seleziona la cartella di output"""
        if folder := filedialog.askdirectory(title="Seleziona Cartella Output"):
            self.output_folder_var.set(folder)

    def select_prompt_folder(self):
        """Seleziona la cartella dei prompt e li carica automaticamente"""
        if folder := filedialog.askdirectory(title="Seleziona Cartella Prompt"):
            self.prompt_folder_var.set(folder)
            self.load_prompts()

    def load_prompts(self):
        """Carica i prompt dalla cartella selezionata"""
        try:
            prompt_folder = Path(self.prompt_folder_var.get())
            if not prompt_folder.is_dir():
                raise ValueError("Cartella dei prompt non valida")
            
            # Pulisci i prompt esistenti
            for widget in self.prompt_scroll_frame.scrollable_frame.winfo_children():
                widget.destroy()
            self.prompt_vars.clear()
            
            # Carica i nuovi prompt
            self.all_prompts = main.read_prompts(prompt_folder)
            if not self.all_prompts:
                raise ValueError("Nessun prompt trovato")
            
            for prompt_file in self.all_prompts.keys():
                var = tk.BooleanVar(value=False)
                frame = ttk.Frame(self.prompt_scroll_frame.scrollable_frame)
                
                chk = ttk.Checkbutton(frame, text=prompt_file, variable=var)
                chk.pack(side='left')
                
                preview_btn = ttk.Button(
                    frame,
                    text="👁",
                    width=3,
                    command=lambda p=prompt_file: self.show_preview(
                        Path(self.prompt_folder_var.get()) / p
                    )
                )
                preview_btn.pack(side='left', padx=2)
                
                frame.pack(anchor='w', fill='x')
                self.prompt_vars[prompt_file] = var
            
            self.update_status(f"Caricati {len(self.all_prompts)} prompt")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Caricamento prompt fallito: {str(e)}")
            self.prompt_vars.clear()

    def load_files(self):
        """Carica i file dalla cartella di input"""
        try:
            input_folder = Path(self.input_folder_var.get())
            if not input_folder.is_dir():
                raise ValueError("Cartella di input non valida")
            
            txt_files = [
                f for f in input_folder.glob("*") 
                if f.suffix.lower() in ['.md', '.txt'] 
                and not f.name.startswith('.')
            ]
            
            if not txt_files:
                raise ValueError("Nessun file valido trovato")
            
            # Pulisci i checkbox esistenti
            for widget in self.file_scroll_frame.scrollable_frame.winfo_children():
                widget.destroy()
            self.file_vars.clear()
            
            # Crea i nuovi checkbox
            for file in txt_files:
                var = tk.BooleanVar(value=False)
                self.file_vars[file.name] = (var, str(file))
                self.create_file_checkbox(file.name, var)
            
            self.update_status(f"Caricati {len(txt_files)} file")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Caricamento file fallito: {str(e)}")
            self.file_vars.clear()

    def update_status(self, message: str):
        """Aggiorna il messaggio nella status bar"""
        self.status_var.set(message)
        self.update_idletasks()

if __name__ == "__main__":
    app = Application()
    app.mainloop()