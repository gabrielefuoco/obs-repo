L'**Heuristic Alpha-Beta** è una variante dell'algoritmo Alpha-Beta in cui, per risparmiare tempo, si interrompe la ricerca prima di raggiungere una profondità massima o uno stato terminale. Quando ciò accade, invece di calcolare l'effettiva utilità del gioco, si usa una **valutazione euristica** che stima quanto sia favorevole lo stato attuale.

### Come funziona:
1. **Test di Taglio**: Quando si raggiunge una certa profondità o una condizione specifica, la ricerca si interrompe (non si prosegue fino agli stati finali).
   
2. **Valutazione euristica (EVAL)**: Viene usata una funzione di valutazione per stimare quanto è buona la posizione attuale, basandosi su elementi dello stato e sulla profondità raggiunta.

3. **Ordine delle mosse**: Se l'euristica è buona, possiamo ordinare le mosse in base alla loro valutazione per favorire il **pruning** (potatura) e ridurre il numero di nodi da esplorare.

L'euristica deve essere rapida da calcolare e deve riflettere le probabilità di vincere o perdere. Sebbene non sia perfetta, deve garantire che la stima sia realistica e che aiuti a prendere decisioni più rapidamente.