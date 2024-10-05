Il concetto di **utilità** si riferisce alla misura di quanto un certo risultato o stato sia desiderabile per un agente. In teoria, un agente razionale cerca di massimizzare la propria utilità. Tuttavia, la definizione di utilità non è sempre semplice e varia a seconda del contesto.
Dunque,==l'utilità rappresenta quanto un agente preferisce diversi risultati o combinazioni di risultati incerti, e la funzione di utilità permette di modellare queste preferenze in modo razionale e coerente.==

### Funzione di Utilità
Una ==**funzione di utilità** trasforma i possibili risultati (o stati del mondo) in numeri reali che rappresentano le preferenze di un agente==. Questo numero indica quanto l'agente preferisce un certo risultato rispetto ad altri.

### Lotterie e Scelte Incerte
Quando ci sono risultati incerti, si possono modellare le preferenze dell’agente con una **lotteria**: una combinazione di due o più risultati con probabilità associate. Ad esempio:
- **A** ha probabilità \(p\)
- **B** ha probabilità \(1 - p\)

L'agente deve scegliere tra risultati certi o lotterie. Per prendere decisioni razionali, le preferenze devono rispettare alcune regole, come la **transitività**, che garantisce coerenza nelle scelte. 

### Assiomi di Razionalità
1. **Transitività**: Se A è preferito a B, e B è preferito a C, allora A deve essere preferito a C. Senza questa proprietà, le scelte dell'agente potrebbero generare cicli infiniti di preferenze.

2. **Orderability**: L'agente deve poter decidere tra due opzioni se una è preferibile all'altra o se sono equivalenti (indifferenti).

3. **Continuità**: Se un premio A è preferito a B, che è preferito a C, esiste una probabilità che rende una lotteria tra A e C equivalente a B.

4. **Sostituibilità**: Se due premi sono indifferenti, anche le lotterie che li coinvolgono in modo analogo devono essere indifferenti.

5. **Monotonicità**: Se A è preferito a B, una lotteria che favorisce A (con una probabilità maggiore) è preferita a una lotteria che favorisce B.

### Teorema di von Neumann-Morgenstern
Se le preferenze dell’agente soddisfano questi assiomi, esiste una funzione di utilità che rappresenta le sue scelte in modo coerente. ==Questa funzione assegna un valore a ogni stato possibile, e le preferenze tra le lotterie si calcolano sommando le utilità dei risultati pesate dalle rispettive probabilità.==

### Principio della Massima Utilità Attesa (MEU)
Afferma che dovremmo scegliere l'opzione con il valore atteso più alto, ma gli esseri umani spesso non seguono questa logica. Ad esempio, preferiscono una somma sicura di denaro (come 400.000 dollari) rispetto a un'opzione con un valore atteso più alto ma rischiosa (come un lancio di moneta per 1 milione).

Le persone tendono a preferire la sicurezza per piccoli valori e accettano più rischi per somme più grandi, e la ricchezza personale influenza fortemente queste scelte. Questa ==differenza tra il valore atteso e la somma certa accettata è chiamata **equivalente certo**, ed è usata nelle assicurazioni.==

Gli esseri umani spesso prendono decisioni irrazionali, mostrando incoerenze nelle loro scelte rispetto a una funzione di utilità razionale.