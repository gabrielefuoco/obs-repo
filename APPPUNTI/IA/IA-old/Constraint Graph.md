Un **Constraint Graph** è una ==rappresentazione grafica dei vincoli di un problema di soddisfacimento di vincoli ==(CSP). In questo grafo:

- I nodi (**V**) rappresentano le variabili.
- Gli archi (**E**) collegano le variabili che condividono un vincolo.

Quindi, ==se due variabili \(V_i\) e \(V_j\) appaiono nello stesso vincolo \(C_i\), c'è un arco tra di loro. ==Questo tipo di grafo funziona bene quando i vincoli coinvolgono solo **due variabili** (vincoli binari).

Ad esempio, se abbiamo vincoli come:

- \(C1(V1, V2, V3)\)
- \(C2(V1, V3, V4)\)
- \(C3(V4, V2)\)

il grafo collegherà tutte le variabili coinvolte in ciascun vincolo, ma con archi solo tra le coppie di variabili.

Tuttavia, ==se i vincoli coinvolgono **più di due variabili**, il **Constraint Graph** diventa meno efficiente. In questi casi, si utilizza un **ipergrafo**, dove un **iper-arco** collega **più di due nodi**. ==L'ipergrafo è più informativo perché mostra meglio come le variabili sono collegate dai vincoli complessi.