Normalize tf using document length

$$tf_i' = \frac{tf_i}{B}$$

$$c_i^{BM25}(tf_i) = \log{\frac{N}{df_i}} \times \frac{(k_1+1)tf_i'}{k_1+tf_i'}$$

$$= \log{\frac{N}{df_i}} \times \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$

BM25 ranking function

$$RSV^{BM25} = \sum_{i \in q} c_i^{BM25}(tf_i)$$

# BM25F with zones

Calculate a weighted variant of total term frequency, and
Calculate a weighted variant of document length

$$ \tilde{tf}_i = \sum_{z=1}^{Z} v_z tf_{zi} $$
$$ \tilde{dl} = \sum_{z=1}^{Z} v_z len_z $$
$$ avg \tilde{dl} = \text{average } \tilde{dl} \text{ across all docs} $$

where

* $v_z$ is zone weight
* $tf_{zi}$ is term frequency in zone $z$
* $len_z$ is length of zone $z$
* $Z$ is the number of zones

# Simple BM25F with zones

Simple interpretation: zone $z$ is "replicated" $v_z$ times

$$RSV_{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1 + 1)\tilde{tf_i}}{k_1((1-b) + b \frac{\tilde{dl_i}}{av\tilde{dl}}) + \tilde{tf_i}}$$

But we may want zone-specific parameters ($k_1$, $b$, IDF)

• Empirically, zone-specific length normalization (i.e., zone-specific $b_z$) has been found to be useful

$$ \tilde{tf}_i = \sum_{z=1}^Z v_z \frac{tf_{zi}}{B_z} $$

$$ B_z = \left( (1 - b_z) + b_z \frac{len_z}{avlen_z} \right), \quad 0 \le b_z \le 1 $$

$$ RSV_{BM2.5F} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1 + 1)\tilde{tf}_i}{k_1 + \tilde{tf}_i} $$

$$RSV^{BM25} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i}$$

Topic assignment to a word at position $i$ in doc $d_j$

Per-document topic distribution

Per-topic word distribution

For each word position in a doc of length $M$

For each doc in a collection of $N$ docs

Word token at position $i$ in doc $d_j$

[Moens and Vulic, Tutorial @WSD]

```
\begin{tikzpicture}[node distance=2cm]
\node[circle, draw] (D) {D};
\node[circle, draw, right of=D] (Z) {Z};
\node[circle, draw, right of=Z] (S) {S};
\node[circle, draw, below of=S] (V) {V};

\draw[->] (D) -- node[above] {$Pr(z|d)$} (Z);
\draw[->] (Z) -- node[above] {$Pr(s|z)$} (S);
\draw[->] (Z) -- node[left, midway] {$Pr(w|z)$} (V);
\draw[->] (S) -- node[right] {$Pr(w|s)$} (V);

\node[rectangle, draw, minimum width=3cm, minimum height=1.5cm, fit=(Z)(S)] {};
\node[rectangle, draw, minimum width=1.5cm, minimum height=1.5cm, fit=(V)] {};

\node[right of=S, node distance=0.5cm] {$|Sd|$};
\node[right of=V, node distance=0.5cm] {$|V|$};
\node[below right of=V, node distance=0.8cm] {$|D|$};
\end{tikzpicture}
```

[It's hot and delicious. [I poured the tea for my uncle]3.document

center word

$$\text{banking} = \begin{bmatrix} 0.286 \\ 0.792 \\ -0.177 \\ -0.107 \\ 0.109 \\ -0.542 \\ 0.349 \\ 0.271 \end{bmatrix}$$

$$\text{monetary} = \begin{bmatrix} 0.413 \\ 0.582 \\ -0.007 \\ 0.247 \\ 0.216 \\ -0.718 \\ 0.147 \\ 0.051 \end{bmatrix}$$

Example windows and process for computing $P(w_{t+j} | w_t)$

$$P(w_{t-2} | w_t)$ $P(w_{t+2} | w_t)$$

$$P(w_{t-1} | w_t)$ $P(w_{t+1} | w_t)$$

... problems turning into banking crises as ...

outside context words center word outside context words
in window of size 2 at position t in window of size 2

Example windows and process for computing $P(w_{t+j} | w_t)$

$$P(w_{t-2} | w_t)$ $P(w_{t+2} | w_t)$$

\begin{tikzpicture}[node distance=0.5cm]
\node[fill=pink, text width=1cm, align=center] (w1) {...};
\node[fill=pink, text width=1cm, align=center, right of=w1] (w2) {problems};
\node[fill=pink, text width=1cm, align=center, right of=w2] (w3) {turning};
\node[fill=pink, text width=1cm, align=center, right of=w3] (w4) {into};
\node[fill=red!70!black, text width=1cm, align=center, right of=w4] (w5) {banking};
\node[fill=pink, text width=1cm, align=center, right of=w5] (w6) {crises};
\node[fill=pink, text width=1cm, align=center, right of=w6] (w7) {as};
\node[fill=pink, text width=1cm, align=center, right of=w7] (w8) {...};

\draw[->] (w3) to[bend left] node[midway, above] {$P(w_{t-1} | w_t)$} (w5);
\draw[->] (w2) to[bend left=35] node[midway, above] {$P(w_{t-2} | w_t)$} (w5);
\draw[->] (w5) to[bend left] node[midway, above] {$P(w_{t+1} | w_t)$} (w7);
\draw[->] (w5) to[bend left=35] node[midway, above] {$P(w_{t+2} | w_t)$} (w8);
\draw (w2.south) -- ++(0,-0.2) -- ++(2,0) -- ++(0,0.2);
\node at (1.5,-0.7) {outside context words};
\node at (1.5,-0.9) {in window of size 2};
\draw (w5.south) -- ++(0,-0.2) -- ++(0.2,0) -- ++(0,0.2);
\node at (3.7,-0.7) {center word};
\node at (3.7,-0.9) {at position t};
\draw (w6.south) -- ++(0,-0.2) -- ++(2,0) -- ++(0,0.2);
\node at (5.5,-0.7) {outside context words};
\node at (5.5,-0.9) {in window of size 2};
\end{tikzpicture}

$$\theta = \begin{bmatrix} v_{aardvark} \\ v_a \\ \vdots \\ v_{zebra} \\ u_{aardvark} \\ u_a \\ \vdots \\ u_{zebra} \end{bmatrix} \in \mathbb{R}^{2dV}$$

# Source Text

The quick brown fox jumps over the lazy dog.

The quick brown fox jumps over the lazy dog.

The quick brown fox jumps over the lazy dog.

The quick brown fox jumps over the lazy dog.

# Training Samples

(the, quick)
(the, brown)
(quick, the)
(quick, brown)
(quick, fox)
(brown, the)
(brown, quick)
(brown, fox)
(brown, jumps)
(fox, quick)
(fox, brown)
(fox, jumps)
(fox, over)

# Input

$x_1$ 0
$x_2$ 0
...
$x_i$ 1
...
$x_V$ 0

# Hidden

$$
\begin{bmatrix}
h_1 \\
h_2 \\
h_3 \\
\vdots \\
h_N
\end{bmatrix}
= V \times
\begin{bmatrix}
w_1
\end{bmatrix}
$$

$N$-dimension vector

# Output

softmax

0 $y_1$
0 $y_2$
...
1 $y_j$
...
0 $y_V$

```
INPUT          PROJECTION          OUTPUT

the             $w_{(t-2)}$                      
cat             $w_{(t-1)}$              SUM     $w_{(t)}$     sat
on              $w_{(t+1)}$
floor           $w_{(t+2)}$

```

# Input layer

Index of cat in vocabulary
```
[0]
[1]
[0]
[0]
[0]
[0]
[0]
[0]
```
cat
one-hot
vector

```
[0]
[0]
[1]
[0]
[0]
[0]
[0]
[0]
```
on

# Hidden layer

# Output layer

```
[0]
[0]
[0]
[0]
[0]
[0]
[1]
[0]
```
sat
one-hot
vector

# We must learn $W$ and $W'$

# Input layer

```
[1]
[0]
[0]
[0]
[0]
[0]
[0]
[0]
```
cat

$$W_{V \times N}$$

V-dim

```
[0]
[0]
[1]
[0]
[0]
[0]
[0]
[0]
```
on

$$W_{V \times N}$$

V-dim

# Hidden layer

N-dim

N will be the size of word vector

# Output layer

```
[0]
[0]
[0]
[0]
[0]
[0]
[1]
[0]
```
sat

$$W'_{N \times V}$$

V-dim

```markdown
$W_{V \times N}^T$  $\times$ $x_{on}$ = $v_{on}$

| Input layer |  |  |
|---|---|---|
| 0 |  | 0 |
| $\color{red}{1}$ | $\times$ | 0 |
| 0 |  | 0 |
| 0 |  | 0 |
| 0 |  | $\color{red}{1}$ |
| 0 |  | 0 |
| 0 |  | 0 |
| ... |  | 0 |
| 0 |  | ... |

$x_{cat}$

V-dim

$\begin{bmatrix} 0.1 & 2.4 & 1.6 & \color{red}{1.8} & 0.5 & 0.9 & ... & 3.2 \\ 0.5 & 2.6 & 1.4 & \color{red}{2.9} & 1.5 & 3.6 & ... & 6.1 \\ ... & ... & ... & ... & ... & ... & ... & ... \\ 0.6 & 1.8 & 2.7 & \color{red}{1.9} & 2.4 & 2.0 & ... & 1.2 \end{bmatrix}$

$W_{V \times N}^T$ $\times$ $x_{cat}$ = $v_{cat}$

+

$\hat{v} = \frac{v_{cat} + v_{on}}{2}$

Hidden layer
N-dim

$W_{V \times N}^T$ $\times$ $x_{on}$ = $v_{on}$

| Output layer |  |
|---|---|
| 0 |  |
| 0 |  |
| 0 |  |
| 0 |  |
| 0 |  |
| 0 | sat |
| $\color{red}{1}$ |  |
| 0 |  |
| ... | V-dim |
| 0 |  |

```

Probability and Ratio
| | $k = solid$ | $k = gas$ | $k = water$ | $k = fashion$ |
|-------------|---------------|-------------|---------------|----------------|
| $P(k|ice)$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k|steam)$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| $\frac{P(k|ice)}{P(k|steam)}$ | $8.9$ | $8.5 \times 10^{-2}$ | $1.36$ | $0.96$ |

```
$\begin{bmatrix}
\begin{array}{c}
\text{FROM } [ ]_k \\
\text{GO poss } ( [ ]_j, \text{ TO } [ ]_i ) \\
\text{[EXCH [GO poss ([MONEY], TO } [ ]_k )]
\end{array}
\end{bmatrix}$
```

```markdown
EWN
Structure

Ontologia di dominio
Ontologia di alto livello

Road
Traffic          location

English WN
...
...
drive

Spanish WN
...
...
conducir

{drive}
Inter-Lingual-Index

Dutch WN
rijden
...
...

Italian WN
guidare
...
...
```

```
GET

Troponym

SELL  ←---Antonyms---> BUY ----> PAY
                                     | Entails doing
                                     |
                                     ---> CHOOSE
                                     | Entails doing

Troponym Troponym

TAKE OVER          PICK UP
```

**wood** [wʊd]
- n.
(material) legno; (timber)
(forest) bosco
(Golf) mazza di legno; (Bowls)
- adj.
(made of wood) di legno
(living etc. in a wood) di bosco, silvestre.

Gloss similarity

* Semantic field

sclerosis $n$ (Med) sclerosi

* Synonyms, hypernyms
reason 1. $n$. a. (motive, cause) ragione,...
sole $n$ (fish) sogliola

* Context
handle 1. $n$. ...
(of knife) manico, impugnatura;
(of door, drawer) maniglia

Shared hypernym and Synonym

albero 1. sm a. ( *pianta* ) tree

\{tree\} -- a tall perennial woody *plant* having a main trunk ...
\{tree, tree diagram\} -- a figure that branches from...

sogliola sf ( *pesce* ) sole

\{sole\} -- right-eyed flatfish; many are valued as food;
$\implies$ \{flatfish\} -- any of several families of *fishes* having...
\{sole\} -- the underside of the foot
$\implies$ \{area, region\} -- a part of an animal that has a special...

$$a:b :: c:?$$

$$d = \text{arg } \max_i \frac{(x_b - x_a + x_c)^T x_i}{||x_b - x_a + x_c||}$$

```
king
woman
man
```

| Word 1 | Word 2 | Human (mean) |
|--------------|------------|----------------|
| tiger | cat | 7.35 |
| tiger | tiger | 10 |
| book | paper | 7.46 |
| computer | internet | 7.58 |
| plane | car | 5.77 |
| professor | doctor | 6.62 |
| stock | phone | 1.62 |
| stock | CD | 1.31 |
| stock | jaguar | 0.92 |

| Model | Size | WS353 | MC | RG | SCWS | RW |
|---|---|---|---|---|---|---|
| SVD | 6B | 35.3 | 35.1 | 42.5 | 38.3 | 25.6 |
| SVD-S | 6B | 56.5 | 71.5 | 71.0 | 53.6 | 34.7 |
| SVD-L | 6B | 65.7 | 72.7 | 75.1 | 56.5 | 37.0 |
| CBOW\* | 6B | 57.2 | 65.6 | 68.2 | 57.0 | 32.5 |
| SG† | 6B | 62.8 | 65.2 | 69.7 | 58.1 | 37.2 |
| GloVe | 6B | 65.8 | 72.7 | 77.8 | 53.9 | 38.1 |
| SVD-L | 42B | 74.0 | 76.4 | 74.1 | 58.3 | 39.9 |
| GloVe | 42B | 75.9 | 83.6 | 82.9 | 59.6 | 47.8 |
| CBOW\* | 100B | 68.4 | 79.6 | 75.4 | 59.4 | 45.5 |

| Model | Dev | Test | ACE | MUC7 |
|---|---|---|---|---|
| Discrete | 91.0 | 85.4 | 77.4 | 73.4 |
| SVD | 90.8 | 85.7 | 77.3 | 73.7 |
| SVD-S | 91.0 | 85.5 | 77.6 | 74.3 |
| SVD-L | 90.5 | 84.8 | 73.6 | 71.5 |
| HPCA | 92.6 | $\mathbf{88.7}$ | 81.7 | 80.7 |
| HSMN | 90.5 | 85.7 | 78.7 | 74.7 |
| CW | 92.2 | 87.4 | 81.7 | 80.2 |
| CBOW | 93.1 | 88.2 | 82.2 | 81.1 |
| GloVe | $\mathbf{93.2}$ | $\mathbf{88.3}$ | $\mathbf{82.9}$ | $\mathbf{82.2}$ |

# Training with cross entropy loss

* Until now, our objective was stated as to maximize the probability of the correct class $y$ or equivalently we can minimize the negative log probability of that class

* Now restated in terms of cross entropy

* Let the true probability distribution be $p$; let our computed model probability be $q$

* The cross entropy is:
$$H(p, q) = - \sum_{c=1}^{C} p(c) \log q(c)$$

* Assuming a ground truth (or true or gold or target) probability distribution that is 1 at the right class and 0 everywhere else, $p = [0, ..., 0, 1, 0, ..., 0]$, then:

* Because of one-hot $p$, the only term left is the negative log probability of the true class $y_i$: $- \log p(y_i | x_i)$

