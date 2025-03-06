Normalize tf using document length

$$tf_i' = \frac{tf_i}{B}$$

$$c_i^{BM25}(tf_i) = \log\frac{N}{df_i} \times \frac{(k_1+1)tf_i'}{k_1+tf_i'}$$
$$= \log\frac{N}{df_i} \times \frac{(k_1+1)tf_i}{k_1(1-b+b\frac{dl}{avdl})+tf_i}$$

BM25 ranking function

$$RSV^{BM25} = \sum_{i \in q} c_i^{BM25}(tf_i)$$

# BM25F with zones

Calculate a weighted variant of total term frequency, and
Calculate a weighted variant of document length

$$ \tilde{tf}_i = \sum_{z=1}^{Z} v_z tf_{zi} $$ $$ d\tilde{l} = \sum_{z=1}^{Z} v_z len_z $$ $$ avd\tilde{l} = \text{average } d\tilde{l} \text{ across all docs} $$

where

* $v_z$ is zone weight
* $tf_{zi}$ is term frequency in zone $z$
* $len_z$ is length of zone $z$
* $Z$ is the number of zones

# Simple BM25F with zones

Simple interpretation: zone z is “replicated” $v_z$ times

$$RSV^{SimpleBM25F} = \sum_{i \in q} log \frac{N}{df_i} \cdot \frac{(k_1 + 1)\tilde{tf}_i}{k_1((1-b)+b\frac{d\tilde{l}}{avd\tilde{l}}) + \tilde{tf}_i}$$

But we may want zone-specific parameters ($k_1, b, IDF$)

- Empirically, zone-specific length normalization (i.e., zone-specific $b$) has been found to be useful

$$ \tilde{tf_i} = \sum_{z=1}^Z v_z \frac{tf_{zi}}{B_z} $$

$$ B_z = \left((1-b_z) + b_z \frac{len_z}{avlen_z}\right), \quad 0 \le b_z \le 1 $$

$$ RSV^{BM25F} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1+1)\tilde{tf_i}}{k_1 + \tilde{tf_i}} $$

$$RSV^{BM25} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b\frac{dl}{avdl}) + tf_i}$$

```
banking = $\begin{bmatrix}
    0.286 \\
    0.792 \\
    -0.177 \\
    -0.107 \\
    0.109 \\
    -0.542 \\
    0.349 \\
    0.271
\end{bmatrix}$

monetary = $\begin{bmatrix}
    0.413 \\
    0.582 \\
    -0.007 \\
    0.247 \\
    0.216 \\
    -0.718 \\
    0.147 \\
    0.051
\end{bmatrix}$
```

Example windows and process for computing $P(w_{t+j} \mid w_t)$

$$P(w_{t-2} \mid w_t)$$

$$P(w_{t-1} \mid w_t)$ $P(w_{t+1} \mid w_t)$$

$$P(w_{t+2} \mid w_t)$$

... problems turning into banking crises as ...

|--------------------------------|-----------------|----------------------------------|
outside context words center word outside context words
in window of size 2 at position t in window of size 2

Example windows and process for computing $P(w_{t+j} \mid w_t)$

... problems turning into banking crises as ...
\ / \ /
\ / \ /
$$P(w_{t-2} \mid w_t)$ $P(w_{t+2} \mid w_t)$$
$$P(w_{t-1} \mid w_t)$ $P(w_{t+1} \mid w_t)$$

<------------> <-----> <------------>
outside context words center word outside context words
in window of size 2 at position t in window of size 2

$$
\theta =
\begin{bmatrix}
v_\text{aardvark} \\
v_a \\
\vdots \\
v_\text{zebra} \\
u_\text{aardvark} \\
u_a \\
\vdots \\
u_\text{zebra}
\end{bmatrix}
\in \mathbb{R}^{2dV}
$$

##### Source Text

The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.

##### Training Samples

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

##### Input

$$x_1$$ 0
$$x_2$$ 0
.
.
.
$$x_i$$ 1
.
.
.
$$x_V$$ 0

$$X$$
Embedding matrix
Vector of word i
$$w1$$
$$N$$

##### Hidden

$$h_1$$
$$h_2$$
$$h_3$$
.
.
.
$$h_N$$
N-dimension vector
$$V = $$
$$X$$

##### Output

softmax
$$y_1$$ 0
$$y_2$$ 0
.
.
.
$$y_j$$ 1
.
.
.
$$y_V$$ 0
$$N = $$
Context matrix
Vector of word j
$$w2$$
$$V$$

$$W_{V \times N}^T \times x_{on} = v_{on}$$
$$W_{V \times N}^T \times x_{cat} = v_{cat}$$

Input layer

$$x_{cat}$$
```
0
1
0
0
0
...
0
```
V-dim

$$x_{on}$$
```
0
0
1
0
0
...
0
```
V-dim

$$W_{V \times N}^T$$
```
0.1  2.4  1.6  1.8  0.5  0.9  ... 3.2
0.5  2.6  1.4  2.9  1.5  3.6  ... 6.1
... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ...
0.6  1.8  2.7  1.9  2.4  2.0  ... 1.2
```
$$\times$$

$$x_{on}$$
```
0
0
1
0
0
...
0
```

$$=$$

$$v_{on}$$
```
1.8
2.9
...
...
1.9
```

Output layer

```
0
0
0
0
0
1
...
0
```
sat

V-dim

Hidden layer
N-dim

$$\hat{v} = \frac{v_{cat} + v_{on}}{2}$$

| Probability and Ratio | $k=solid$ | $k=gas$ | $k=water$ | $k=fashion$ |
|-------------------------------|---------------|---------------|---------------|---------------|
| $P(k|ice)$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k|steam)$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| $P(k|ice)/P(k|steam)$ | $8.9$ | $8.5 \times 10^{-2}$ | $1.36$ | $0.96$ |

```
$$
\left[
\begin{array}{l}
\text{GO poss } ( [ \ ]_j,
\begin{bmatrix}
\text{FROM } [ \ ]_k \\
\text{TO } [ \ ]_i
\end{bmatrix}
) \\
\text{EXCH } [ \text{GO poss } ([\text{MONEY}],
\begin{bmatrix}
\text{FROM } [ \ ]_i \\
\text{TO } [ \ ]_k
\end{bmatrix}
) ]
\end{array}
\right]
$$
```

EWN
Structure

Ontologia di dominio Ontologia di alto livello

... Road
Traffic location ...

English WN Dutch WN

...
drive rijden
...
...
...
{drive}
Inter-Lingual-Index

Spanish WN Italian WN

...
conducir guidare
...
...
...

```
   GET
    ↑
   Troponym
    |
   BUY  ── Antonyms ──> SELL
    |   \
    |    Entails doing
    |     ↘
  Troponym  Troponym      PAY
    |     ↗
    |    Entails doing
    |   /
 TAKE OVER   PICK UP        CHOOSE
```

