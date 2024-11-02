# Un vero progetto di Machine Learning

In questa lezione, affronterai un progetto di esempio dall'inizio alla fine, come un vero data scientist.

Ecco una panoramica di tutti i passaggi che affronterai:

1. Definire il problema e guardare il quadro generale
2. Ottenere i dati
3. Scoprire e visualizzare i dati per ottenere informazioni
4. Preparare i dati per gli algoritmi di Machine Learning
5. Selezionare un modello e addestrarlo
6. Affinare il tuo modello.
7. Presentare la tua soluzione.
8. Lanciare, monitorare e mantenere il tuo sistema.

## Guarda il quadro generale e definisci il problema

Il dataset con cui lavoreremo è un dataset molto famoso, ovvero il dataset _California Housing Prices_.

Contiene informazioni sulle case, situate in un determinato distretto della California. 
Si basa su dati raccolti da un censimento californiano del 1990.

Si prega di notare che il dataset deve essere _pulito_. 
Sono necessari alcuni passaggi di pre-elaborazione per renderlo adatto a risolvere un compito di Data Mining.

*Attributi*

Gli attributi del dataset sono piuttosto autoesplicativi.

* longitudine
* latitudine
* housing_median_age
* total_rooms
* total_bedrooms
* popolazione
* famiglie
* median_income
* median_house_value
* ocean_proximity


Nelle prime fasi del tuo progetto, devi capire a quale tipo di problema ti stai avvicinando.

In questo contesto, sei tenuto a costruire un modello in grado di approssimare il prezzo delle case
in un distretto della California.

Più specificamente, desideri essere in grado di prevedere il prezzo associato a qualsiasi dato
_blocco_.

Un blocco è una piccola unità geografica generale.

Il modello che andrai ad addestrare sarà in grado di prevedere il prezzo medio delle case di qualsiasi _blocco_
appartenente al distretto a cui si riferiscono questi dati.

Una volta stabilito l'obiettivo generale, puoi affrontare i dettagli del problema in questione.

---

In uno scenario reale, devi capire come il tuo modello è destinato ad essere utilizzato.

Ad esempio, il tuo modello potrebbe essere parte di un pipeline più ampio o un progetto autonomo.


Ad esempio, supponiamo che il modello sia eventualmente incorporato nel seguente pipeline.

![](img/pipeline.png)


---

**Domanda**

Pensa alla natura del problema che stai per risolvere. 
Qual è il modo migliore per modellare questo problema?

*Risposta*: ....


----

Una volta che hai deciso come affrontare il problema, devi trovare un modo adatto per misurare le prestazioni del tuo modello.

In questo contesto, le metriche più comunemente utilizzate sono: le opzioni sono:

> **Errore quadratico medio radice**
    $$
        RMSE(X,h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2}
    $$
    
> **Errore assoluto medio**
    $$
        MAE(X,h) = \frac{1}{m} \sum_{i=1}^{m} | h(x^{(i)}) - y^{(i)}|
    $$
    
Entrambe le definizioni indicano un modo per misurare la distanza tra vettori.
Più specificamente, RMSE corrisponde alla *distanza euclidea* ($l_2$ norma)
mentre MAE corrisponde alla *norma di Manhattan* ($l_1$ distanza).

In generale, le norme più elevate sono più sensibili agli outlier.
Infatti, RMSE è più sensibile agli outlier se confrontato con MAE.

Più in generale, la norma $l_k$ di un vettore $v$ è definita come
$$
|| v_k|| = (\sum_i v_i^k)^{\frac{1}{k}}
$$

Quando $k$ aumenta, si concentra di più sui valori grandi.

Questo è uno dei motivi principali per cui utilizziamo tecniche come _scaling_ e _standardizzazione_ delle funzionalità per
migliorare le prestazioni di un algoritmo di addestramento.

Ad esempio, RMSE funziona meglio quando 
i dati seguono una curva a forma di campana.





# Setup


```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "img")

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
```

# Get the data


```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def maybe_fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # check wether the dataset is already
    # in the filesytesm
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    # make the request to download the compressed file    
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    # open and decompress the file
    housing_tgz = tarfile.open(tgz_path) 
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```


```python
maybe_fetch_housing_data()
```


```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```


```python
housing = load_housing_data()
```

## Take a Quick Look at the Data Structure



```python
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    

Some observations:

1.  ``total_bedrooms`` contains some missing values.

2. All the attributes but ``ocean_proximity`` are numerical (as suggested by the ``object`` dtype).


In fact, this feature contains text.


```python
housing["ocean_proximity"].value_counts()
```




    ocean_proximity
    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: count, dtype: int64



The ``describe`` function provides useful information about the data distribution.


```python
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



Clearly, plots are more appealing.


```python
# plot some graphs useful for visualizing the distribution of
# the values wrt to each individual feature
%matplotlib inline
import matplotlib.pyplot as plt


housing.hist(bins=50, figsize=(20,15))
#save_fig("attribute_histogram_plots")
plt.show()
```


    
![png](output_16_0.png)
    


You can draw some observations

1. ``median_income`` - draw your considerations 
2. ``median_house_age`` and ``median_house_value``  - draw your consideration

3. Look at the scales - draw your considerations 
4. Look at the shape of these distributions. Remember, RMSE works better with a bell-shaped curve.

## Create a  Test Set

It might seem weird to put aside part of the data already at this early stage.

However, it is always a good practice to put aside some of  the data point, so  to prevent a phenomena known as _data snooping bias_. 

The idea of letting the data drive the entire process is tempting, however this approach may be counter-productive.

For this reason,  you should always have a portion of the data that your algorithm has never seen before.


```python
import numpy as np
# your code here
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```


```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
```

    16512 train + 4128 test
    

Instead of writing the function yourself, you can rely  ``sklearn``.

It comes with  a useful function for splitting your data into train and test set.


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set.shape, test_set.shape
```




    ((16512, 10), (4128, 10))



Splitting the data is a delicate and crucial task. 

The main thing you want to avoid is to introduce: _sampling bias_.
Also, you want the data contained in the test set to be a 
__meaningful__ view on the original dataset.


**Example**
When a survey
company decides to call 1,000 people to ask them a few questions, they don’t just pick
1,000 people randomly in a phone booth. 

They try to ensure that these 1,000 people are representative of the whole population. 

For example, the US population is composed of 51.3% female and 48.7% male, so a well-conducted survey in the US would
try to maintain this ratio in the sample: 513 female and 487 male. 

This strategy is called __stratified sampling__ (more [here](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)). 

In order to do stratified sampling, the whole dataset is first divided into groups, i.e., _strata_.

The sampling procedure then must guarantee that each _stratum_ has a sufficient number of sample.
Where sufficient means that there are enough data to characterize the elements belonging to that specific category.


In the scenario depicted above, a poor sampling strategy has the 12% chance of providing a strongly unbalanced, thus
not meaningful, test set. 

__Note__: ``unbalanced does not imply not-meaningful


Clearly, if we introduce bias at this stage, we compromise all the training process.

---

What does it mean for a test set to be meaningful in our California housing problem?

A meaningful test set should guarantee a sufficient number of samples for every category of households income.
(How rich people in a certain neighborhood are?)

In order to understand if the test set is well constructed, you may want to compare train set distribution with
the one of the test set.




```python
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20046</th>
      <td>-119.01</td>
      <td>36.06</td>
      <td>25.0</td>
      <td>1505.0</td>
      <td>NaN</td>
      <td>1392.0</td>
      <td>359.0</td>
      <td>1.6812</td>
      <td>47700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3024</th>
      <td>-119.46</td>
      <td>35.14</td>
      <td>30.0</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>1565.0</td>
      <td>584.0</td>
      <td>2.5313</td>
      <td>45800.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15663</th>
      <td>-122.44</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>3830.0</td>
      <td>NaN</td>
      <td>1310.0</td>
      <td>963.0</td>
      <td>3.4801</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>20484</th>
      <td>-118.72</td>
      <td>34.28</td>
      <td>17.0</td>
      <td>3051.0</td>
      <td>NaN</td>
      <td>1705.0</td>
      <td>495.0</td>
      <td>5.7376</td>
      <td>218600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9814</th>
      <td>-121.93</td>
      <td>36.62</td>
      <td>34.0</td>
      <td>2351.0</td>
      <td>NaN</td>
      <td>1063.0</td>
      <td>428.0</td>
      <td>3.7250</td>
      <td>278000.0</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing["median_income"].hist()
```




    <Axes: >




    
![png](output_24_1.png)
    



```python
test_set["median_income"].hist()
```




    <Axes: >




    
![png](output_25_1.png)
    


The distributions seem roughly equivalent.
Thus we have not introduced any bias so far, at least as far as the median income is concerned.

---

Now let's focus on a slightly different scenario. 

Suppose you already know that ``median_income`` is an important feature.

In this case you want to be sure that each value for ``median_income`` is
sufficiently represented inside the test set, i.e., you want to ensure
that each ''category'' of ``median_income``has enough samples in the the test set.

The histogram suggests that values are mostly concentrated within 2 and 5.

Therefore you want to be sure that this peculiarity persists in the 
test set.


```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```


```python
housing["income_cat"].value_counts().sort_index()
```




    income_cat
    1     822
    2    6581
    3    7236
    4    3639
    5    2362
    Name: count, dtype: int64




```python
housing["income_cat"].hist()
```




    <Axes: >




    
![png](output_29_1.png)
    


Now that you have created the categories, namely the _strata_,
you can perform a stratified sampling with 
 ``StratifiedShuffleSplit``.

Each different value for ``income_cat`` is regarded as a stratum.



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

Let's see if it worked as expected


```python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```




    income_cat
    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114341
    1    0.039971
    Name: count, dtype: float64




```python
housing["income_cat"].value_counts() / len(housing)
```




    income_cat
    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: count, dtype: float64




```python
strat_test_set['income_cat'].hist()
```




    <Axes: >




    
![png](output_35_1.png)
    



```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
```


```python
compare_props
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall</th>
      <th>Stratified</th>
      <th>Random</th>
      <th>Rand. %error</th>
      <th>Strat. %error</th>
    </tr>
    <tr>
      <th>income_cat</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.039826</td>
      <td>0.039971</td>
      <td>0.040213</td>
      <td>0.973236</td>
      <td>0.364964</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.318847</td>
      <td>0.318798</td>
      <td>0.324370</td>
      <td>1.732260</td>
      <td>-0.015195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.350581</td>
      <td>0.350533</td>
      <td>0.358527</td>
      <td>2.266446</td>
      <td>-0.013820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.176308</td>
      <td>0.176357</td>
      <td>0.167393</td>
      <td>-5.056334</td>
      <td>0.027480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.114438</td>
      <td>0.114341</td>
      <td>0.109496</td>
      <td>-4.318374</td>
      <td>-0.084674</td>
    </tr>
  </tbody>
</table>
</div>



Now you may remove the ``income_cat`` feature in order to get back to the original dataset.

__Note__: In this example we assumed that ``median_income`` is an important feature for the problem in hand.
For this reason we are allowed to use a stratified sampling strategy wrt to this attribute. 

However, most of the time, you don't know which may be an important feature, therefore it is difficult to define
strata upon which performing a stratified sampling. 

__That said, you must always be aware that sampling can potentially mess up your entire project!__



```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

# Discover and visualize the data to gain insights
The following stage is very important. 
In fact, plotting is probably the best way to visualize and to gain insights about your data.

Let's create a copy of the training set in order to keep everything clean. 

__Note__: We are plotting the training set. 



```python
housing = strat_train_set.copy()
```


```python
# scatter plot of longitude and latitude
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
```

    Saving figure bad_visualization_plot
    


    
![png](output_42_1.png)
    


Pretty bad right?

Remember to be careful about the quality of your plots.

Let's change the value of ``alpha``




```python
import seaborn as sns
ax = sns.scatterplot(data=housing, x="longitude", y="latitude", alpha=0.1)
ax.annotate('Bay Area', xy=(-122.5, 38), xytext=(-124,40 ), xycoords='data',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
ax.annotate('L.A. ', xy=(-118, 34), xytext=(-120,33), xycoords='data',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
save_fig("better_visualization_plot")
```

    Saving figure better_visualization_plot
    


    
![png](output_44_1.png)
    


Now, we are able to see that most of the houses are concentrated between the Bay Area and around the
regions of L.A. and San Diego.

Let's try to introduce other information inside the plot.

1. The price of the houses -> denoted by the color of the circle, darker is higher
2. Population of the district -> denoted by the size of the circle




```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```

    Saving figure housing_prices_scatterplot
    


    
![png](output_46_1.png)
    


...fancier...


```python
import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/img/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()
```

    /var/folders/jq/lttlg11s5r32_42f7dnjwchw0000gn/T/ipykernel_77161/719456902.py:16: UserWarning: FixedFormatter should only be used together with FixedLocator
      cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
    

    Saving figure california_housing_prices_plot
    


    
![png](output_48_2.png)
    


_Question_:
Try to draw some observation given the plot above.


## Looking for Correlations
When the dataset is not too large you can compute the *standard correlation coefficient* (also known as the Pearson's coefficient).

This coefficient quantify how much two features are correlated with each other.



```python
corr_matrix = housing.corr(numeric_only=True)
```


```python
corr_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.000000</td>
      <td>-0.924478</td>
      <td>-0.105823</td>
      <td>0.048909</td>
      <td>0.076686</td>
      <td>0.108071</td>
      <td>0.063146</td>
      <td>-0.019615</td>
      <td>-0.047466</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.924478</td>
      <td>1.000000</td>
      <td>0.005737</td>
      <td>-0.039245</td>
      <td>-0.072550</td>
      <td>-0.115290</td>
      <td>-0.077765</td>
      <td>-0.075146</td>
      <td>-0.142673</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.105823</td>
      <td>0.005737</td>
      <td>1.000000</td>
      <td>-0.364535</td>
      <td>-0.325101</td>
      <td>-0.298737</td>
      <td>-0.306473</td>
      <td>-0.111315</td>
      <td>0.114146</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.048909</td>
      <td>-0.039245</td>
      <td>-0.364535</td>
      <td>1.000000</td>
      <td>0.929391</td>
      <td>0.855103</td>
      <td>0.918396</td>
      <td>0.200133</td>
      <td>0.135140</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.076686</td>
      <td>-0.072550</td>
      <td>-0.325101</td>
      <td>0.929391</td>
      <td>1.000000</td>
      <td>0.876324</td>
      <td>0.980167</td>
      <td>-0.009643</td>
      <td>0.047781</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.108071</td>
      <td>-0.115290</td>
      <td>-0.298737</td>
      <td>0.855103</td>
      <td>0.876324</td>
      <td>1.000000</td>
      <td>0.904639</td>
      <td>0.002421</td>
      <td>-0.026882</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.063146</td>
      <td>-0.077765</td>
      <td>-0.306473</td>
      <td>0.918396</td>
      <td>0.980167</td>
      <td>0.904639</td>
      <td>1.000000</td>
      <td>0.010869</td>
      <td>0.064590</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.019615</td>
      <td>-0.075146</td>
      <td>-0.111315</td>
      <td>0.200133</td>
      <td>-0.009643</td>
      <td>0.002421</td>
      <td>0.010869</td>
      <td>1.000000</td>
      <td>0.687151</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.047466</td>
      <td>-0.142673</td>
      <td>0.114146</td>
      <td>0.135140</td>
      <td>0.047781</td>
      <td>-0.026882</td>
      <td>0.064590</td>
      <td>0.687151</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Since you are interesetd in predicting the ``median_house_value`` we can look at how much the other
features are correlated with it.


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687151
    total_rooms           0.135140
    housing_median_age    0.114146
    households            0.064590
    total_bedrooms        0.047781
    population           -0.026882
    longitude            -0.047466
    latitude             -0.142673
    Name: median_house_value, dtype: float64



The coefficients range from -1 to 1. A value of -1 means there is a strong negative correlation, while a value of +1 means there is
a strong positive correlation.

The strength of the correlation gradually fade away as the coefficient approaches to $0$.

***
__Question__:

Pearson's coefficient has a major limitation. What is it?

***

Now, you might have spotted some important features. 
Let's try to focus on those features.


```python
# from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```

    Saving figure scatter_matrix_plot
    


    
![png](output_55_1.png)
    


Take a look at the results.  Which is the most promising feature?


```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.2)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")
```

    Saving figure income_vs_house_value_scatterplot
    


    
![png](output_57_1.png)
    


What can be concluded by the above plot? ......
Also, there are some strange patterns in this figure: the straight line at the top of the plot.
What is it?...


```python
housing_low_income = housing[(housing['median_income']<2) & (housing['median_house_value']==housing['median_house_value'].max()) ]
housing_low_income.head(16)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4861</th>
      <td>-118.28</td>
      <td>34.02</td>
      <td>29.0</td>
      <td>515.0</td>
      <td>229.0</td>
      <td>2690.0</td>
      <td>217.0</td>
      <td>0.4999</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-122.27</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>249.0</td>
      <td>78.0</td>
      <td>396.0</td>
      <td>85.0</td>
      <td>1.2434</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>6651</th>
      <td>-118.14</td>
      <td>34.15</td>
      <td>41.0</td>
      <td>1256.0</td>
      <td>407.0</td>
      <td>855.0</td>
      <td>383.0</td>
      <td>1.9923</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>15615</th>
      <td>-122.41</td>
      <td>37.81</td>
      <td>31.0</td>
      <td>3991.0</td>
      <td>1311.0</td>
      <td>2305.0</td>
      <td>1201.0</td>
      <td>1.8981</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>-120.10</td>
      <td>38.91</td>
      <td>33.0</td>
      <td>1561.0</td>
      <td>282.0</td>
      <td>30.0</td>
      <td>11.0</td>
      <td>1.8750</td>
      <td>500001.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4259</th>
      <td>-118.34</td>
      <td>34.10</td>
      <td>29.0</td>
      <td>3193.0</td>
      <td>1452.0</td>
      <td>2039.0</td>
      <td>1265.0</td>
      <td>1.8209</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16642</th>
      <td>-120.67</td>
      <td>35.30</td>
      <td>19.0</td>
      <td>1540.0</td>
      <td>715.0</td>
      <td>1799.0</td>
      <td>635.0</td>
      <td>0.7025</td>
      <td>500001.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>6688</th>
      <td>-118.08</td>
      <td>34.15</td>
      <td>28.0</td>
      <td>238.0</td>
      <td>58.0</td>
      <td>142.0</td>
      <td>31.0</td>
      <td>0.4999</td>
      <td>500001.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>6639</th>
      <td>-118.15</td>
      <td>34.15</td>
      <td>52.0</td>
      <td>275.0</td>
      <td>123.0</td>
      <td>273.0</td>
      <td>111.0</td>
      <td>1.1667</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>459</th>
      <td>-122.25</td>
      <td>37.87</td>
      <td>52.0</td>
      <td>609.0</td>
      <td>236.0</td>
      <td>1349.0</td>
      <td>250.0</td>
      <td>1.1696</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>17819</th>
      <td>-121.90</td>
      <td>37.39</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>14.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>1.7361</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>10448</th>
      <td>-117.67</td>
      <td>33.47</td>
      <td>22.0</td>
      <td>2728.0</td>
      <td>616.0</td>
      <td>1081.0</td>
      <td>566.0</td>
      <td>1.6393</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>10574</th>
      <td>-117.70</td>
      <td>33.72</td>
      <td>6.0</td>
      <td>211.0</td>
      <td>51.0</td>
      <td>125.0</td>
      <td>44.0</td>
      <td>1.9659</td>
      <td>500001.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>15661</th>
      <td>-122.42</td>
      <td>37.78</td>
      <td>27.0</td>
      <td>1728.0</td>
      <td>884.0</td>
      <td>1211.0</td>
      <td>752.0</td>
      <td>0.8543</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age", 'ocean_proximity']
sns.pairplot(housing[attributes].sample(200), hue='ocean_proximity')
```

    /Users/lucio/.pyenv/versions/dev/lib/python3.9/site-packages/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)
    




    <seaborn.axisgrid.PairGrid at 0x17d4665e0>




    
![png](output_60_2.png)
    


## Experimenting with Attribute Combinations

One last thing you may want to do before actually preparing the data for Machine
Learning algorithms is to try to combine different attributes together.

For instance, the total number of rooms inside a district is not very useful 
unless we combine this information with the number of households in a district.

Indeed, it interesting to look at the number of rooms per household.



```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
```

There are other attributes that can be treated as the ``total_rooms`` attribute.


```python
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["rooms_per_household"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

Now, let's check if these transformations bring any improvement to the correlation matrix.



```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687151
    rooms_per_household         0.146255
    total_rooms                 0.135140
    housing_median_age          0.114146
    households                  0.064590
    total_bedrooms              0.047781
    bedrooms_per_room          -0.006456
    population_per_household   -0.021991
    population                 -0.026882
    longitude                  -0.047466
    latitude                   -0.142673
    Name: median_house_value, dtype: float64



Not bad! The new attributes are more correlated than the original counterparts.


### Recap
Keep in mind that gaining insights from the plots, making assumptions based on the data is good. 
However it is an __iterative__ process. 

After you have built a prototype of your model, it is always good to go back and explore different paths.


# Prepare the data for Machine Learning algorithms
First, you need to separate the Xs from the ys.


```python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```

## Data Cleaning
We know from the previous section there are missing data we need to take care of.

``total_bedroom`` has missing values.
We can decide wether to fill these values or to drop the entries associated with them.

This time, we decide to replace the missing values with the median.



```python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1606</th>
      <td>-122.08</td>
      <td>37.88</td>
      <td>26.0</td>
      <td>2947.0</td>
      <td>NaN</td>
      <td>825.0</td>
      <td>626.0</td>
      <td>2.9330</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>10915</th>
      <td>-117.87</td>
      <td>33.73</td>
      <td>45.0</td>
      <td>2264.0</td>
      <td>NaN</td>
      <td>1970.0</td>
      <td>499.0</td>
      <td>3.4193</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19150</th>
      <td>-122.70</td>
      <td>38.35</td>
      <td>14.0</td>
      <td>2313.0</td>
      <td>NaN</td>
      <td>954.0</td>
      <td>397.0</td>
      <td>3.7813</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4186</th>
      <td>-118.23</td>
      <td>34.13</td>
      <td>48.0</td>
      <td>1308.0</td>
      <td>NaN</td>
      <td>835.0</td>
      <td>294.0</td>
      <td>4.2891</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16885</th>
      <td>-122.40</td>
      <td>37.58</td>
      <td>26.0</td>
      <td>3281.0</td>
      <td>NaN</td>
      <td>1145.0</td>
      <td>480.0</td>
      <td>6.3580</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1606</th>
      <td>-122.08</td>
      <td>37.88</td>
      <td>26.0</td>
      <td>2947.0</td>
      <td>433.0</td>
      <td>825.0</td>
      <td>626.0</td>
      <td>2.9330</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>10915</th>
      <td>-117.87</td>
      <td>33.73</td>
      <td>45.0</td>
      <td>2264.0</td>
      <td>433.0</td>
      <td>1970.0</td>
      <td>499.0</td>
      <td>3.4193</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19150</th>
      <td>-122.70</td>
      <td>38.35</td>
      <td>14.0</td>
      <td>2313.0</td>
      <td>433.0</td>
      <td>954.0</td>
      <td>397.0</td>
      <td>3.7813</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4186</th>
      <td>-118.23</td>
      <td>34.13</td>
      <td>48.0</td>
      <td>1308.0</td>
      <td>433.0</td>
      <td>835.0</td>
      <td>294.0</td>
      <td>4.2891</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>16885</th>
      <td>-122.40</td>
      <td>37.58</td>
      <td>26.0</td>
      <td>3281.0</td>
      <td>433.0</td>
      <td>1145.0</td>
      <td>480.0</td>
      <td>6.3580</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



Another way of replacing missing values in a dataset is via the class
``Imputer`` of ``sklearn``.




```python
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median")
```

Remove the attributes containing text because the median can only be computed wrt numerical attributes:


```python
#housing_num = housing.drop('ocean_proximity', axis=1)
housing_num = housing.select_dtypes(include=[np.number])
housing_num.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.8750</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
    </tr>
  </tbody>
</table>
</div>



Now you can fit the imputer instance to the training data using the ``fit()`` method:


```python
imputer.fit(housing_num)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div></div></div>




```python
imputer.statistics_
```




    array([-118.51   ,   34.26   ,   29.     , 2119.     ,  433.     ,
           1164.     ,  408.     ,    3.54155])



At this point the imputer, after the call to the  ``fit`` method, has only computed
the values for for replacing potential NaN in the dataset.

The NaN are still in the dataset at this point!

Also, the median is compute wrt to every attribute in the dataset. 
Even though we have missing values only in ``total_bedrooms`` it is preferable to define
a replace strategy for every attribute, as it may happen that, at some point, there are missing values in 
other fields of the dataset.

So we are taking precautions!

Check that this is the same as manually computing the median of each attribute.


```python
housing_num.median().values
```




    array([-118.51   ,   34.26   ,   29.     , 2119.     ,  433.     ,
           1164.     ,  408.     ,    3.54155])



The imputer is trained. It can be used to modify the dataset.



```python
X = imputer.transform(housing_num)
```


```python
X
```




    array([[-1.2146e+02,  3.8520e+01,  2.9000e+01, ...,  2.2370e+03,
             7.0600e+02,  2.1736e+00],
           [-1.1723e+02,  3.3090e+01,  7.0000e+00, ...,  2.0150e+03,
             7.6800e+02,  6.3373e+00],
           [-1.1904e+02,  3.5370e+01,  4.4000e+01, ...,  6.6700e+02,
             3.0000e+02,  2.8750e+00],
           ...,
           [-1.2272e+02,  3.8440e+01,  4.8000e+01, ...,  4.5800e+02,
             1.7200e+02,  3.1797e+00],
           [-1.2270e+02,  3.8310e+01,  1.4000e+01, ...,  1.2080e+03,
             5.0100e+02,  4.1964e+00],
           [-1.2214e+02,  3.9970e+01,  2.7000e+01, ...,  6.2500e+02,
             1.9700e+02,  3.1319e+00]])



The result is a plain Numpy array containing the transformed features. 
If you want a pandas dataframe again:



```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))
```


```python
housing_tr.loc[sample_incomplete_rows.index.values]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1606</th>
      <td>-122.08</td>
      <td>37.88</td>
      <td>26.0</td>
      <td>2947.0</td>
      <td>433.0</td>
      <td>825.0</td>
      <td>626.0</td>
      <td>2.9330</td>
    </tr>
    <tr>
      <th>10915</th>
      <td>-117.87</td>
      <td>33.73</td>
      <td>45.0</td>
      <td>2264.0</td>
      <td>433.0</td>
      <td>1970.0</td>
      <td>499.0</td>
      <td>3.4193</td>
    </tr>
    <tr>
      <th>19150</th>
      <td>-122.70</td>
      <td>38.35</td>
      <td>14.0</td>
      <td>2313.0</td>
      <td>433.0</td>
      <td>954.0</td>
      <td>397.0</td>
      <td>3.7813</td>
    </tr>
    <tr>
      <th>4186</th>
      <td>-118.23</td>
      <td>34.13</td>
      <td>48.0</td>
      <td>1308.0</td>
      <td>433.0</td>
      <td>835.0</td>
      <td>294.0</td>
      <td>4.2891</td>
    </tr>
    <tr>
      <th>16885</th>
      <td>-122.40</td>
      <td>37.58</td>
      <td>26.0</td>
      <td>3281.0</td>
      <td>433.0</td>
      <td>1145.0</td>
      <td>480.0</td>
      <td>6.3580</td>
    </tr>
  </tbody>
</table>
</div>



---
**Aside: Scikit-Learn Design**

Here are some of the main design princoples of the Scikit-Learn's API


* **Consistency**. All objects share a consistent and simple interface
    * *Estimators* - Any object that can estimate some parameters based on a dataset
       is called an estimator (e.g., an imputer is an estimator). The estimation itself is
       performed by the fit() method, and it takes only a dataset as a parameter (or
       two for supervised learning algorithms; the second dataset contains the
       labels). Any other parameter needed to guide the estimation process is con
       sidered a hyperparameter (such as an imputer ’s strategy ), and it must be set
       as an instance variable (generally via a constructor parameter).
    * *Transformers* - Some estimators (e.g., the imputer) can also transfom a dataset. 
       The transformation is performed by the ``transform()`` method with the dataset to
       transform as a parameter. It returns the transformed dataset.
       
       All the transformer also have a convenient method called ``fit_transform()`` which performs
       both the fit and the transform stages in a single step.
    * *Predictors* - some estimators are capable of making predictions given a dataset (e.g., ``LinearRegression``).
        A predictor has a ``predict()`` method that takes a
        dataset of new instances and returns a dataset of corresponding predictions. It
        also has a ``score()`` method that measures the quality of the predictions given
        a test set
        
* **Inspection** - All the estimator’s hyperparameters are accessible directly via public
    instance variables (e.g., ``imputer.strategy`` ), and all the estimator’s learned
    parameters are also accessible via public instance variables with an underscore
    suffix (e.g., ``imputer.statistics_``)     
    
* **Nonproliferation** of classes. Datasets are represented as NumPy arrays or SciPy
sparse matrices, instead of homemade classes. Hyperparameters are just regular
Python strings or numbers.

* **Composition**. Existing building blocks are reused as much as possible. For
example, it is easy to create a Pipeline estimator from an arbitrary sequence of
transformers followed by a final estimator, as we will see.

* **Sensible defaults**. Scikit-Learn provides reasonable default values for most
parameters, making it easy to create a baseline working system quickly.

---

### Handling Text and Categorical Attributes
Now let's preprocess the categorical input feature, `ocean_proximity`:


```python
housing_cat = housing.select_dtypes(include=['object'])
housing_cat.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>18125</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>5830</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17989</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



sklearn provides a class for transforming categorical values into numerical values.


```python
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
```


```python
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[1.],
           [4.],
           [1.],
           [4.],
           [0.],
           [3.],
           [0.],
           [0.],
           [0.],
           [0.]])




```python
print(ordinal_encoder.categories_)
```

    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
          dtype=object)]
    

This is not the best way to treat categorical data, though.

In fact one issue that may occur is that ML algorithm will assume that two close values have a similar meaning.

This is not always the case, especially in our context. 

In fact the fact  "<1H OCEAN" and "INLAND" are not similar at all, despite their close corresponding numerical values.

For this reason it is always better to transform categorical values according to a one-hot-encoding strategy.

Of course, ``sklearn`` provides a transformer: the ``OneHotEncoder``.


```python
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array  by calling  `toarray()`:


```python
housing_cat_1hot.toarray()
```




    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])



Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:


```python
print(cat_encoder.categories_)
```

    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
          dtype=object)]
    

Let's create a custom transformer to add extra attributes:


```python
housing.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'ocean_proximity'],
          dtype='object')



## Custom Transformers

Although Scikit-Learn provides many useful transformers, 
sometimes you may need to write your own transformer.


You want your transformer to work seamlessly with Scikit-Learn functionalities (such as pipelines), and
since Scikit-Learn relies on duck typing (not inheritance), all you need is to create a class and implement three methods: ``fit()``
(returning ``self`` ), ``transform()`` , and ``fit_transform()``.

You can get the last one for
free by simply adding TransformerMixin as a base class. Also, if you add BaseEstimator as a base class 
(and avoid ``*args`` and ``**kargs`` in your constructor) you will get
two extra methods ( ``get_params()`` and ``set_params()``) that will be useful for automatic hyperparameter tuning. 
For example, here is a small transformer class that adds
the combined attributes we discussed earlier:


```python
from sklearn.base import BaseEstimator, TransformerMixin
# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            #np.c_ concatenates arrays
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

```


```python
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)
pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+['roomsPerHouseHolds', 'PopulationPerHouseholds', 'bedRoomsPerRoom']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>roomsPerHouseHolds</th>
      <th>PopulationPerHouseholds</th>
      <th>bedRoomsPerRoom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>INLAND</td>
      <td>5.485836</td>
      <td>3.168555</td>
      <td>0.205784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>NEAR OCEAN</td>
      <td>6.927083</td>
      <td>2.623698</td>
      <td>0.160714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.875</td>
      <td>INLAND</td>
      <td>5.393333</td>
      <td>2.223333</td>
      <td>0.191595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>NEAR OCEAN</td>
      <td>3.886128</td>
      <td>1.859213</td>
      <td>0.276505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-118.7</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>&lt;1H OCEAN</td>
      <td>6.096552</td>
      <td>3.167241</td>
      <td>0.182692</td>
    </tr>
  </tbody>
</table>
</div>



Alternatively, you can use Scikit-Learn's `FunctionTransformer` class which lets you easily create a transformer based only on a function. 

Note that we need to set `validate=False` because the data contains non-float values (`validate` by default is `False` in Scikit-Learn 0.22).


```python
from sklearn.preprocessing import FunctionTransformer # it works as a wrapper

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)
```


```python
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>INLAND</td>
      <td>5.485836</td>
      <td>3.168555</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>NEAR OCEAN</td>
      <td>6.927083</td>
      <td>2.623698</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.875</td>
      <td>INLAND</td>
      <td>5.393333</td>
      <td>2.223333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>NEAR OCEAN</td>
      <td>3.886128</td>
      <td>1.859213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-118.7</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>&lt;1H OCEAN</td>
      <td>6.096552</td>
      <td>3.167241</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Scaling
There are two common strategies:
1. MinMax Scaling
2. Standardization
ML algorithms prefer input data with the same scale.
Otherwise, the performance of the algorithm can be altered.




```python
housing_num.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.575635</td>
      <td>35.639314</td>
      <td>28.653404</td>
      <td>2622.539789</td>
      <td>534.914639</td>
      <td>1419.687379</td>
      <td>497.011810</td>
      <td>3.875884</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.001828</td>
      <td>2.137963</td>
      <td>12.574819</td>
      <td>2138.417080</td>
      <td>412.665649</td>
      <td>1115.663036</td>
      <td>375.696156</td>
      <td>1.904931</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.499900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.940000</td>
      <td>18.000000</td>
      <td>1443.000000</td>
      <td>295.000000</td>
      <td>784.000000</td>
      <td>279.000000</td>
      <td>2.566950</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.510000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2119.000000</td>
      <td>433.000000</td>
      <td>1164.000000</td>
      <td>408.000000</td>
      <td>3.541550</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.720000</td>
      <td>37.000000</td>
      <td>3141.000000</td>
      <td>644.000000</td>
      <td>1719.000000</td>
      <td>602.000000</td>
      <td>4.745325</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6210.000000</td>
      <td>35682.000000</td>
      <td>5358.000000</td>
      <td>15.000100</td>
    </tr>
  </tbody>
</table>
</div>



Scikit-Learn provides two transformers: (i)``MinMaxScaler``, (ii) ``StandardScaler``.

Like other transformers they provide the same ``fit()``-> ``transform()`` mechanism.



## Transformation Pipelines
A  best practice when working with transformers or imputer is to define Pipelines
of transformations.

Scikit-Learn provides the Pipeline class.

Here is a small pipeline for the numerical attributes:


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```


```python
housing_num_tr
```




    array([[-0.94135046,  1.34743822,  0.02756357, ...,  0.01739526,
             0.00622264, -0.12236839],
           [ 1.17178212, -1.19243966, -1.72201763, ...,  0.56925554,
            -0.04081077, -0.76455102],
           [ 0.26758118, -0.1259716 ,  1.22045984, ..., -0.01802432,
            -0.07537122, -0.32454514],
           ...,
           [-1.5707942 ,  1.31001828,  1.53856552, ..., -0.5092404 ,
            -0.03743619,  0.29100658],
           [-1.56080303,  1.2492109 , -1.1653327 , ...,  0.32814891,
            -0.05915604, -0.43510673],
           [-1.28105026,  2.02567448, -0.13148926, ...,  0.01407228,
             0.00657083, -0.1229037 ]])



The Pipeline constructor takes a list of <name,estimator> pairs defining a sequence of
steps. 

Every object in a pipeline, except the last one, must be a transformer.

The last spot of the pipeline is usually dedicated to either an estimator or a predictor.


Calling ``fit()`` on the pipeline implies the call to ``fit_transform()`` on every
object in the pipeline, sequentially.

The output of a transformer becomes the input to the next object in the pipeline,
until the end of the pipeline is reached.

So far, we implemented a pipeline for the numerical values.
However we have to deal also with not-numerical values.

A possible approach is to design two different pipelines, one for each type of values and
then using a ``ColumnTransformer``.

*How does it work?*
This Transformer enables the possibility of applying each transformer (or another Pipeline) 
to a specific subset of column.

Then the result is merged in  a single feature space.

A ``ColumnTransformer`` asks for a list of tuples.

Each tuple in the list contains: the name of the transformer, the transformer, the column(s) upon which the transformations
need to be performed.


```python
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20
```


```python
list(housing_num)
```




    ['longitude',
     'latitude',
     'housing_median_age',
     'total_rooms',
     'total_bedrooms',
     'population',
     'households',
     'median_income']




```python
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```


```python
housing_prepared
```




    array([[-0.94135046,  1.34743822,  0.02756357, ...,  0.        ,
             0.        ,  0.        ],
           [ 1.17178212, -1.19243966, -1.72201763, ...,  0.        ,
             0.        ,  1.        ],
           [ 0.26758118, -0.1259716 ,  1.22045984, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [-1.5707942 ,  1.31001828,  1.53856552, ...,  0.        ,
             0.        ,  0.        ],
           [-1.56080303,  1.2492109 , -1.1653327 , ...,  0.        ,
             0.        ,  0.        ],
           [-1.28105026,  2.02567448, -0.13148926, ...,  0.        ,
             0.        ,  0.        ]])




```python
housing_prepared[0]
```




    array([-0.94135046,  1.34743822,  0.02756357,  0.58477745,  0.63818349,
            0.73260236,  0.55628602, -0.8936472 ,  0.01739526,  0.00622264,
           -0.12236839,  0.        ,  1.        ,  0.        ,  0.        ,
            0.        ])



# Select and train a model 


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>



Done! You now have a working Linear Regression model.

It is not so hard, is it? 

Let's use the regressor to make some prediction.


```python
# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
# the data need to be transformed according to the same pipeline used for the training set
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
```

    Predictions: [ 86528. 303680. 153984. 184384. 245120.]
    

Compare it against the actual values:


```python
print("Labels:", list(some_labels))
```

    Labels: [72100.0, 279600.0, 82700.0, 112500.0, 238300.0]
    


```python
some_data_prepared
```




    array([[-0.94135046,  1.34743822,  0.02756357,  0.58477745,  0.63818349,
             0.73260236,  0.55628602, -0.8936472 ,  0.01739526,  0.00622264,
            -0.12236839,  0.        ,  1.        ,  0.        ,  0.        ,
             0.        ],
           [ 1.17178212, -1.19243966, -1.72201763,  1.26146668,  0.77941474,
             0.53361152,  0.72131799,  1.292168  ,  0.56925554, -0.04081077,
            -0.76455102,  0.        ,  0.        ,  0.        ,  0.        ,
             1.        ],
           [ 0.26758118, -0.1259716 ,  1.22045984, -0.46977281, -0.54767198,
            -0.67467519, -0.52440722, -0.52543365, -0.01802432, -0.07537122,
            -0.32454514,  0.        ,  1.        ,  0.        ,  0.        ,
             0.        ],
           [ 1.22173797, -1.35147437, -0.37006852, -0.34865152, -0.03875249,
            -0.46761716, -0.03729672, -0.86592882, -0.59513997, -0.10680295,
             0.88532487,  0.        ,  0.        ,  0.        ,  0.        ,
             1.        ],
           [ 0.43743108, -0.63581817, -0.13148926,  0.42717947,  0.27049524,
             0.37406031,  0.22089846,  0.32575178,  0.2512412 ,  0.00610923,
            -0.45139128,  1.        ,  0.        ,  0.        ,  0.        ,
             0.        ]])



It works. 

However, we still do not know how good this model actually is.

A classic approach when working with regression problem is to use the RMSE error in order to measure
its performace.


```python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    68732.09014153534



Well, it is not a great score considering that the ``median_housing_values`` ranges between 120k and 265k.

Now, compute some other metric over the training set ($l_1$ loss function).


```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
```




    49576.95221656977



Usually, if a model yield poor performance, the first thing to do is to try with another model.


(Obviously, you first need to check if your results are correct, i.e., you are not messed up with something in your model or with the data preparation stack)

Why don't you try with ``DecisionTreeRegressor``?


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor(random_state=42)</pre></div></div></div></div></div>




```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



Wow! 0 error! 

Should you be worried about this result or should you be happy for it?  Are you the legit ML GOAT?

## Better Evaluation Using Cross-Validation
A better way to evaluate the performance of a model during the training phase
is to use cross-validation, as it returns more reliable measures.

One way to evaluate the Decision Tree model would be to use the train_test_split
function to split the training set into a smaller training set and a validation set, then
train your models against the smaller training set and evaluate them against the validation set. 
It’s a bit of work, but nothing too difficult and it would work fairly well.

A great alternative is to use Scikit-Learn’s cross-validation feature. The following code
performs K-fold cross-validation: it randomly splits the training set into 10 distinct
subsets called folds, then it trains and evaluates the Decision Tree model 10 times,
picking a different fold for evaluation every time and training on the other 9 folds.
The result is an array containing the 10 evaluation scores:


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print("Scores:", rmse_scores)
print("Mean: ", rmse_scores.mean())
print("Std.: ", rmse_scores.std())
```

    Scores: [73268.75829306 70767.11195306 68982.65511803 73010.90062103
     69637.90724119 76320.96227001 70665.8419634  72510.2110183
     69528.32128254 72003.47458751]
    Mean:  71669.61443481396
    Std.:  2111.02350467858
    

Scikit-Learn cross-validation features expects a utility function
(greater is better) rather than a cost function (lower is better), this is why you need
to stick the negative sign.

Now the Decision Tree doesn’t look as good as it did earlier.

In fact, it seems to be worse than the Linear Regression model! 

Notice that cross-validation allows you to get not only an estimate of the performance of your model,
but it also measures the consistency of your model as the standard deviation of its performance.

Now let's do the same thing with the linear regression estimator.


```python
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Std.: ", scores.std())


print_scores(lin_rmse_scores)
```

    Scores: [71948.7546488  64283.1516368  67860.17012633 68745.28680085
     66883.94479826 72654.73108884 74605.05915192 68957.24746271
     66585.83620475 70208.45965328]
    Mean:  69273.26415725367
    Std.:  2968.2148979977046
    

#### Draw some conclusions...


Let's try something new. Here your are going to build a ``RandomForestRegressor`` (it is the ``ensamble`` module of sklearn).

It is an Ensamble Learning Method. 

**Note**: you need to specify `n_estimators=10` to avoid a warning about the fact that the default value is going to change to 100 in Scikit-Learn 0.22.


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
```




<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_estimators=10, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(n_estimators=10, random_state=42)</pre></div></div></div></div></div>




```python
# compute some predictions
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    22341.5023358149




```python
# compute the score of the model
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print_scores(forest_rmse_scores)
```

    Scores: [53929.2744405  50516.26306858 48469.73920774 53802.36353013
     50510.0169913  54493.51638967 55992.26456548 52421.10485337
     51330.28025752 56410.61931435]
    Mean:  52787.54426186342
    Std.:  2447.199161086955
    

**Question**

Is there any sign of overfitting?

*Motivate your answer*...



## Fine-Tune Your Model
After you tried a number of different solutions, you end up with a short list of promising models.

The goal is now to try to boost their performance via parameter tuning.


### Grid Search

You  want Scikit-Learn’s ``GridSearchCV`` to search the best configuration for you.

For example, the following code aims at finding  the best combination
of hyperparameters  for the RandomForestRegressor.

It should be noted that each dict  in ``param_grid`` is considered only once. 
Therefore each dict in the list corresponds to a configuration for the algorithm.

Sklearn also considers every possible combination of the selected parameters for the given model, therefore
it might required a lot of time!


```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
             param_grid=[{&#x27;max_features&#x27;: [2, 4, 6, 8],
                          &#x27;n_estimators&#x27;: [3, 10, 30]},
                         {&#x27;bootstrap&#x27;: [False], &#x27;max_features&#x27;: [2, 3, 4],
                          &#x27;n_estimators&#x27;: [3, 10]}],
             return_train_score=True, scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
             param_grid=[{&#x27;max_features&#x27;: [2, 4, 6, 8],
                          &#x27;n_estimators&#x27;: [3, 10, 30]},
                         {&#x27;bootstrap&#x27;: [False], &#x27;max_features&#x27;: [2, 3, 4],
                          &#x27;n_estimators&#x27;: [3, 10]}],
             return_train_score=True, scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>



Once the grid search has done, you get the best hyperparameters combination with:


```python
grid_search.best_params_
```




    {'max_features': 6, 'n_estimators': 30}



You can also take a reference directly to the best estimator.


```python
grid_search.best_estimator_
```




<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_features=6, n_estimators=30, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(max_features=6, n_estimators=30, random_state=42)</pre></div></div></div></div></div>



Let's look at the scores of each hyperparameter combination tested during the grid search:


```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    63288.87670039338 {'max_features': 2, 'n_estimators': 3}
    55424.127110004076 {'max_features': 2, 'n_estimators': 10}
    53040.925093237885 {'max_features': 2, 'n_estimators': 30}
    60716.02970729203 {'max_features': 4, 'n_estimators': 3}
    52712.42382890588 {'max_features': 4, 'n_estimators': 10}
    50635.4734837428 {'max_features': 4, 'n_estimators': 30}
    58846.03036812111 {'max_features': 6, 'n_estimators': 3}
    51675.04782358738 {'max_features': 6, 'n_estimators': 10}
    50035.014576859954 {'max_features': 6, 'n_estimators': 30}
    59040.6949299778 {'max_features': 8, 'n_estimators': 3}
    52862.945101484795 {'max_features': 8, 'n_estimators': 10}
    50200.73938739378 {'max_features': 8, 'n_estimators': 30}
    61811.05278347192 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54875.31396267672 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59775.04315789536 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52463.63507029581 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    58007.16857501234 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51178.07251852549 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
    


```python
pd.DataFrame(grid_search.cv_results_).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_features</th>
      <th>param_n_estimators</th>
      <th>param_bootstrap</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>...</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.048770</td>
      <td>0.010776</td>
      <td>0.002100</td>
      <td>0.000666</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 3}</td>
      <td>-4.022589e+09</td>
      <td>-4.017505e+09</td>
      <td>...</td>
      <td>-4.005482e+09</td>
      <td>6.417279e+07</td>
      <td>18</td>
      <td>-1.138549e+09</td>
      <td>-1.159107e+09</td>
      <td>-1.150563e+09</td>
      <td>-1.035764e+09</td>
      <td>-1.118302e+09</td>
      <td>-1.120457e+09</td>
      <td>4.450869e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.146819</td>
      <td>0.003618</td>
      <td>0.006154</td>
      <td>0.000248</td>
      <td>2</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 10}</td>
      <td>-2.881916e+09</td>
      <td>-3.060012e+09</td>
      <td>...</td>
      <td>-3.071834e+09</td>
      <td>1.263879e+08</td>
      <td>11</td>
      <td>-5.810240e+08</td>
      <td>-6.081265e+08</td>
      <td>-5.905365e+08</td>
      <td>-5.856255e+08</td>
      <td>-5.985000e+08</td>
      <td>-5.927625e+08</td>
      <td>9.619323e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.436163</td>
      <td>0.003151</td>
      <td>0.017875</td>
      <td>0.000086</td>
      <td>2</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 30}</td>
      <td>-2.763005e+09</td>
      <td>-2.747263e+09</td>
      <td>...</td>
      <td>-2.813340e+09</td>
      <td>9.853812e+07</td>
      <td>9</td>
      <td>-4.350603e+08</td>
      <td>-4.471071e+08</td>
      <td>-4.343791e+08</td>
      <td>-4.308928e+08</td>
      <td>-4.462991e+08</td>
      <td>-4.387477e+08</td>
      <td>6.652582e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.075655</td>
      <td>0.001108</td>
      <td>0.001828</td>
      <td>0.000097</td>
      <td>4</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 3}</td>
      <td>-3.687271e+09</td>
      <td>-3.479423e+09</td>
      <td>...</td>
      <td>-3.686436e+09</td>
      <td>2.064749e+08</td>
      <td>16</td>
      <td>-9.456080e+08</td>
      <td>-1.019586e+09</td>
      <td>-9.883500e+08</td>
      <td>-9.644960e+08</td>
      <td>-9.591666e+08</td>
      <td>-9.754413e+08</td>
      <td>2.603878e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.243728</td>
      <td>0.001634</td>
      <td>0.005741</td>
      <td>0.000128</td>
      <td>4</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 10}</td>
      <td>-2.723172e+09</td>
      <td>-2.714183e+09</td>
      <td>...</td>
      <td>-2.778600e+09</td>
      <td>7.612906e+07</td>
      <td>7</td>
      <td>-4.931502e+08</td>
      <td>-5.438250e+08</td>
      <td>-5.221810e+08</td>
      <td>-5.154513e+08</td>
      <td>-5.292002e+08</td>
      <td>-5.207616e+08</td>
      <td>1.670768e+07</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



## Randomized Search
Instead of providing each possible combination by hand,
 you can use ``RandomizedSearchCV``.

It is especially useful when you have a lot of hyperparameters, each of which may vary in an large range.

It evaluates a given number of random combinations by selecting a random
value for each hyperparameter at every iteration. This approach has two main
benefits:
1. 1.000 iterations will corresponds to 1000 different combination for each hyperparameter
2. you have more control over the amount of resources you want to allocate


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=11),
        'max_features': randint(low=1, high=3),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
```


```python
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

## Analyze the Best Models and Their Errors
the RandomForestRegressor is able to indicate the relative importance of each feature.



```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

Let’s display these importance scores next to their corresponding attribute names:


```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```

With this information in mind, you might want to revisit some of the steps you have included in yours pipeline.

For instance, you may want to try dropping some of the less useful features.
(e.g., apparently only one ocean_proximity category is really useful, so you could try
dropping the others).

---
### Evaluate Your System on the Test Set 
Training performance measures how good you model is in terms of __approximation__.

However, your ultimate goal is to provide a model that is as good as possible in terms of __generalization__.

However, the real test is performed against the test set. 

Hopefully, the best test on the training data will be the same as the one on the test set.

__Careful: Performance reported on the test must no affect the decision process when choosing the final model!__

You can take the ``full_pipeline`` we build before, as it is already trained, and call
the ``transform`` method in order to transform the data in the test set.

**Note:** Be careful! You need to transform the data according to the same parameters of the transformer objects
obtained during the training stage. 
__If you call fit upon the test set, it is an error!__


```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test) #!!!!!!!!! do not call fit or fit_transform!
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```


```python
final_rmse
```

**What to expect from this evaluation?**

The performance should roughly match the ones obtained using cross validation.

This is not the case, but if that happens, you must avoid the trap of trying to fit the data
in the test set, usually the boost in performance will actually be a ''fake" improvement, since your model
will likely fail to generalize to new data.


# Extra material

## A full pipeline with both preparation and prediction



```python
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)
```

## Model persistence using joblib
One the training phase is done, it is
better to save your progresses, i.e., your trained model.

Scikit-Learn provides the module ``joblib`` for easily 
save and restore  python objects in a regular .pickle file.

You should always prefer ``joblib`` over other serialization
techniques, as it is generally more efficient.



```python
my_model = full_pipeline_with_predictor
```

from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF
 



## 1
Let's train a support vector machine regressor mode (`sklearn.svm.SVR`).
It has several parameters, e.g., `kernel="linear|rbf"` and others (see the doc file).

Find the best possible configuration via parameter tuning.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100.]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30.],
         'gamma': [0.01, 0.03]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search.fit(housing_prepared, housing_labels)
```

The best model achieves the following score (evaluated using 5-fold cross validation):


```python
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
```

It should be worse than the `RandomForestRegressor`. 
Let's check the best hyperparameters found:


```python
grid_search.best_params_
```

## 2.

Replace `GridSearchCV` with `RandomizedSearchCV`.


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=5, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
```

The best model achieves the following score (evaluated using 5-fold cross validation):


```python
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
```

Now this is much closer to the performance of the `RandomForestRegressor` (but not quite there yet). Let's check the best hyperparameters found:


```python
rnd_search.best_params_
```

This time we ended up with RBF being the best choice for the kernel function.

Randomized search tends to find better hyperparameters than grid search.

Let's look at the exponential distribution we used, with `scale=1.0`. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.


```python
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
```

The distribution we used for `C` looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:


```python
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
```

The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

## 3.

Add a new transformer into the preparation pipeline. 
It selects the most important attributes.

The feature importance scores can be computed by any estimator object (e.g., the ``Random Forest Regressor``).

It is always better to provide the scores to the transformer, as opposed to let it compute them at every stage of training.


```python
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
```

Let's define the number of top features we want to keep:


```python
k = 5
```

Now let's look for the indices of the top k features:


```python
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices
```


```python
np.array(attributes)[top_k_feature_indices]
```

Let's double check that these are indeed the top k features:


```python
sorted(zip(feature_importances, attributes), reverse=True)[:k]
```

Looking good!
Now you can create a new pipeline which includes the ``TopFeatureSelector``.


```python
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])
```


```python
housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
```

Let's look at the features of the first 3 instances:


```python
housing_prepared_top_k_features[0:3]
```

Now let's double check that these are indeed the top k features:


```python
housing_prepared[0:3, top_k_feature_indices]
```

Everything's correct!

## 4.

Create a single pipeline responsible of the following steps:
1. preparation
2. feature selection
3. training of a model of your choice (``SVR``)


```python
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])
```


```python
prepare_select_and_predict_pipeline.fit(housing, housing_labels)
```

Let's try the full pipeline on a few instances:


```python
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))
```

The full pipeline works fine. 

You can try a separate pipeline with a more powerful model like the ``RandomForestRegressor``.

