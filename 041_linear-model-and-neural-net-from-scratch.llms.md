# Linear model and Neural Net from scratch

In this notebook we’re going to build and train a deep learning model “from scratch” – by which I mean that we’re not going to use any pre-built architecture, or optimizers, or data loading frameworks, etc.

Author

Benedict Thekkel

We’ll be assuming you already know the basics of how a neural network works. If you don’t, read this notebook first: [How does a neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work). We’ll be using Kaggle’s [Titanic](https://www.kaggle.com/competitions/titanic/) competition in this notebook, because it’s very small and simple, but also has displays many of the tricky real-life issues that we need to handle in most practical projects. (Note, however, that this competition is a small “learner” competition on Kaggle, so don’t expect to actually see much benefits from using a neural net just yet; that will come once we try our some real competitions!)

It’s great to be able to run the same notebook on your own machine or Colab, as well as Kaggle. To allow for this, we use this code to download the data as needed when not on Kaggle (see [this notebook](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners/) for details about this technique):

Note that the data for Kaggle comps always lives in the `../input` folder. The easiest way to get the path is to click the “K” button in the top-right of the Kaggle notebook, click on the folder shown there, and click the copy button.

We’ll be using *numpy* and *pytorch* for array calculations in this notebook, and *pandas* for working with tabular data, so we’ll import them and set them to display using a bit more space than they default to.

``` python
!pip list | grep nbdevAuto
```

    nbdevAuto                     0.0.95

``` python
from nbdevAuto.functions import kaggle_competition_download
from pathlib import Path
```

``` python
name = 'titanic'
datapath = Path('./Data')
kaggle_competition_download(name, folderpath = datapath)
```

    file exists

``` python
path = f'{datapath}/{name}'
path
```

    'Data/titanic'

``` python
import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)
```

## Cleaning the data

This is a *tabular data* competition – the data is in the form of a table. It’s provided as a Comma Separated Values (CSV) file. We can open it using the *pandas* library, which will create a `DataFrame`.

``` python
df = pd.read_csv(f'{path}/train.csv')
df
```

|  | PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 1 | 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 2 | 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN | S |
| 3 | 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S |
| 4 | 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 886 | 887 | 0 | 2 | Montvila, Rev. Juozas | male | 27.0 | 0 | 0 | 211536 | 13.0000 | NaN | S |
| 887 | 888 | 1 | 1 | Graham, Miss. Margaret Edith | female | 19.0 | 0 | 0 | 112053 | 30.0000 | B42 | S |
| 888 | 889 | 0 | 3 | Johnston, Miss. Catherine Helen "Carrie" | female | NaN | 1 | 2 | W./C. 6607 | 23.4500 | NaN | S |
| 889 | 890 | 1 | 1 | Behr, Mr. Karl Howell | male | 26.0 | 0 | 0 | 111369 | 30.0000 | C148 | C |
| 890 | 891 | 0 | 3 | Dooley, Mr. Patrick | male | 32.0 | 0 | 0 | 370376 | 7.7500 | NaN | Q |

891 rows × 12 columns

As we learned in the *How does a neural net really work* notebook, we going to want to multiply each column by some coefficients. But we can see in the `Cabin` column that there are `NaN` values, which is how Pandas refers to missing values. We can’t multiply something by a missing value!

Let’s check which columns contain `NaN` values. Pandas’ `isna()` function returns `True` (which is treated as `1` when used as a number) for `NaN` values, so we can just add them up for each column:

``` python
df.isna().sum()
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64

Notice that by default Pandas sums over columns.

We’ll need to replace the missing values with something. It doesn’t generally matter too much what we choose. We’ll use the most common value (the “*mode*”). We can use the `mode` function for that. One wrinkle is that it returns more than one row in the case of ties, so we just grab the first row with `iloc[0]`:

``` python
modes = df.mode().iloc[0]
modes
```

    PassengerId                      1
    Survived                       0.0
    Pclass                         3.0
    Name           Abbing, Mr. Anthony
    Sex                           male
    Age                           24.0
    SibSp                          0.0
    Parch                          0.0
    Ticket                        1601
    Fare                          8.05
    Cabin                      B96 B98
    Embarked                         S
    Name: 0, dtype: object

BTW, it’s never a good idea to use functions without understanding them. So be sure to google for anything you’re not familiar with. E.g if you want to learn about `iloc` (which is a very important function indeed!) then Google will give you a link to a [great tutorial](https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/).

Now that we’ve got the mode of each column, we can use `fillna` to replace the missing values with the mode of each column. We’ll do it “in place” – meaning that we’ll change the dataframe itself, rather than returning a new one.

``` python
df.fillna(modes, inplace=True)
```

We can now check there’s no missing values left:

``` python
df.isna().sum()
```

    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Cabin          0
    Embarked       0
    dtype: int64

Here’s how we get a quick summary of all the numeric columns in the dataset:

``` python
import numpy as np

df.describe(include=(np.number))
```

|  | PassengerId | Survived | Pclass | Age | SibSp | Parch | Fare |
|----|----|----|----|----|----|----|----|
| count | 891.000000 | 891.000000 | 891.000000 | 891.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean | 446.000000 | 0.383838 | 2.308642 | 28.566970 | 0.523008 | 0.381594 | 32.204208 |
| std | 257.353842 | 0.486592 | 0.836071 | 13.199572 | 1.102743 | 0.806057 | 49.693429 |
| min | 1.000000 | 0.000000 | 1.000000 | 0.420000 | 0.000000 | 0.000000 | 0.000000 |
| 25% | 223.500000 | 0.000000 | 2.000000 | 22.000000 | 0.000000 | 0.000000 | 7.910400 |
| 50% | 446.000000 | 0.000000 | 3.000000 | 24.000000 | 0.000000 | 0.000000 | 14.454200 |
| 75% | 668.500000 | 1.000000 | 3.000000 | 35.000000 | 1.000000 | 0.000000 | 31.000000 |
| max | 891.000000 | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |

We can see that `Fare` contains mainly values of around `0` to `30`, but there’s a few really big ones. This is very common with fields contain monetary values, and it can cause problems for our model, because once that column is multiplied by a coefficient later, the few rows with really big values will dominate the result.

You can see the issue most clearly visually by looking at a histogram, which shows a long tail to the right (and don’t forget: if you’re not entirely sure what a histogram is, Google “[histogram tutorial](https://www.google.com/search?q=histogram+tutorial&oq=histogram+tutorial)” and do a bit of reading before continuing on):

``` python
df['Fare'].hist();
```

![](041_linear-model-and-neural-net-from-scratch_files/figure-html/cell-13-output-1.png)

To fix this, the most common approach is to take the logarithm, which squishes the big numbers and makes the distribution more reasonable. Note, however, that there are zeros in the `Fare` column, and `log(0)` is infinite – to fix this, we’ll simply add `1` to all values first:

``` python
df['LogFare'] = np.log(df['Fare']+1)
```

The histogram now shows a more even distribution of values without the long tail:

``` python
df['LogFare'].hist();
```

![](041_linear-model-and-neural-net-from-scratch_files/figure-html/cell-15-output-1.png)

It looks from the `describe()` output like `Pclass` contains just 3 values, which we can confirm by looking at the [Data Dictionary](https://www.kaggle.com/competitions/titanic/data) (which you should always study carefully for any project!) –

``` python
pclasses = sorted(df.Pclass.unique())
pclasses
```

    [1, 2, 3]

Here’s how we get a quick summary of all the non-numeric columns in the dataset:

``` python
df.describe(include=[object])
```

|        | Name                    | Sex  | Ticket | Cabin   | Embarked |
|--------|-------------------------|------|--------|---------|----------|
| count  | 891                     | 891  | 891    | 891     | 891      |
| unique | 891                     | 2    | 681    | 147     | 3        |
| top    | Braund, Mr. Owen Harris | male | 347082 | B96 B98 | S        |
| freq   | 1                       | 577  | 7      | 691     | 646      |

Clearly we can’t multiply strings like `male` or `S` by coefficients, so we need to replace those with numbers.

We do that by creating new columns containing *dummy variables*. A dummy variable is a column that contains a `1` where a particular column contains a particular value, or a `0` otherwise. For instance, we could create a dummy variable for `Sex='male'`, which would be a new column containing `1` for rows where `Sex` is `'male'`, and 0 for rows where it isn’t.

Pandas can create these automatically using `get_dummies`, which also remove the original columns. We’ll create dummy variables for `Pclass`, even although it’s numeric, since the numbers `1`, `2`, and `3` correspond to first, second, and third class cabins - not to counts or measures that make sense to multiply by. We’ll also create dummies for `Sex` and `Embarked` since we’ll want to use those as predictors in our model. On the other hand, `Cabin`, `Name`, and `Ticket` have too many unique values for it to make sense creating dummy variables for them.

``` python
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
df.columns
```

    Index(['PassengerId', 'Survived', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'LogFare', 'Sex_female', 'Sex_male',
           'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'],
          dtype='object')

We can see that 5 columns have been added to the end – one for each of the possible values of each of the three columns we requested, and that those three requested columns have been removed.

Here’s what the first few rows of those newly added columns look like:

``` python
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
df[added_cols].head()
```

|  | Sex_male | Sex_female | Pclass_1 | Pclass_2 | Pclass_3 | Embarked_C | Embarked_Q | Embarked_S |
|----|----|----|----|----|----|----|----|----|
| 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| 3 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |

Now we can create our independent (predictors) and dependent (target) variables. They both need to be PyTorch tensors. Our dependent variable is `Survived`:

``` python
from torch import tensor

t_dep = tensor(df.Survived)
```

Our independent variables are all the continuous variables of interest plus all the dummy variables we just created:

``` python
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
df[indep_cols] = df[indep_cols].astype('float32')  
t_indep = tensor(df[indep_cols].values, dtype=torch.float)
t_indep
```

    tensor([[22.0000,  1.0000,  0.0000,  2.1102,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
            [38.0000,  1.0000,  0.0000,  4.2806,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
            [26.0000,  0.0000,  0.0000,  2.1889,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
            [35.0000,  1.0000,  0.0000,  3.9908,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
            [35.0000,  0.0000,  0.0000,  2.2028,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
            [24.0000,  0.0000,  0.0000,  2.2469,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000],
            [54.0000,  0.0000,  0.0000,  3.9677,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
            ...,
            [25.0000,  0.0000,  0.0000,  2.0857,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
            [39.0000,  0.0000,  5.0000,  3.4054,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000],
            [27.0000,  0.0000,  0.0000,  2.6391,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000],
            [19.0000,  0.0000,  0.0000,  3.4340,  0.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
            [24.0000,  1.0000,  2.0000,  3.1966,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
            [26.0000,  0.0000,  0.0000,  3.4340,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
            [32.0000,  0.0000,  0.0000,  2.1691,  1.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000]])

Here’s the number of rows and columns we have for our independent variables:

``` python
t_indep.shape
```

    torch.Size([891, 12])

## Setting up a linear model

Now that we’ve got a matrix of independent variables and a dependent variable vector, we can work on calculating our predictions and our loss. In this section, we’re going to manually do a single step of calculating predictions and loss for every row of our data.

Our first model will be a simple linear model. We’ll need a coefficient for each column in `t_indep`. We’ll pick random numbers in the range `(-0.5,0.5)`, and set our manual seed so that my explanations in the prose in this notebook will be consistent with what you see when you run it.

``` python
torch.manual_seed(442)

n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5
coeffs
```

    tensor([-0.4629,  0.1386,  0.2409, -0.2262, -0.2632, -0.3147,  0.4876,  0.3136,  0.2799, -0.4392,  0.2103,  0.3625])

Our predictions will be calculated by multiplying each row by the coefficients, and adding them up. One interesting point here is that we don’t need a separate constant term (also known as a “bias” or “intercept” term), or a column of all `1`s to give the same effect has having a constant term. That’s because our dummy variables already cover the entire dataset – e.g. there’s a column for “male” and a column for “female”, and everyone in the dataset is in exactly one of these; therefore, we don’t need a separate intercept term to cover rows that aren’t otherwise part of a column.

Here’s what the multiplication looks like:

``` python
t_indep*coeffs
```

    tensor([[-10.1838,   0.1386,   0.0000,  -0.4772,  -0.2632,  -0.0000,   0.0000,   0.0000,   0.2799,  -0.0000,   0.0000,   0.3625],
            [-17.5902,   0.1386,   0.0000,  -0.9681,  -0.0000,  -0.3147,   0.4876,   0.0000,   0.0000,  -0.4392,   0.0000,   0.0000],
            [-12.0354,   0.0000,   0.0000,  -0.4950,  -0.0000,  -0.3147,   0.0000,   0.0000,   0.2799,  -0.0000,   0.0000,   0.3625],
            [-16.2015,   0.1386,   0.0000,  -0.9025,  -0.0000,  -0.3147,   0.4876,   0.0000,   0.0000,  -0.0000,   0.0000,   0.3625],
            [-16.2015,   0.0000,   0.0000,  -0.4982,  -0.2632,  -0.0000,   0.0000,   0.0000,   0.2799,  -0.0000,   0.0000,   0.3625],
            [-11.1096,   0.0000,   0.0000,  -0.5081,  -0.2632,  -0.0000,   0.0000,   0.0000,   0.2799,  -0.0000,   0.2103,   0.0000],
            [-24.9966,   0.0000,   0.0000,  -0.8973,  -0.2632,  -0.0000,   0.4876,   0.0000,   0.0000,  -0.0000,   0.0000,   0.3625],
            ...,
            [-11.5725,   0.0000,   0.0000,  -0.4717,  -0.2632,  -0.0000,   0.0000,   0.0000,   0.2799,  -0.0000,   0.0000,   0.3625],
            [-18.0531,   0.0000,   1.2045,  -0.7701,  -0.0000,  -0.3147,   0.0000,   0.0000,   0.2799,  -0.0000,   0.2103,   0.0000],
            [-12.4983,   0.0000,   0.0000,  -0.5968,  -0.2632,  -0.0000,   0.0000,   0.3136,   0.0000,  -0.0000,   0.0000,   0.3625],
            [ -8.7951,   0.0000,   0.0000,  -0.7766,  -0.0000,  -0.3147,   0.4876,   0.0000,   0.0000,  -0.0000,   0.0000,   0.3625],
            [-11.1096,   0.1386,   0.4818,  -0.7229,  -0.0000,  -0.3147,   0.0000,   0.0000,   0.2799,  -0.0000,   0.0000,   0.3625],
            [-12.0354,   0.0000,   0.0000,  -0.7766,  -0.2632,  -0.0000,   0.4876,   0.0000,   0.0000,  -0.4392,   0.0000,   0.0000],
            [-14.8128,   0.0000,   0.0000,  -0.4905,  -0.2632,  -0.0000,   0.0000,   0.0000,   0.2799,  -0.0000,   0.2103,   0.0000]])

We can see we’ve got a problem here. The sums of each row will be dominated by the first column, which is `Age`, since that’s bigger on average than all the others.

Let’s make all the columns contain numbers from `0` to `1`, by dividing each column by its `max()`:

``` python
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals
```

As we see, that removes the problem of one column dominating all the others:

``` python
t_indep*coeffs
```

    tensor([[-0.1273,  0.0173,  0.0000, -0.0765, -0.2632, -0.0000,  0.0000,  0.0000,  0.2799, -0.0000,  0.0000,  0.3625],
            [-0.2199,  0.0173,  0.0000, -0.1551, -0.0000, -0.3147,  0.4876,  0.0000,  0.0000, -0.4392,  0.0000,  0.0000],
            [-0.1504,  0.0000,  0.0000, -0.0793, -0.0000, -0.3147,  0.0000,  0.0000,  0.2799, -0.0000,  0.0000,  0.3625],
            [-0.2025,  0.0173,  0.0000, -0.1446, -0.0000, -0.3147,  0.4876,  0.0000,  0.0000, -0.0000,  0.0000,  0.3625],
            [-0.2025,  0.0000,  0.0000, -0.0798, -0.2632, -0.0000,  0.0000,  0.0000,  0.2799, -0.0000,  0.0000,  0.3625],
            [-0.1389,  0.0000,  0.0000, -0.0814, -0.2632, -0.0000,  0.0000,  0.0000,  0.2799, -0.0000,  0.2103,  0.0000],
            [-0.3125,  0.0000,  0.0000, -0.1438, -0.2632, -0.0000,  0.4876,  0.0000,  0.0000, -0.0000,  0.0000,  0.3625],
            ...,
            [-0.1447,  0.0000,  0.0000, -0.0756, -0.2632, -0.0000,  0.0000,  0.0000,  0.2799, -0.0000,  0.0000,  0.3625],
            [-0.2257,  0.0000,  0.2008, -0.1234, -0.0000, -0.3147,  0.0000,  0.0000,  0.2799, -0.0000,  0.2103,  0.0000],
            [-0.1562,  0.0000,  0.0000, -0.0956, -0.2632, -0.0000,  0.0000,  0.3136,  0.0000, -0.0000,  0.0000,  0.3625],
            [-0.1099,  0.0000,  0.0000, -0.1244, -0.0000, -0.3147,  0.4876,  0.0000,  0.0000, -0.0000,  0.0000,  0.3625],
            [-0.1389,  0.0173,  0.0803, -0.1158, -0.0000, -0.3147,  0.0000,  0.0000,  0.2799, -0.0000,  0.0000,  0.3625],
            [-0.1504,  0.0000,  0.0000, -0.1244, -0.2632, -0.0000,  0.4876,  0.0000,  0.0000, -0.4392,  0.0000,  0.0000],
            [-0.1852,  0.0000,  0.0000, -0.0786, -0.2632, -0.0000,  0.0000,  0.0000,  0.2799, -0.0000,  0.2103,  0.0000]])

One thing you hopefully noticed is how amazingly cool this line of code is:

    t_indep = t_indep / vals

That is dividing a matrix by a vector – what on earth does that mean?!? The trick here is that we’re taking advantage of a technique in numpy and PyTorch (and many other languages, going all the way back to APL) called [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). In short, this acts as if there’s a separate copy of the vector for every row of the matrix, so it divides each row of the matrix by the vector. In practice, it doesn’t actually make any copies, and does the whole thing in a highly optimized way, taking full advantage of modern CPUs (or, indeed, GPUs, if we’re using them). Broadcasting is one of the most important techniques for making your code concise, maintainable, and fast, so it’s well worth studying and practicing.

We can now create predictions from our linear model, by adding up the rows of the product:

``` python
preds = (t_indep*coeffs).sum(axis=1)
```

Let’s take a look at the first few:

``` python
preds[:10]
```

    tensor([ 0.1927, -0.6239,  0.0979,  0.2056,  0.0968,  0.0066,  0.1306,  0.3476,  0.1613, -0.6285])

Of course, these predictions aren’t going to be any use, since our coefficients are random – they’re just a starting point for our gradient descent process.

To do gradient descent, we need a loss function. Taking the average error of the rows (i.e. the absolute value of the difference between the prediction and the dependent) is generally a reasonable approach:

``` python
loss = torch.abs(preds-t_dep).mean()
loss
```

    tensor(0.5382)

Now that we’ve tested out a way of calculating predictions, and loss, let’s pop them into functions to make life easier:

``` python
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

## Doing a gradient descent step

In this section, we’re going to do a single “epoch” of gradient descent manually. The only thing we’re going to automate is calculating gradients, because let’s face it that’s pretty tedious and entirely pointless to do by hand! To get PyTorch to calculate gradients, we’ll need to call `requires_grad_()` on our `coeffs` (if you’re not sure why, review the previous notebook, [How does a neural net really work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work), before continuing):

``` python
coeffs.requires_grad_()
```

    tensor([-0.4629,  0.1386,  0.2409, -0.2262, -0.2632, -0.3147,  0.4876,  0.3136,  0.2799, -0.4392,  0.2103,  0.3625], requires_grad=True)

Now when we calculate our loss, PyTorch will keep track of all the steps, so we’ll be able to get the gradients afterwards:

``` python
loss = calc_loss(coeffs, t_indep, t_dep)
loss
```

    tensor(0.5382, grad_fn=<MeanBackward0>)

Use `backward()` to ask PyTorch to calculate gradients now:

``` python
loss.backward()
```

Let’s see what they look like:

``` python
coeffs.grad
```

    tensor([-0.0106,  0.0129, -0.0041, -0.0484,  0.2099, -0.2132, -0.1212, -0.0247,  0.1425, -0.1886, -0.0191,  0.2043])

Note that each time we call `backward`, the gradients are actually *added* to whatever is in the `.grad` attribute. Let’s try running the above steps again:

``` python
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
coeffs.grad
```

    tensor([-0.0212,  0.0258, -0.0082, -0.0969,  0.4198, -0.4265, -0.2424, -0.0494,  0.2851, -0.3771, -0.0382,  0.4085])

As you see, our `.grad` values are have doubled. That’s because it added the gradients a second time. For this reason, after we use the gradients to do a gradient descent step, we need to set them back to zero.

We can now do one gradient descent step, and check that our loss decreases:

``` python
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))
```

    tensor(0.4945)

Note that `a.sub_(b)` subtracts `b` from `a` in-place. In PyTorch, any method that ends in `_` changes its object in-place. Similarly, `a.zero_()` sets all elements of a tensor to zero.

## Training the linear model

Before we begin training our model, we’ll need to ensure that we hold out a validation set for calculating our metrics (for details on this, see “[Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners#Test-and-validation-sets)”.

There’s lots of different ways we can do this. In the next notebook we’ll be comparing our approach here to what the fastai library does, so we’ll want to ensure we split the data in the same way. So let’s use `RandomSplitter` to get indices that will split our data into training and validation sets:

``` python
from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df)
```

Now we can apply those indicies to our independent and dependent variables:

``` python
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)
```

    (713, 178)

We’ll create functions for the three things we did manually above: updating `coeffs`, doing one full gradient descent step, and initilising `coeffs` to random numbers:

``` python
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

``` python
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")
```

``` python
def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()
```

We can now use these functions to train our model:

``` python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs
```

Let’s try it. Our loss will print at the end of every step, so we hope we’ll see it going down:

``` python
coeffs = train_model(18, lr=0.2)
```

    0.536; 0.502; 0.477; 0.454; 0.431; 0.409; 0.388; 0.367; 0.349; 0.336; 0.330; 0.326; 0.329; 0.304; 0.314; 0.296; 0.300; 0.289; 

It does!

Let’s take a look at the coefficients for each column:

``` python
def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))
show_coeffs()
```

    {'Age': tensor(-0.2694),
     'SibSp': tensor(0.0901),
     'Parch': tensor(0.2359),
     'LogFare': tensor(0.0280),
     'Sex_male': tensor(-0.3990),
     'Sex_female': tensor(0.2345),
     'Pclass_1': tensor(0.7232),
     'Pclass_2': tensor(0.4112),
     'Pclass_3': tensor(0.3601),
     'Embarked_C': tensor(0.0955),
     'Embarked_Q': tensor(0.2395),
     'Embarked_S': tensor(0.2122)}

## Measuring accuracy

The Kaggle competition is not, however, scored by absolute error (which is our loss function). It’s scored by *accuracy* – the proportion of rows where we correctly predict survival. Let’s see how accurate we were on the validation set. First, calculate the predictions:

``` python
preds = calc_preds(coeffs, val_indep)
```

We’ll assume that any passenger with a score of over `0.5` is predicted to survive. So that means we’re correct for each row where `preds>0.5` is the same as the dependent variable:

``` python
results = val_dep.bool()==(preds>0.5)
results[:16]
```

    tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False,  True,  True, False])

Let’s see what our average accuracy is:

``` python
results.float().mean()
```

    tensor(0.7865)

That’s not a bad start at all! We’ll create a function so we can calcuate the accuracy easy for other models we train:

``` python
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)
```

    tensor(0.7865)

## Using sigmoid

Looking at our predictions, there’s one obvious problem – some of our predictions of the probability of survival are `>1`, and some are `<0`:

``` python
preds[:28]
```

    tensor([ 0.8160,  0.1295, -0.0148,  0.1831,  0.1520,  0.1350,  0.7279,  0.7754,  0.3222,  0.6740,  0.0753,  0.0389,  0.2216,  0.7631,
             0.0678,  0.3997,  0.3324,  0.8278,  0.1078,  0.7126,  0.1023,  0.3627,  0.9937,  0.8050,  0.1153,  0.1455,  0.8652,  0.3425])

To fix this, we should pass every prediction through the *sigmoid function*, which has a minimum at zero and maximum at one, and is defined as follows:

``` python
import sympy
sympy.plot("1/(1+exp(-x))", xlim=(-5,5));
```

![](041_linear-model-and-neural-net-from-scratch_files/figure-html/cell-50-output-1.png)

PyTorch already defines that function for us, so we can modify `calc_preds` to use it:

``` python
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))
```

Let’s train a new model now, using this updated function to calculate predictions:

``` python
coeffs = train_model(lr=100)
```

    0.510; 0.327; 0.294; 0.207; 0.201; 0.199; 0.198; 0.197; 0.196; 0.196; 0.196; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 

The loss has improved by a lot. Let’s check the accuracy:

``` python
acc(coeffs)
```

    tensor(0.8258)

That’s improved too! Here’s the coefficients of our trained model:

``` python
show_coeffs()
```

    {'Age': tensor(-1.5061),
     'SibSp': tensor(-1.1575),
     'Parch': tensor(-0.4267),
     'LogFare': tensor(0.2543),
     'Sex_male': tensor(-10.3320),
     'Sex_female': tensor(8.4185),
     'Pclass_1': tensor(3.8389),
     'Pclass_2': tensor(2.1398),
     'Pclass_3': tensor(-6.2331),
     'Embarked_C': tensor(1.4771),
     'Embarked_Q': tensor(2.1168),
     'Embarked_S': tensor(-4.7958)}

These coefficients seem reasonable – in general, older people and males were less likely to survive, and first class passengers were more likely to survive.

## Submitting to Kaggle

Now that we’ve got a trained model, we can prepare a submission to Kaggle. To do that, first we need to read the test set:

``` python
tst_df = pd.read_csv(f'{path}/test.csv')
```

In this case, it turns out that the test set is missing `Fare` for one passenger. We’ll just fill it with `0` to avoid problems:

``` python
tst_df['Fare'] = tst_df.Fare.fillna(0)
```

Now we can just copy the same steps we did to our training set and do the same exact things on our test set to preprocess the data:

``` python
tst_df.fillna(modes, inplace=True)
tst_df['LogFare'] = np.log(tst_df['Fare']+1)
tst_df = pd.get_dummies(tst_df, columns=["Sex","Pclass","Embarked"])

tst_df[indep_cols] = tst_df[indep_cols].astype('float64') 

tst_indep = tensor(tst_df[indep_cols].values, dtype=torch.float)
tst_indep = tst_indep / vals
```

Let’s calculate our predictions of which passengers survived in the test set:

``` python
tst_df['Survived'] = (calc_preds(tst_indep, coeffs)>0.5).int()
```

The sample submission on the Kaggle competition site shows that we’re expected to upload a CSV with just `PassengerId` and `Survived`, so let’s create that and save it:

``` python
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)
```

We can check the first few rows of the file to make sure it looks reasonable:

``` python
!head sub.csv
```

    PassengerId,Survived
    892,0
    893,0
    894,0
    895,0
    896,0
    897,0
    898,1
    899,0
    900,1

When you click “save version” in Kaggle, and wait for the notebook to run, you’ll see that `sub.csv` appears in the “Data” tab. Clicking on that file will show a *Submit* button, which allows you to submit to the competition.

## Using matrix product

We can make things quite a bit neater…

Take a look at the inner-most calculation we’re doing to get the predictions:

``` python
(val_indep*coeffs).sum(axis=1)
```

    tensor([ 12.3288, -14.8119, -15.4540, -13.1513, -13.3512, -13.6469,   3.6248,   5.3429, -22.0878,   3.1233, -21.8742, -15.6421, -21.5504,
              3.9393, -21.9190, -12.0010, -12.3775,   5.3550, -13.5880,  -3.1015, -21.7237, -12.2081,  12.9767,   4.7427, -21.6525, -14.9135,
             -2.7433, -12.3210, -21.5886,   3.9387,   5.3890,  -3.6196, -21.6296, -21.8454,  12.2159,  -3.2275, -12.0289,  13.4560, -21.7230,
             -3.1366, -13.2462, -21.7230, -13.6831,  13.3092, -21.6477,  -3.5868, -21.6854, -21.8316, -14.8158,  -2.9386,  -5.3103, -22.2384,
            -22.1097, -21.7466, -13.3780, -13.4909, -14.8119, -22.0690, -21.6666, -21.7818,  -5.4439, -21.7407, -12.6551, -21.6671,   4.9238,
            -11.5777, -13.3323, -21.9638, -15.3030,   5.0243, -21.7614,   3.1820, -13.4721, -21.7170, -11.6066, -21.5737, -21.7230, -11.9652,
            -13.2382, -13.7599, -13.2170,  13.1347, -21.7049, -21.7268,   4.9207,  -7.3198,  -5.3081,   7.1065,  11.4948, -13.3135, -21.8723,
            -21.7230,  13.3603, -15.5670,   3.4105,  -7.2857, -13.7197,   3.6909,   3.9763, -14.7227, -21.8268,   3.9387, -21.8743, -21.8367,
            -11.8518, -13.6712, -21.8299,   4.9440,  -5.4471, -21.9666,   5.1333,  -3.2187, -11.6008,  13.7920, -21.7230,  12.6369,  -3.7268,
            -14.8119, -22.0637,  12.9468, -22.1610,  -6.1827, -14.8119,  -3.2838, -15.4540, -11.6950,  -2.9926,  -3.0110, -21.5664, -13.8268,
              7.3426, -21.8418,   5.0744,   5.2582,  13.3415, -21.6289, -13.9898, -21.8112,  -7.3316,   5.2296, -13.4453,  12.7891, -22.1235,
            -14.9625,  -3.4339,   6.3089, -21.9839,   3.1968,   7.2400,   2.8558,  -3.1187,   3.7965,   5.4667, -15.1101, -15.0597, -22.9391,
            -21.7230,  -3.0346, -13.5206, -21.7011,  13.4425,  -7.2690, -21.8335, -12.0582,  13.0489,   6.7993,   5.2160,   5.0794, -12.6957,
            -12.1838,  -3.0873, -21.6070,   7.0744, -21.7170, -22.1001,   6.8159, -11.6002, -21.6310])

Multiplying elements together and then adding across rows is identical to doing a matrix-vector product! Python uses the `@` operator to indicate matrix products, and is supported by PyTorch tensors. Therefore, we can replicate the above calculate more simply like so:

``` python
val_indep@coeffs
```

    tensor([ 12.3288, -14.8119, -15.4540, -13.1513, -13.3511, -13.6468,   3.6248,   5.3429, -22.0878,   3.1233, -21.8742, -15.6421, -21.5504,
              3.9393, -21.9190, -12.0010, -12.3775,   5.3550, -13.5880,  -3.1015, -21.7237, -12.2081,  12.9767,   4.7427, -21.6525, -14.9135,
             -2.7433, -12.3210, -21.5886,   3.9387,   5.3890,  -3.6196, -21.6296, -21.8454,  12.2159,  -3.2275, -12.0289,  13.4560, -21.7230,
             -3.1366, -13.2462, -21.7230, -13.6831,  13.3092, -21.6477,  -3.5868, -21.6854, -21.8316, -14.8158,  -2.9386,  -5.3103, -22.2384,
            -22.1097, -21.7466, -13.3780, -13.4909, -14.8119, -22.0690, -21.6666, -21.7818,  -5.4439, -21.7407, -12.6551, -21.6671,   4.9238,
            -11.5777, -13.3323, -21.9638, -15.3030,   5.0243, -21.7614,   3.1820, -13.4721, -21.7170, -11.6066, -21.5737, -21.7230, -11.9652,
            -13.2382, -13.7599, -13.2170,  13.1347, -21.7049, -21.7268,   4.9207,  -7.3198,  -5.3081,   7.1065,  11.4948, -13.3135, -21.8723,
            -21.7230,  13.3603, -15.5670,   3.4105,  -7.2857, -13.7197,   3.6909,   3.9763, -14.7227, -21.8268,   3.9387, -21.8743, -21.8367,
            -11.8518, -13.6712, -21.8299,   4.9440,  -5.4471, -21.9666,   5.1333,  -3.2187, -11.6008,  13.7920, -21.7230,  12.6369,  -3.7268,
            -14.8119, -22.0637,  12.9468, -22.1610,  -6.1827, -14.8119,  -3.2838, -15.4540, -11.6950,  -2.9926,  -3.0110, -21.5664, -13.8268,
              7.3426, -21.8418,   5.0744,   5.2582,  13.3415, -21.6289, -13.9898, -21.8112,  -7.3316,   5.2296, -13.4453,  12.7891, -22.1235,
            -14.9625,  -3.4339,   6.3089, -21.9839,   3.1968,   7.2400,   2.8558,  -3.1187,   3.7965,   5.4667, -15.1101, -15.0597, -22.9391,
            -21.7230,  -3.0346, -13.5206, -21.7011,  13.4425,  -7.2690, -21.8335, -12.0582,  13.0489,   6.7993,   5.2160,   5.0794, -12.6957,
            -12.1838,  -3.0873, -21.6070,   7.0744, -21.7170, -22.1001,   6.8159, -11.6002, -21.6310])

It also turns out that this is much faster, because matrix products in PyTorch are very highly optimised.

Let’s use this to replace how `calc_preds` works:

``` python
def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)
```

In order to do matrix-matrix products (which we’ll need in the next section), we need to turn `coeffs` into a column vector (i.e. a matrix with a single column), which we can do by passing a second argument `1` to `torch.rand()`, indicating that we want our coefficients to have one column:

``` python
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()
```

We’ll also need to turn our dependent variable into a column vector, which we can do by indexing the column dimension with the special value `None`, which tells PyTorch to add a new dimension in this position:

``` python
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]
```

We can now train our model as before and confirm we get identical outputs…:

``` python
coeffs = train_model(lr=100)
```

    0.512; 0.323; 0.290; 0.205; 0.200; 0.198; 0.197; 0.197; 0.196; 0.196; 0.196; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 

…and identical accuracy:

``` python
acc(coeffs)
```

    tensor(0.8258)

## A neural network

We’ve now got what we need to implement our neural network.

First, we’ll need to create coefficients for each of our layers. Our first set of coefficients will take our `n_coeff` inputs, and create `n_hidden` outputs. We can choose whatever `n_hidden` we like – a higher number gives our network more flexibility, but makes it slower and harder to train. So we need a matrix of size `n_coeff` by `n_hidden`. We’ll divide these coefficients by `n_hidden` so that when we sum them up in the next layer we’ll end up with similar magnitude numbers to what we started with.

Then our second layer will need to take the `n_hidden` inputs and create a single output, so that means we need a `n_hidden` by `1` matrix there. The second layer will also need a constant term added.

``` python
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()
```

Now we have our coefficients, we can create our neural net. The key steps are the two matrix products, `indeps@l1` and `res@l2` (where `res` is the output of the first layer). The first layer output is passed to `F.relu` (that’s our non-linearity), and the second is passed to `torch.sigmoid` as before.

``` python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
```

Finally, now that we have more than one set of coefficients, we need to add a loop to update each one:

``` python
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

That’s it – we’re now ready to train our model!

``` python
coeffs = train_model(lr=1.4)
```

    0.543; 0.532; 0.520; 0.505; 0.487; 0.466; 0.439; 0.407; 0.373; 0.343; 0.319; 0.301; 0.286; 0.274; 0.264; 0.256; 0.250; 0.245; 0.240; 0.237; 0.234; 0.231; 0.229; 0.227; 0.226; 0.224; 0.223; 0.222; 0.221; 0.220; 

``` python
coeffs = train_model(lr=20)
```

    0.543; 0.400; 0.260; 0.390; 0.221; 0.211; 0.197; 0.195; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 

It’s looking good – our loss is lower than before. Let’s see if that translates to a better result on the validation set:

``` python
acc(coeffs)
```

    tensor(0.8258)

In this case our neural net isn’t showing better results than the linear model. That’s not surprising; this dataset is very small and very simple, and isn’t the kind of thing we’d expect to see neural networks excel at. Furthermore, our validation set is too small to reliably see much accuracy difference. But the key thing is that we now know exactly what a real neural net looks like!

## Deep learning

The neural net in the previous section only uses one hidden layer, so it doesn’t count as “deep” learning. But we can use the exact same technique to make our neural net deep, by adding more matrix multiplications.

First, we’ll need to create additional coefficients for each layer:

``` python
def init_coeffs():
    hiddens = [10, 10]  # <-- set this to the size of each hidden layer you want
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts
```

You’ll notice here that there’s a lot of messy constants to get the random numbers in just the right ranges. When you train the model in a moment, you’ll see that the tiniest changes to these initialisations can cause our model to fail to train at all! This is a key reason that deep learning failed to make much progress in the early days – it’s very finicky to get a good starting point for our coefficients. Nowadays, we have ways to deal with that, which we’ll learn about in other notebooks.

Our deep learning `calc_preds` looks much the same as before, but now we loop through each layer, instead of listing them separately:

``` python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)
```

We also need a minor update to `update_coeffs` since we’ve got `layers` and `consts` separated now:

``` python
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

Let’s train our model…

``` python
coeffs = train_model(lr=4)
```

    0.521; 0.483; 0.427; 0.379; 0.379; 0.379; 0.379; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.377; 0.376; 0.371; 0.333; 0.239; 0.224; 0.208; 0.204; 0.203; 0.203; 0.207; 0.197; 0.196; 0.195; 

…and check its accuracy:

``` python
acc(coeffs)
```

    tensor(0.8258)

## Final thoughts

It’s actually pretty cool that we’ve managed to create a real deep learning model from scratch and trained it to get over 80% accuracy on this task, all in the course of a single notebook!

The “real” deep learning models that are used in research and industry look very similar to this, and in fact if you look inside the source code of any deep learning model you’ll recognise the basic steps are the same.

The biggest differences in practical models to what we have above are:

- How initialisation and normalisation is done to ensure the model trains correctly every time
- Regularization (to avoid over-fitting)
- Modifying the neural net itself to take advantage of knowledge of the problem domain
- Doing gradient descent steps on smaller batches, rather than the whole dataset.

I’ll be adding notebooks about all these later, and will add links here once they’re ready.

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. (BTW, be sure you’re looking at my [original notebook here](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch) when you do that, and are not on your own copy of it, otherwise your upvote won’t get counted!) And if you have any questions or comments, please pop them below – I read every comment I receive!

Back to top
