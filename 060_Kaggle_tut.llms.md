# Kaggle tutorial

Kagglle tutorial

Author

Benedict Thekkel

In [Iterate Like a Grandmaster](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster) I explained that when working on a Kaggle project:

> …the focus generally should be two things:
>
> 1.  Creating an effective validation set
> 2.  Iterating rapidly to find changes which improve results on the validation set.

Here I’m going to go further, showing the process I used to tackle the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) competition, leading to four submissions in a row which all were (at the time of submission) in 1st place, each one more accurate than the last. You might be surprised to discover that the process of doing this was nearly entirely mechanistic and didn’t involve any consideration of the actual data or evaluation details at all.

This notebook is the first in a series showing every step of the process. At the end of this notebook we’ll have a basic submission; by the end of the series you’ll see how I got to the top of the table!:

![](https://user-images.githubusercontent.com/346999/174389920-60d67ead-0f36-41d0-9649-e23b08720c8a.png)

As a special extra, I’m also opening up early a selection of “walkthru” videos that we’ve been preparing for the new upcoming fast.ai course. Each day I do a walkthru with fast.ai fellows and registered students, and we record those sessions. They’ll all be released at the same time as the next course (probably August 2022), but I’m releasing the ones covering this competition right now! Here they are:

- [Walkthru 8](https://www.youtube.com/watch?v=-Scs4gbwWXg)
- [Walkthru 9](https://www.youtube.com/watch?v=EK5wJRzffas)
- [Walkthru 10](https://youtu.be/zhBRynq9Yvo)
- [Walkthru 11](https://youtu.be/j-zMF2VirA8)
- [Walkthru 12](https://youtu.be/GuCkpjXHdTc)
- [Walkthru 13](https://youtu.be/INrkhUGCXHg)

When you’re done with this notebook, take a look at [part 2 of the series](https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2/).

## Getting set up

First, we’ll get the data. I’ve just created a new library called [fastkaggle](https://fastai.github.io/fastkaggle/) which has a few handy features, including getting the data for a competition correctly regardless of whether we’re running on Kaggle or elsewhere. Note you’ll need to first accept the competition rules and join the competition, and you’ll need your kaggle API key file `kaggle.json` downloaded if you’re running this somewhere other than on Kaggle. `setup_comp` is the function we use in `fastkaggle` to grab the data, and install or upgrade our needed python modules when we’re running on Kaggle:

``` python
from nbdevAuto.functions import kaggle_competition_download
from pathlib import Path

datapath = Path('./Data')
name = 'paddy-disease-classification'
path = Path(f'{datapath}/{name}')

kaggle_competition_download(name, datapath)
```

    Downloading paddy-disease-classification.zip to Data/paddy-disease-classification

    100%|███████████████████████████████████████████████████████████████████████████████████████████| 1.02G/1.02G [02:55<00:00, 6.25MB/s]

``` python
path
```

    PosixPath('Data/paddy-disease-classification')

Now we can import the stuff we’ll need from fastai, set a seed (for reproducibility – just for the purposes of making this notebook easier to write; I don’t recommend doing that in your own analysis however) and check what’s in the data:

``` python
from fastai.vision.all import *
set_seed(42)

path.ls()
```

    (#5) [Path('Data/paddy-disease-classification/paddy-disease-classification.zip'),Path('Data/paddy-disease-classification/train_images'),Path('Data/paddy-disease-classification/train.csv'),Path('Data/paddy-disease-classification/sample_submission.csv'),Path('Data/paddy-disease-classification/test_images')]

## Looking at the data

The images are in `train_images`, so let’s grab a list of all of them:

``` python
trn_path = path/'train_images'
files = get_image_files(trn_path)
```

…and take a look at one:

``` python
img = PILImage.create(files[0])
print(img.size)
img.to_thumb(128)
```

    (480, 640)

![](060_Kaggle_tut_files/figure-html/cell-6-output-2.png)

Looks like the images might be 480x640 – let’s check all their sizes. This is faster if we do it in parallel, so we’ll use fastcore’s `parallel` for this:

``` python
from fastcore.parallel import *
```

``` python
def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=16)
pd.Series(sizes).value_counts()
```

    (480, 640)    10403
    (640, 480)        4
    dtype: int64

They’re nearly all the same size, except for a few. Because of those few, however, we’ll need to make sure we always resize each image to common dimensions first, otherwise fastai won’t be able to create batches. For now, we’ll just squish them to 480x480 images, and then once they’re in batches we do a random resized crop down to a smaller size, along with the other default fastai augmentations provided by `aug_transforms`. We’ll start out with small resized images, since we want to be able to iterate quickly:

``` python
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize((480,360), method='squish'),
    batch_tfms=aug_transforms(size=(128,128), min_scale=0.75))

dls.show_batch(max_n=6)
```

![](060_Kaggle_tut_files/figure-html/cell-9-output-1.png)

## Our first model

Let’s create a model. To pick an architecture, we should look at the options in [The best vision models for fine-tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning). I like the looks of `resnet26d`, which is the fastest resolution-independent model which gets into the top-15 lists there.

``` python
import timm
```

``` python
??timm
```

    Type:        module
    String form: <module 'timm' from '/home/ben/mambaforge/envs/cfast/lib/python3.11/site-packages/timm/__init__.py'>
    File:        ~/mambaforge/envs/cfast/lib/python3.11/site-packages/timm/__init__.py
    Source:     
    from .version import __version__
    from .layers import is_scriptable, is_exportable, set_scriptable, set_exportable
    from .models import create_model, list_models, list_pretrained, is_model, list_modules, model_entrypoint, \
        is_model_pretrained, get_pretrained_cfg, get_pretrained_cfg_value

``` python
timm.list_models('convnext*')
```

    ['convnext_atto',
     'convnext_atto_ols',
     'convnext_base',
     'convnext_femto',
     'convnext_femto_ols',
     'convnext_large',
     'convnext_large_mlp',
     'convnext_nano',
     'convnext_nano_ols',
     'convnext_pico',
     'convnext_pico_ols',
     'convnext_small',
     'convnext_tiny',
     'convnext_tiny_hnf',
     'convnext_xlarge',
     'convnext_xxlarge',
     'convnextv2_atto',
     'convnextv2_base',
     'convnextv2_femto',
     'convnextv2_huge',
     'convnextv2_large',
     'convnextv2_nano',
     'convnextv2_pico',
     'convnextv2_small',
     'convnextv2_tiny']

``` python
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
```

Let’s see what the learning rate finder shows:

``` python
learn.lr_find(suggest_funcs=(valley, slide))
```

    SuggestedLRs(valley=0.0014454397605732083, slide=0.0014454397605732083)

![](060_Kaggle_tut_files/figure-html/cell-14-output-4.png)

[`lr_find`](https://bthek1.github.io/fastAIcourse/learner.html#lr_find) generally recommends rather conservative learning rates, to ensure that your model will train successfully. I generally like to push it a bit higher if I can. Let’s train a few epochs and see how it looks:

``` python
learn.fine_tune(5)
```

| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|-------|
| 0     | 2.014854   | 1.233058   | 0.388275   | 01:01 |

| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|-------|
| 0     | 1.348343   | 0.917006   | 0.298895   | 01:03 |
| 1     | 1.131407   | 0.720297   | 0.234503   | 01:07 |
| 2     | 0.927136   | 0.603111   | 0.189332   | 01:15 |
| 3     | 0.790826   | 0.523225   | 0.161941   | 01:16 |
| 4     | 0.710640   | 0.512586   | 0.159058   | 01:12 |

We’re now ready to build our first submission. Let’s take a look at the sample Kaggle provided to see what it needs to look like:

``` python
arch = 'convnext_small'
```

``` python
learn2 = vision_learner(dls, arch, metrics=error_rate, path='.').to_fp16()
```

``` python
learn2.fine_tune(1)
```

| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|-------|
| 0     | 1.611437   | 0.826959   | 0.266699   | 04:09 |

| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|-------|
| 0     | 0.785174   | 0.426409   | 0.135512   | 07:11 |

## Submitting to Kaggle

``` python
ss = pd.read_csv(path/'sample_submission.csv')
ss
```

|      | image_id   | label |
|------|------------|-------|
| 0    | 200001.jpg | NaN   |
| 1    | 200002.jpg | NaN   |
| 2    | 200003.jpg | NaN   |
| 3    | 200004.jpg | NaN   |
| 4    | 200005.jpg | NaN   |
| ...  | ...        | ...   |
| 3464 | 203465.jpg | NaN   |
| 3465 | 203466.jpg | NaN   |
| 3466 | 203467.jpg | NaN   |
| 3467 | 203468.jpg | NaN   |
| 3468 | 203469.jpg | NaN   |

3469 rows × 2 columns

OK so we need a CSV containing all the test images, in alphabetical order, and the predicted label for each one. We can create the needed test set using fastai like so:

``` python
tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)
```

We can now get the probabilities of each class, and the index of the most likely class, from this test set (the 2nd thing returned by `get_preds` are the targets, which are blank for a test set, so we discard them):

``` python
probs,_,idxs = learn2.get_preds(dl=tst_dl, with_decoded=True)
idxs
```

    tensor([7, 8, 6,  ..., 8, 7, 5])

These need to be mapped to the names of each of these diseases, these names are stored by fastai automatically in the `vocab`:

``` python
dls.vocab
```

    ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

We can create an apply this mapping using pandas:

``` python
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results
```

    0              hispa
    1             normal
    2       downy_mildew
    3              blast
    4              blast
                ...     
    3464      dead_heart
    3465           hispa
    3466          normal
    3467           hispa
    3468      dead_heart
    Name: idxs, Length: 3469, dtype: object

Kaggle expects the submission as a CSV file, so let’s save it, and check the first few lines:

``` python
ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv
```

    image_id,label
    200001.jpg,hispa
    200002.jpg,normal
    200003.jpg,downy_mildew
    200004.jpg,blast
    200005.jpg,blast
    200006.jpg,hispa
    200007.jpg,dead_heart
    200008.jpg,brown_spot
    200009.jpg,hispa

Let’s submit this to kaggle. We can do it from the notebook if we’re running on Kaggle, otherwise we can use the API:

``` python
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'initial rn26d 128px', name)
```

    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)
    Cell In[29], line 1
    ----> 1 if not iskaggle:
          2     from kaggle import api
          3     api.competition_submit_cli('subm.csv', 'initial rn26d 128px', name)

    NameError: name 'iskaggle' is not defined

Success! We successfully created a submission.

## Conclusion

Our initial submission is not very good (top 80% of teams) but it only took a minute to train. The important thing is that we have a good starting point to iterate from, and we can do rapid iterations. Every step from loading the data to creating the model to submitting to Kaggle is all automated and runs quickly.

Therefore, we can now try lots of things quickly and easily and use those experiments to improve our results. In the next notebook, we’ll do exactly that! So if you’re ready, take a look at [part 2 of the series](https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2/).

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. And if you have any questions or comments, please pop them below – I read every comment I receive!

## Addendum

`fastkaggle` also provides a function that pushes a notebook to Kaggle Notebooks. I wrote this notebook on my own machine, and pushed it to Kaggle from there – here’s the command I used:

``` python
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```

``` python
if not iskaggle:
    push_notebook('Benson Thekkel', '060_Kaggle_tut',
                  title='060_Kaggle_tut1',
                  file='060_Kaggle_tut.ipynb',
                  competition=name, private=False, gpu=True)
```

    Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.
    Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/bensonthekkel/060-kaggle-tut1

Back to top
