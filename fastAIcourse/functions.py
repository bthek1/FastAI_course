# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/900_Functions.ipynb.

# %% auto 0
__all__ = ['read_from_file', 'download_pic', 'create_searches_folder', 'download_search_image', 'download_search_images',
           'verify_pics', 'resize_pics', 'create_data_folder', 'classify_images']

# %% ../nbs/900_Functions.ipynb 3
#|code-fold: true
from fastbook import search_images_ddg
from fastdownload import download_url
from fastai.vision.all import *
import os
import shutil

# %% ../nbs/900_Functions.ipynb 5
#|code-fold: true
def read_from_file(file_path):
    countries = ()
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace and newline characters
            country = line.strip()
            # Add the country to the tuple
            countries += (country,)

    return countries

# %% ../nbs/900_Functions.ipynb 7
#|code-fold: true
def download_pic(
    image:str, #image name
    n_images:int=1,
    name:str='', #name name
    folder:str='',   # File path of the image
    show_progress:bool=False,
    recreate:bool=False
): 
    'Downloads the image into the folder provided and displays it'
    assert isinstance(image, str), "image must be a str."
    assert isinstance(name, str), "name must be a str."
    assert isinstance(folder, str), "folder must be a str."
    assert isinstance(n_images, int), "n_images must be an integer."
    assert isinstance(show_progress, bool), "show_progress must be a bool."
    assert isinstance(recreate, bool), "recreate must be a bool."
    

    if folder == '': folder = '.'
    if name == '': name = image

    image_path = f'{folder}/{name}{0}.jpg'
    # Check if the image file exists
    if recreate is False and os.path.exists(image_path):
        print("Image file exists.")
    else:
        search_links = search_images_ddg(
                        f'{image}',
                        max_images=n_images
        )

        for i in range(n_images): 
            try:
                image_path = f'{folder}/{name}{i}.jpg' 
                print(f"Downloading image_path.{i}")
                download_url(
                    search_links[i], image_path,
                    show_progress=show_progress
                )
            except Exception as e:
                # Code to handle any unhandled exceptions
                print("An error occurred:", e)

    return Image.open(image_path).to_thumb(256,256)

# %% ../nbs/900_Functions.ipynb 9
#|code-fold: true
def create_searches_folder(folder_path, searches):
    for i in searches:
        dest = (folder_path/i)
        dest.mkdir(exist_ok=True, parents=True)
        print(f'created {i} folder')

# %% ../nbs/900_Functions.ipynb 10
#|code-fold: true
def download_search_image(folder_path, item, before, after, amount):
    imgAmount = amount
    try:
        urls=search_images_ddg(f'{before}{item}{after}', imgAmount)
        print(f"downloading {imgAmount} images for:{before}{item}{after}")

        download_images(
        folder_path/item,
        urls=urls,
        n_workers=16
        )
        
    except Exception as e:
        # Code to handle any unhandled exceptions
        print(f"Error with {imgAmount} images of {before}{item}{after}:", e)
        imgAmount -= 20
        if imgAmount > 0: download_search_image(folder_path, item, before, after, imgAmount)

        


# %% ../nbs/900_Functions.ipynb 11
#|code-fold: true
def download_search_images(folder_path, searches, before, after, amount):
    for item in searches:
        imgAmount = amount
        download_search_image(folder_path, item, before, after, amount)
            


# %% ../nbs/900_Functions.ipynb 12
#|code-fold: true
def verify_pics(folder_path):
    failed = verify_images(get_image_files(folder_path))
    failed.map(Path.unlink)
    print(f"Number of images failed: {len(failed)}")

# %% ../nbs/900_Functions.ipynb 13
#|code-fold: true
def resize_pics(folder_path, searches):
    for k in searches:
        resize_images(
            folder_path/k,
            max_size=400,
            dest=folder_path/k,
            max_workers=8
        )
        print(f"resizing images for: {k}")

# %% ../nbs/900_Functions.ipynb 14
#|code-fold: true
def create_data_folder(
    folder_path:str,
    searches:tuple,
    before:str='',
    after:str='',
    amount:int=200,
    recreate:bool=False
):
    'generate image data'
    assert isinstance(searches, tuple), "searches must be a list."
    assert isinstance(amount, int), "amount must be an int."
    assert isinstance(recreate, bool), "recreate must be a bool."
    assert isinstance(before, str), "before must be a str."
    assert isinstance(after, str), "after must be a str."
    
    if recreate is False and os.path.exists(folder_path):
        print(f"Folder already exists: {folder_path}") 
    else:   
        if recreate is True and os.path.exists(folder_path): 
            shutil.rmtree(folder_path)
        create_searches_folder(folder_path, searches)
        download_search_images(folder_path, searches, before, after, amount)
        verify_pics(folder_path)
        resize_pics(folder_path, searches)

        

# %% ../nbs/900_Functions.ipynb 16
#|code-fold: true
def classify_images(learn, img):
    'image classifer'
    categories = learn.dls.vocab
    pred,idx,probs = learn.predict(PILImage.create(img))
    rounded_probs = [round(float(prob*100), 5) for prob in probs]
    return dict(zip(categories, rounded_probs))
