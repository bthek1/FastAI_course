# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_Deployment.ipynb.

# %% auto 0
__all__ = ['learn', 'download_pic', 'create_searches_folder', 'download_search_images', 'verify_pics', 'resize_pics',
           'create_data_folder']

# %% ../nbs/01_Deployment.ipynb 10
#|code-fold: true
def download_pic(name):   
    # File path of the image
    image_path = f'{name}.jpg'

    # Check if the image file exists
    if os.path.exists(image_path):
        print("Image file exists.")
    else:
        print("Image file does not exist.")
        download_url(
            search_images_ddg(f'{name}',
            max_images=1)[0], f'{name}.jpg',
            show_progress=False
        )

    return Image.open(f'{name}.jpg').to_thumb(256,256)

# %% ../nbs/01_Deployment.ipynb 12
#|code-fold: true
def create_searches_folder(folder_path, searches):
    for i in searches:
        dest = (folder_path/i)
        dest.mkdir(exist_ok=True, parents=True)
        print(f'created {i} folder')

# %% ../nbs/01_Deployment.ipynb 13
#|code-fold: true
def download_search_images(folder_path, searches, amount):
    for j in searches:
        print(f"downloading images for: {j}")
        download_images(
            folder_path/j,
            urls=search_images_ddg(f'{j} photo', amount),
            n_workers=16
        )

# %% ../nbs/01_Deployment.ipynb 14
#|code-fold: true
def verify_pics(folder_path):
    failed = verify_images(get_image_files(folder_path))
    failed.map(Path.unlink)
    print(f"Number of images failed: {len(failed)}")

# %% ../nbs/01_Deployment.ipynb 15
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

# %% ../nbs/01_Deployment.ipynb 16
#|code-fold: true
def create_data_folder(folder_path, searches, amount):
    if os.path.exists(folder_path):
        print(f"Folder already exists: {folder_path}")
    else:   
        create_searches_folder(folder_path, searches)
        download_search_images(folder_path, searches, amount)
        verify_pics(folder_path)
        resize_pics(folder_path, searches)


# %% ../nbs/01_Deployment.ipynb 30
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(7)
