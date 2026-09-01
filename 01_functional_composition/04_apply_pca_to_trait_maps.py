"""Apply the published 10-trait PCA pixel-wise to the 30 m trait maps.

Purpose
-------
Project every 30 m pixel of the contiguous-US trait maps onto the three
functional-composition axes (PC1-PC3) fitted in 01_fit_pca.ipynb. The maps are
processed in WINDOW_SIZE x WINDOW_SIZE pixel tiles in parallel; each tile is
written as a 3-band Int16 GeoTIFF and 05_mosaic_pc_tiles.py joins them.

This step is only needed to regenerate the 30 m PC maps. The 0.05 degree PC maps
used for Fig. 2d-g (../data/grid_005_layers/PC_scores/PCA_all_PC{1,2,3}_005_clip.tif)
are already in the archive; they were aggregated from the 30 m mosaic in ArcGIS.

Inputs
------
../data/trait_maps_30m/<Trait>.tif
    Nine leaf-trait maps (Carbon, Cellulose, ChlorophyllsArea, EWT, Lignin,
    Nitrogen, NSC, Phenolics, SLA): 30 m, UInt16, NoData 0. Distributed in the
    archive.
../data/trait_maps_30m/canopy_height.tif
    NOT distributed. Build it from the Lang et al. (2023) global canopy height
    map: resample to the trait-map grid (same CRS, extent and 30 m pixels as the
    trait maps), multiply by 100, store as Int16 with NoData 32767.
../data/pca_model/scaler_rp_10traits.pkl, ../data/pca_model/PCA_model_rp_sa_10traits.pkl
    The fitted scaler and PCA (01_fit_pca.ipynb writes identical copies to ./results/).

Processing (per pixel)
----------------------
A pixel is kept only if none of the nine trait bands is 0 and canopy height is
not 32767. Canopy height + 1 (as in the PCA training data), natural log of all
ten raw integer values, StandardScaler, PCA. Within each tile every PC score is
multiplied by 1000 / (max - min) of that tile and cast to Int16.

Outputs
-------
./results/pc_tiles/PCA_<row>_<col>.tif
    3-band Int16 tiles (PC1, PC2, PC3), NoData -9999, LZW. Existing tiles are
    skipped, so the script can be re-run to resume.

Run from this folder:  python 04_apply_pca_to_trait_maps.py
"""
import concurrent.futures
import os
import pickle

import numpy as np
import pandas as pd
import rasterio
from rasterio import windows

TRAIT_DIR = '../data/trait_maps_30m'
MODEL_DIR = '../data/pca_model'
OUT_DIR = './results/pc_tiles'

# same order as the PCA training columns; canopy_height must stay last (different NoData)
TRAIT_LIST = ['Carbon', 'Cellulose', 'ChlorophyllsArea', 'EWT', 'Lignin',
              'Nitrogen', 'NSC', 'Phenolics', 'SLA', 'canopy_height']
WINDOW_SIZE = 1000
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def PCA_transform(img, pca, scaler):
    """img: (n_traits, n_pixels) integer array -> (n_pixels, 3) Int16 PC scores."""
    non_index = ~(
        np.concatenate([img[:-1, ] == 0, (img[-1,] == 32767).reshape(1, -1)], axis=0)
    ).any(axis=0)
    img_filter = img[:, non_index]
    img_filter = img_filter.T
    img_filter[:, -1] = img_filter[:, -1] + 1
    X = np.log(img_filter)
    # column order = the trait names the scaler was fitted with (silences sklearn's feature-name warning)
    X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    X = scaler.transform(X)
    X_scale_reduced = pca.transform(X)
    scalePC1 = X_scale_reduced[:, 0] * 1000 / (X_scale_reduced[:, 0].max() - X_scale_reduced[:, 0].min())
    scalePC2 = X_scale_reduced[:, 1] * 1000 / (X_scale_reduced[:, 1].max() - X_scale_reduced[:, 1].min())
    scalePC3 = X_scale_reduced[:, 2] * 1000 / (X_scale_reduced[:, 2].max() - X_scale_reduced[:, 2].min())
    img_PCA = np.full([img.shape[1], 3], dtype=np.int16, fill_value=-9999)
    img_PCA[non_index, 0] = scalePC1.astype(np.int16)
    img_PCA[non_index, 1] = scalePC2.astype(np.int16)
    img_PCA[non_index, 2] = scalePC3.astype(np.int16)
    return img_PCA


def trait_path(trait):
    return os.path.join(folder, trait + '.tif')


def process_window(i, j):
    # uses the worker globals set in init_worker: folder, out_folder, pca, scaler, trait_list, meta, window_size
    out_path = os.path.join(out_folder, f'PCA_{i}_{j}.tif')
    if os.path.exists(out_path):
        return

    # window clipped to the raster extent (edge tiles are smaller); transform from the first trait map
    with rasterio.open(trait_path(trait_list[0])) as src:
        window = windows.Window(j, i, min(window_size, src.width - j), min(window_size, src.height - i))
        transform = src.window_transform(window)

    img_list = []
    for trait in trait_list:
        with rasterio.open(trait_path(trait)) as src:
            img = src.read(1, window=window)
            img_list.append(img)

    img_all = np.stack(img_list, axis=0).reshape(len(trait_list), -1)

    meta.update(
        dtype=rasterio.int16, compress='lzw', count=3, nodata=-9999,
        width=window.width, height=window.height, transform=transform
    )

    # transform only if the tile has at least one valid pixel
    if ~(np.concatenate([img_all[:-1, ] == 0, (img_all[-1,] == 32767).reshape(1, -1)], axis=0)).all():
        img_PCA = PCA_transform(img_all, pca, scaler)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(img_PCA[:, 0].reshape(window.height, window.width), 1)
            dst.write(img_PCA[:, 1].reshape(window.height, window.width), 2)
            dst.write(img_PCA[:, 2].reshape(window.height, window.width), 3)
    else:
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(np.full((window.height, window.width), -9999, dtype=np.int16), 1)
            dst.write(np.full((window.height, window.width), -9999, dtype=np.int16), 2)
            dst.write(np.full((window.height, window.width), -9999, dtype=np.int16), 3)


def init_worker(folder_val, out_folder_val, pca_val, scaler_val, trait_list_val, meta_val, window_size_val):
    """Set the globals in every worker process (needed on Windows, where workers are spawned)."""
    global folder, out_folder, pca, scaler, trait_list, meta, window_size
    folder = folder_val
    out_folder = out_folder_val
    pca = pca_val
    scaler = scaler_val
    trait_list = trait_list_val
    meta = meta_val
    window_size = window_size_val


if __name__ == '__main__':
    folder = TRAIT_DIR
    out_folder = OUT_DIR
    os.makedirs(out_folder, exist_ok=True)

    pca = pickle.load(open(os.path.join(MODEL_DIR, 'PCA_model_rp_sa_10traits.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_rp_10traits.pkl'), 'rb'))

    trait_list = TRAIT_LIST
    window_size = WINDOW_SIZE

    missing = [t for t in trait_list if not os.path.exists(trait_path(t))]
    if missing:
        raise FileNotFoundError(f'missing trait maps in {folder}: {missing} '
                                '(canopy_height.tif must be prepared by the user, see the header)')

    # basic metadata and raster size from the first trait map
    with rasterio.open(trait_path(trait_list[0])) as src:
        meta = src.meta.copy()
        ncols, nrows = src.width, src.height

    n_tiles = len(range(0, nrows, window_size)) * len(range(0, ncols, window_size))
    print(f'{nrows} x {ncols} pixels -> {n_tiles} tiles of {window_size} px, {N_WORKERS} workers')

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=init_worker,
            initargs=(folder, out_folder, pca, scaler, trait_list, meta, window_size)
    ) as executor:
        futures = []
        for i in range(0, nrows, window_size):
            for j in range(0, ncols, window_size):
                futures.append(executor.submit(process_window, i, j))
        concurrent.futures.wait(futures)
        for fut in futures:
            fut.result()  # re-raise worker errors
    print('done, tiles written to', out_folder)
