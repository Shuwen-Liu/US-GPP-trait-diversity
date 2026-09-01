"""Mosaic the 30 m PC tiles written by 04_apply_pca_to_trait_maps.py.

Inputs
------
./results/pc_tiles/PCA_*.tif   3-band Int16 tiles (PC1, PC2, PC3), NoData -9999

Outputs
-------
./results/PCA_30m_mosaic.tif   3-band Int16 BigTIFF, LZW, NoData -9999

rasterio.merge.merge writes the mosaic straight to disk in chunks of MEM_LIMIT MB
(requires rasterio >= 1.4), so the full CONUS raster is never held in memory
and no GDAL Python bindings are needed.

Run from this folder:  python 05_mosaic_pc_tiles.py
"""
import glob
import os

from rasterio.merge import merge

TILE_DIR = './results/pc_tiles'
OUT_PATH = './results/PCA_30m_mosaic.tif'
NODATA = -9999
MEM_LIMIT = 1024  # MB per output chunk

tiles = sorted(glob.glob(os.path.join(TILE_DIR, 'PCA_*.tif')))
if not tiles:
    raise FileNotFoundError(f'no tiles found in {TILE_DIR}; run 04_apply_pca_to_trait_maps.py first')
print(f'mosaicking {len(tiles)} tiles -> {OUT_PATH}')

merge(
    tiles,
    nodata=NODATA,
    dtype='int16',
    dst_path=OUT_PATH,
    dst_kwds=dict(driver='GTiff', compress='lzw', BIGTIFF='YES', tiled=True,
                  blockxsize=512, blockysize=512),
    mem_limit=MEM_LIMIT,
)
print('done')
