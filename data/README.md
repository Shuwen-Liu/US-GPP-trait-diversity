# data/

Download every file of the Zenodo archive (https://doi.org/10.5281/zenodo.22215583)
into this folder and run

```bash
python organize_zenodo_files.py
```

which sorts the flat Zenodo file names into the folders the scripts expect:

```text
data/
  analysis_ready_tables/   data.csv, data_aridity_index_class_scaler.csv
  grid_005_layers/         PC_scores/, predictors/, masks/
  fd_layers_005/           FRic_*.tif, FDiv_*.tif, Fbeta_*.tif
  pca_model/               PCA_model_rp_sa_10traits.pkl, scaler_rp_10traits.pkl, pca_sample_points.csv
  trait_maps_30m/          Carbon.tif ... SLA.tif  (only needed to regenerate the 30-m PC maps)
```

The 30-m trait maps (about 136 GB) are only needed by
`01_functional_composition/04_apply_pca_to_trait_maps.py`; everything else runs from the
small files. See the archive's README.md for variable definitions and units.
