Fixed Giotto masked-gene CSVs live here.

The training script expects one CSV per train/validation slide:

slide{slide_id}_giotto_ranked_genes.csv

Required columns:

- rank
- gene

The first `gene_mask_imputer.mask_n_genes` rows are always masked from model input
and excluded from training losses. If a slide CSV is missing and
`create_fixed_gene_csv_if_missing` is true, `train_gene_mask_imputer.py` creates it
from whole-slide Giotto ranking on the first run.
