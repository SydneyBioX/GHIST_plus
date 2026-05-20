"""Image loading helpers used by the GHIST+ datasets."""

import tifffile
import imageio
from pathlib import Path
import numpy as np


def load_image(fp):
    if Path(fp).suffix.lower() in [".tif", ".ome.tif"]:
        image = tifffile.imread(fp)
    else:
        image = imageio.imread(fp)

    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    return image
