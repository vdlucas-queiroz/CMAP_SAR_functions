"""drop_duplicates_by_class × filtro dplyr; extract_polygon_samples × raster::extract (R)."""
import sys, subprocess; sys.path.insert(0, '..')
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.geometry import Polygon
from rasterio.transform import from_origin
from sar_classification import samples, sampling
open('dup.R', 'w').write("""suppressMessages(library(dplyr))
d <- data.frame(Class = c('A','A','A','A','B','B'), HH = c(1, 1, 2, 3, 1, 1), HV = c(5, 6, 5, 7, 5, 9))
write.csv(d %>% group_by(Class) %>% filter(!duplicated(HH) & !duplicated(HV)) %>% ungroup(), 'dup_R.csv', row.names = FALSE)""")
subprocess.run(['Rscript', 'dup.R'], check=True)
d = pd.DataFrame({'Class': ['A', 'A', 'A', 'A', 'B', 'B'], 'HH': [1, 1, 2, 3, 1, 1], 'HV': [5, 6, 5, 7, 5, 9]})
print('drop_duplicates_by_class == dplyr:', samples.drop_duplicates_by_class(d).reset_index(drop=True).equals(pd.read_csv('dup_R.csv')))
rng = np.random.default_rng(4); H, W = 50, 60; tr = from_origin(500000, 9000000, 12.5, 12.5)
with rasterio.open('img2.tif', 'w', driver='GTiff', height=H, width=W, count=2, dtype='float32', crs='EPSG:31983', transform=tr) as f:
    f.write(rng.gamma(3, .03, (2, H, W)).astype('float32'))
polys = []
for k in range(8):
    c, r = rng.uniform(2, 45), rng.uniform(2, 35)
    x0, y0 = tr * (c, r); x1, y1 = tr * (c + rng.uniform(4, 12), r + rng.uniform(4, 12))
    polys.append(Polygon([(x0, y0), (x1, y0 + 7), (x1 - 9, y1), (x0 + 3, y1 - 4)]))
g = gpd.GeoDataFrame({'DN': [1, 2, 3, 1, 2, 3, 1, 2]}, geometry=polys, crs='EPSG:31983'); g.to_file('polys.gpkg')
open('ext.R', 'w').write("""suppressMessages({library(raster); library(sf)})
b <- brick('img2.tif'); p <- st_read('polys.gpkg', quiet = TRUE); u <- aggregate(p, by = list(DN = p$DN), FUN = function(x) x[1])
e <- raster::extract(b, u, cellnumbers = TRUE)
write.csv(do.call(rbind, lapply(seq_along(e), function(i) data.frame(DN = u$DN[i], cell = e[[i]][, 'cell']))), 'R_ext.csv', row.names = FALSE)""")
subprocess.run(['Rscript', 'ext.R'], check=True)
py = sampling.extract_polygon_samples(g, 'img2.tif', 'DN')
r = pd.read_csv('R_ext.csv'); r['row'], r['col'] = (r.cell - 1) // W, (r.cell - 1) % W
dup = r.duplicated(['row', 'col'], keep=False)
print('extract_polygon_samples == raster::extract (fora das sobreposições):',
      set(zip(r[~dup].DN.astype(str), r[~dup].row, r[~dup].col)) == set(zip(py.Class, py.row, py.col)))
