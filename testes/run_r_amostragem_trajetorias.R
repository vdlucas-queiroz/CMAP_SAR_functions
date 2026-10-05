suppressMessages({library(raster); library(sp); library(sf)})
writeOGR <- function(obj, dsn, layer, driver) {                     # substitui apenas a gravação
  write.csv(data.frame(coordinates(obj), obj@data), paste0(sub("\\.shp$", "", gsub(" ", "_", dsn)), ".csv"), row.names = FALSE)
}
src <- function(f) { e <- parse(text = gsub("\r", "", readLines(f, warn = FALSE)))
  for (x in e) if (is.call(x) && as.character(x[[1]]) %in% c("<-", "=") && is.call(x[[3]]) && identical(x[[3]][[1]], as.name("function"))) eval(x, globalenv()) }
src('originais_14_sample_design.R'); src('originais_SAMPLES_TO_POINTS.R')
src('originais_trajectory_assessment.R'); src('originais_cloud_masking.R')
cn <- c('A','B','C','D','E')
dir.create('R_out', showWarnings = FALSE)
raster_to_points('samples.tif', 'R_out', cn, prefix = 'p')
lags <- data.frame(Classe = 1:5, h = c(3, 2, 1, 4, 2), v = c(2, 3, 1, 2, 5))
raster_to_shapefile_sample_by_lags('samples.tif', 'R_out', cn, lags, prefix = 'lag')
set.seed(1); raster_to_shapefile_sample('samples.tif', 'R_out/rand', 0.7, cn, prefix = 'r')
set.seed(1); raster_to_shapefile_sample('samples.tif', 'R_out/bal2', 0.7, cn, balance = 2, prefix = 'b')
# trajetórias impossíveis (a função exige 'prefixo'; arquivos ML_1..ML_4)
for (i in 1:4) file.copy(sprintf('ML_%d.tif', i), sprintf('ML_%d.TIF', i), overwrite = TRUE)
pct <- calcularTransicoesImpossiveis('', 'ML_', 1:4, c('T1.txt', 'T2.txt', 'T3.txt'))
r <- raster('trajetorias_impossiveis.tif'); write.csv(data.frame(v = r[]), 'R_out/impossible.csv', row.names = FALSE)
cat('IMPOSSIVEIS_PCT', pct, '\n')
# comparação ML x CMAP (arquivos Classification_<ano>.TIF e CMAP_<i>.TIF)
for (i in 1:4) { file.copy(sprintf('ML_%d.tif', i), sprintf('Classification_%d.TIF', i), overwrite = TRUE); file.copy(sprintf('CMAP_%d.tif', i), sprintf('CMAP_%d.TIF', i), overwrite = TRUE) }
out <- capture.output(res <- compareRasterClassification('', '', 1:4, 'cloud.tif'))
cat('COMPARE', out, sep = '\n')
write.csv(data.frame(v = res$accumulatedDifferences[]), 'R_out/accumulated.csv', row.names = FALSE)
# máscara
img <- brick('img.tif'); m <- raster('cloud.tif')
out <- masking_replace(img, m, -1)
write.csv(as.data.frame(values(out)), 'R_out/masked.csv', row.names = FALSE)
