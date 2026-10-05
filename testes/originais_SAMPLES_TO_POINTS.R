# Instale as bibliotecas se ainda não tiver instalado
# install.packages("raster")
# install.packages("sf")

# Carregue as bibliotecas
library(raster)
library(sf)

# Função para converter imagem raster em shapefile
raster_to_points <- function(raster_file, output_dir, class_names, prefix = '2010_morans') {
  
  # Raster with samples
  raster_data <- raster(raster_file)
  
  # Vector of classes
  classes_pixels <- raster_data[]
  
  # Identifying classes
  classes <- sort(unique(classes_pixels))
  
  
  dir <- file.path(output_dir, "points")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  
  point_samples <- list()
  
  for (current_class in classes) {
    
    # Here '0' means no-data
    if (current_class == 0) {
      next
    }
    
    pixels_class <- which(classes_pixels == current_class)
    
    # Random stratified samples for training
    #train_samples_class <- sample(pixels_class, num_pixels_train, replace = FALSE)
    samples_coords <- xyFromCell(raster_data, pixels_class)
    samples_attributes <- rep(current_class, length(pixels_class))
    sp_points <- SpatialPointsDataFrame(samples_coords, data = data.frame(class_name = class_names[current_class], Class = samples_attributes))
    
  
    # Appending to a list
    point_samples[[current_class]] <- sp_points
    
  }
  
  # Combining samples
  points_sp <- do.call(rbind, point_samples)
  
  # Getting CRS
  raster_crs <- projection(raster_data)
  
  # Assigning CRS
  proj4string(points_sp) <- raster_crs
  
  # Creating shapefiles
  shapefile_name_points <- file.path(dir, paste(prefix,"points.shp"))
  writeOGR(points_sp, dsn = shapefile_name_points, layer = paste('points'), driver = "ESRI Shapefile")
  
}
raster_file = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/raster/amostras_total_2010.tif"
output_dir = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/"
class_names = c('AP','AC','FD','FP','PL','PS','SE','VS1','VS2','VS3')


raster_to_points(raster_file, output_dir, class_names, prefix = '2010_moran')
  