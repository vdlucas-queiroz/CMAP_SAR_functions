library(raster)
library(sp)
library(rgdal)
library(sf)
library(dplyr)

# Function: raster_to_shapefile_sample
# Description: This function takes a raster file with class samples, samples them according to a specified proportion, and saves them as shapefiles for training and validation.

# Parameters:
#   raster_file: A character string representing the path to the input raster file.
#   output_dir: A character string representing the directory where the output shapefiles will be saved.
#   proportion: A numeric value between 0 and 1, indicating the proportion of samples to allocate to the training set. The remaining samples will go to the validation set.
#   class_names: A character vector containing the names of the classes represented in the raster.
#   balance: An optional numeric parameter for balancing the number of samples in each class. If NULL, no balancing is applied. If a numeric value is provided, it can be used to control class balancing.
#   prefix: An optional prefix to be added to the output shapefile names.

# Usage:
#   raster_to_shapefile_sample(raster_file, output_dir, proportion, class_names, balance = NULL, prefix = NULL)

# Example:
#   raster_file <- "path/to/your/raster/file.tif"
#   output_dir <- "path/to/output/directory/"
#   proportion <- 0.7  # 70% of samples for training, 30% for validation
#   class_names <- c('Class1', 'Class2', 'Class3')  # Replace with your class names
#   balance <- 10  # An optional balance factor for class balancing
#   prefix <- "prefix_"  # An optional prefix for shapefile names
#   raster_to_shapefile_sample(raster_file, output_dir, proportion, class_names, balance, prefix)

# Notes:
# - This function assumes that the raster contains class labels, and it samples these classes based on the specified proportion.
# - The shapefiles for training and validation sets will be saved in separate directories within the 'output_dir'.
# - The function ensures that '0' values (no-data) are not included in the samples.

# Author: [Vinícius D'Lucas Bezerra e Queiroz]
# Date: [2023/10/23]


# A lista de classes deve estar na ordem de id que o raster
raster_to_shapefile_sample <- function(raster_file, output_dir, proportion, class_names, 
                                       balance = NULL,prefix=NULL) {
  
  # Raster with samples
  raster_data <- raster(raster_file)
  
  # Vector of classes
  classes_pixels <- raster_data[]
  
  # Identifying classes
  classes <- sort(unique(classes_pixels))
  
  # Path to train and validation sets
  train_dir <- file.path(output_dir, "train")
  valid_dir <- file.path(output_dir, "valid")
  dir.create(train_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(valid_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Creating SpatialPoints with attributes
  train_samples <- list()
  valid_samples <- list()
  
  for (current_class in classes) {
    
    # Here '0' means no-data
    if (current_class == 0) {
      next
    }
    
    pixels_class <- which(classes_pixels == current_class)
    num_pixels_class <- length(pixels_class)
    num_pixels_train <- round(proportion * num_pixels_class)
    
    if (!is.null(balance)) {
      min_samples <- min(table(classes_pixels))
      if (balance == 1) {
        # Calculate the minimum number of samples
        num_pixels_train <- min(num_pixels_train, min_samples)
      } else {
        # Calculate the maximum number of samples based on the balance factor
        max_samples <- min(min_samples * balance, num_pixels_class)
        num_pixels_train <- min(num_pixels_train, max_samples)
      }
    }
    
    # Random stratified samples for training
    train_samples_class <- sample(pixels_class, num_pixels_train, replace = FALSE)
    train_samples_coords <- xyFromCell(raster_data, train_samples_class)
    train_samples_attributes <- rep(current_class, length(train_samples_class))
    sp_points_df_train <- SpatialPointsDataFrame(train_samples_coords, data = data.frame(class_name = class_names[current_class], Class = train_samples_attributes))
    
    # Validation samples
    pixels_valid <- setdiff(pixels_class, train_samples_class)
    valid_samples_coords <- xyFromCell(raster_data, pixels_valid)
    valid_samples_attributes <- rep(current_class, length(pixels_valid))
    sp_points_df_valid <- SpatialPointsDataFrame(valid_samples_coords, data = data.frame(class_name = class_names[current_class], Class = valid_samples_attributes))
    
    # Appending to a list
    train_samples[[current_class]] <- sp_points_df_train
    valid_samples[[current_class]] <- sp_points_df_valid
  }
  
  # Combining samples
  train_sp <- do.call(rbind, train_samples)
  valid_sp <- do.call(rbind, valid_samples)
  
  # Getting CRS
  raster_crs <- projection(raster_data)
  
  # Assigning CRS
  proj4string(train_sp) <- raster_crs
  proj4string(valid_sp) <- raster_crs
  
  # Creating shapefiles
  shapefile_name_train <- file.path(train_dir, paste(prefix,"train.shp"))
  shapefile_name_valid <- file.path(valid_dir, paste(prefix,"valid.shp"))
  writeOGR(train_sp, dsn = shapefile_name_train, layer = paste('train'), driver = "ESRI Shapefile")
  writeOGR(valid_sp, dsn = shapefile_name_valid, layer = paste('valid'), driver = "ESRI Shapefile")
}


# raster_file = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/raster_samples/raster_samples2010_30m.tif"
# output_dir = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/"
# class_names = c('AP+AC','VS3','SE','FP','VS1','PL+PS')
# proportion = 0.5
# # balance = 1
# 
# raster_to_shapefile_sample(raster_file, output_dir, proportion, class_names, prefix = '2010_samples')


# --------------------------------------------------------------------------------------------------------------------------------------

raster_to_shapefile_sample_by_lags <- function(raster_file, output_dir, class_names, lags, prefix=NULL) {
  
  # Raster with samples
  raster_data <- raster(raster_file)
  ext <- extent(raster_data)
  res <- res(raster_data)
  
  # Vector of classes
  classes_pixels <- raster_data[]
  
  # Identifying classes
  classes <- sort(unique(classes_pixels))
  
  # Path to train and validation sets
  train_dir <- file.path(output_dir, "train")
  valid_dir <- file.path(output_dir, "valid")
  dir.create(train_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(valid_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Creating SpatialPoints with attributes
  train_samples <- list()
  valid_samples <- list()
  
  for (current_class in classes) {
    
    # Skipping no-data
    if (current_class == 0 | current_class == -9999) {
      next
    }
    
    # Lags for current class
    h_lag <- lags[['h']][current_class]
    v_lag <- lags[['v']][current_class]
    
    coords_x <- seq(ext@xmin + res[1]/2, ext@xmax, by = res[1] * h_lag)
    coords_y <- seq(ext@ymin + res[2]/2, ext@ymax, by = res[2] * v_lag)
    
    coords_grid <- expand.grid(x = coords_x, y = coords_y)
    
    # Filter pixels by class and lags
    lagged_pixels <- cellFromXY(raster_data, coords_grid)
    lagged_pixels <- lagged_pixels[!is.na(lagged_pixels)] # Filtrar índices inválidos
    class_lagged_pixels <- lagged_pixels[classes_pixels[lagged_pixels] == current_class]
    valid_class_lagged_pixels <- class_lagged_pixels[!is.na(class_lagged_pixels) & class_lagged_pixels %in% 1:ncell(raster_data)]
    
    if (h_lag == 1 & v_lag == 1) {
      # Skip validation sample creation if either lag is 1
      # Creating SpatialPoints for training
      train_samples_coords <- xyFromCell(raster_data, valid_class_lagged_pixels)
      train_samples_attributes <- rep(current_class, length(valid_class_lagged_pixels))
      sp_points_df_train <- SpatialPointsDataFrame(train_samples_coords, data = data.frame(class_name = class_names[current_class], Class_id = train_samples_attributes))
      # Appending to list
      train_samples[[current_class]] <- sp_points_df_train
        
      message(paste("Skipping validation sample creation for class", current_class, "due to h_lag = 1 and v_lag = 1"))
      
      } else { 
        
      # Creating SpatialPoints for training
      train_samples_coords <- xyFromCell(raster_data, valid_class_lagged_pixels)
      train_samples_attributes <- rep(current_class, length(valid_class_lagged_pixels))
      sp_points_df_train <- SpatialPointsDataFrame(train_samples_coords, data = data.frame(class_name = class_names[current_class], Class_id = train_samples_attributes))
      # Appending to list
      train_samples[[current_class]] <- sp_points_df_train
      
      # Remaining pixels for validation
      all_pixels_class <- which(classes_pixels == current_class)
      valid_pixels <- setdiff(all_pixels_class, class_lagged_pixels)
      valid_samples_coords <- xyFromCell(raster_data, valid_pixels)
      valid_samples_attributes <- rep(current_class, length(valid_pixels))
      sp_points_df_valid <- SpatialPointsDataFrame(valid_samples_coords, data = data.frame(class_name = class_names[current_class], Class_id = valid_samples_attributes))
      # Appending to list
      valid_samples[[current_class]] <- sp_points_df_valid
    }
    

  }
  
  # Combining samples
  train_sp <- do.call(rbind, train_samples)
  valid_samples <- valid_samples[!sapply(valid_samples, is.null)]
  valid_sp <- do.call(rbind, valid_samples)
  
  # Getting CRS
  raster_crs <- projection(raster_data)
  
  # Assigning CRS
  proj4string(train_sp) <- raster_crs
  proj4string(valid_sp) <- raster_crs
  
  # Creating shapefiles
  shapefile_name_train <- file.path(train_dir, paste0(prefix,"_train.shp"))
  shapefile_name_valid <- file.path(valid_dir, paste0(prefix,"_valid.shp"))
  writeOGR(train_sp, dsn = shapefile_name_train, layer = paste0('train'), driver = "ESRI Shapefile")
  writeOGR(valid_sp, dsn = shapefile_name_valid, layer = paste0('valid'), driver = "ESRI Shapefile")
}
  

#raster_file = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/raster/amostras_total_2008_10classes.tif"
#output_dir = "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/"
#class_names = c('AP','AC','FD','FP','PL','PS','SE','SV1','SV2','SV3')



# #lags_HV_2008_filtered <- data.frame(
#   Classe = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#   h = c(14, 10, 43, 31, 10, 14, 13, 14, 17, 18),
#   v = c(18, 13, 29, 25, 12, 12, 13, 12, 12, 21)
# )
# 


# #lags_HH_2008_filtered <- data.frame(
#   Classe = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#   h = c(13, 10, 44, 30, 12, 15, 13, 13, 16, 17),
#   v = c(18, 13, 29, 25, 16, 14, 13, 14, 12, 21)
# )



# raster_to_shapefile_sample_by_lags(raster_file, output_dir, class_names, lags_HV_2008_filtered,prefix = '2008_samples_lags_HV_filtered')
# raster_to_shapefile_sample_by_lags(raster_file, output_dir, class_names, lags_HH_2008_filtered,prefix = '2008_samples_lags_HH_filtered')

#----------------------------------------------------------------------------------------------------------------------------------------

# Nesse caso, o usuário já tem suas amostras em shapefile
# Função para dividir um conjunto de pontos no formato shapefile em treinamento e validação
divide_data <- function(shp_path, class_name, output_dir, train_ratio) {
  # Carregar shapefile
  data <- st_read(shp_path)
  
  # Garantir que existe uma coluna de classe
  if (!(class_name %in% colnames(data))) {
    stop("A coluna 'Class' não foi encontrada no shapefile.")
  }
  
  # Dividir os dados por classe e realizar amostragem estratificada
  train_samples <- data %>%
    group_by(class_name) %>%  # Usar o nome correto da coluna de classe
    slice_sample(prop = train_ratio) %>% # Usa slice_sample para manter estrutura sf
    ungroup() %>%
    st_as_sf()  # Garantir que os dados permanecem como sf
  
  valid_samples <- data %>% 
    filter(!geometry %in% train_samples$geometry) %>%
    st_as_sf()
  
  # Criar diretório de saída se não existir
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Salvar os conjuntos de treinamento e validação como shapefiles
  st_write(train_samples, file.path(output_dir, "train_samples.shp"), delete_dsn = TRUE)
  st_write(valid_samples, file.path(output_dir, "valid_samples.shp"), delete_dsn = TRUE)
  
  return(list(train_samples = train_samples, valid_samples = valid_samples))
}

#shp_path <- "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/train/2008_samples_lags_HH train.shp"
#output_dir <- "G:/Meu Drive/INPE/projeto_dissertacao/samples_design/reis_vec_samples/vector/classification_test/2008"
#train_ratio <- 0.7

# Executar a função
#result <- divide_data(shp_path, output_dir, train_ratio)




