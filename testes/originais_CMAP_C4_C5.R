# Function: CMAP_classifier
# Description: This function performs time-series Compound Maximum a Posteriori
# classification of raster images based on class likelihoods and trajectory weights. 
# It computes the classification for each time point and outputs the result as raster files.

# Parameters:
#   class_likelihood_paths: A character vector containing the file paths to raster images for each date. Each raster file represents the likelihood of classes for a specific date.
#   trajectory_matrix: A dataframe representing the trajectories and their corresponding weights for classification.
#   output_path: A character string indicating the directory where the output raster files will be saved.

# Usage:
#   CMAP_classifier(class_likelihood_paths, trajectory_matrix, output_path)

# Example:
#   class_likelihood_paths <- c("path/to/2005_probs.tif", "path/to/2006_probs.tif")
#   trajectory_matrix <- read.table("path/to/trajectory_weights.txt", sep = ";", header = TRUE)
#   output_path <- "path/to/output/directory/"
#   CMAP_classifier(class_likelihood_paths, trajectory_matrix, output_path)

# Notes:
# - The function assumes that each raster file contains the likelihood for different classes at a given date.
# - The trajectory matrix should contain weights for different classification trajectories.
# - algorithm classifies all the images in the time series at the same time.
# - The output raster files will represent the classification result for each date and will be saved in the specified output directory.
# - The order of classes in the trajectory matrix should match the class IDs in the raster files.

# Author: [Mariane Souza Reis]
# Adaptation and optimization: [Vinícius D'Lucas Bezerra e Queiroz]
# Date: [30/01/2024]



#-------------------------------CMAP CLASSIFIER-------------------------------------
CMAP_classifier <- function(class_likelihood_paths,traj_weights,output_path, masking = NULL, sufix = NULL){
  
  
  n_dates <- length(class_likelihood_paths) 
  n_classes <- unname(sapply(class_likelihood_paths, function(x) nlyr(rast(x))))
  
  # Pre-allocating the 'data' df with the correct size
  
  total_cols <- sum(n_classes)
  num_rows <- nrow(rast(class_likelihood_paths[1]))*ncol(rast(class_likelihood_paths[1]))
  
  data <- as.data.frame(matrix(nrow = num_rows, ncol = total_cols))
  
  current_col <- 1
  
  # Turning each likelihood image into a column of a df
  
  for (i in 1:n_dates){
    temp <- as.data.frame(values(rast(class_likelihood_paths[i])))
    
    # Determine the range of columns for 'temp' in 'data'
    next_col <- current_col + ncol(temp) - 1
    
    # Assign 'temp' to the range of columns in 'data'
    data[, current_col:next_col] <- temp
    
    # Update the index of the current column for the next iteration
    current_col <- next_col + 1
  }
  
  # Initializing the classification result
  # Note: number of columns equals the number of dates;
  # Each column refers to one date  
  
  classification <- matrix(0, nrow = num_rows, ncol = n_dates)
  
  valorm <- rep(-100, num_rows)
  
  # Facilitating the shifting of columns in the dataframe
  index_shifts <- cumsum(c(0, n_classes[-length(n_classes)]))
  
  n_trajectories <- nrow(traj_weights)
  
  #traj_weights$final_weights <- traj_weights$pesoF
  
  for (s in 1:n_trajectories) {
    if (traj_weights$final_weights[s] != 0) {
      trajectory_sequence <- as.numeric(traj_weights[s, 1:n_dates])
      
      value <- rep(1, num_rows)
      
      # Multiplying the likelihood values of each class w 
      # of the trajectory s
            for (w in seq_along(trajectory_sequence)) {
        col_index <- trajectory_sequence[w] + index_shifts[w]
        value <- value * data[, col_index]       # Value is now the product of the probabilities of the trajectory classes s
        
      }
      
      # Multiplying the result by the trajectory weight s
      # If using discriminant function, add...value would start with 0
      value <- value * traj_weights$final_weights[s]
      
      # Checking which indices have higher values than the previous one
      update_indices <- which(value > valorm)
      
      # Only replaces at the index where the value is greater
      # Replaces the trajectory by complet
      if (length(update_indices) > 0) {
        classification[update_indices, ] <- matrix(trajectory_sequence, nrow = length(update_indices), ncol = n_dates, byrow = TRUE)
        
        # Update the minimum value
        valorm[update_indices] <- value[update_indices]
      }
    }
    cat("trajectory", s,"\n")
  }
  
  # Writing the raster
  b1 <- setValues(raster(class_likelihood_paths[1]), rep(0, num_rows))
  
  for (i in 1:n_dates){
    
      b1[]<-classification[,i]
    
    if (!is.null(masking)){
      
      if(masking[i]!= '0'){
      
        cloud_mask <- brick(masking[i])
      
        b1 <- masking_replace(b1,cloud_mask,0)
      }
    }
    writeRaster(b1,paste0(output_path,"CMAP_",i,"_", sufix,".tif"),overwrite=TRUE)
    
  }
}

masking_replace <- function(image, mask, value) {
  # Verificar se a imagem e a máscara têm o mesmo número de linhas e colunas
  if (dim(image)[1] != dim(mask)[1] || dim(image)[2] != dim(mask)[2]) {
    stop("A imagem e a máscara devem ter a mesma dimensão")
  }
  
  # Aplicar a máscara em todas as bandas da imagem
  for (band in 1:nlayers(image)) {
    # Obter a banda atual
    actual_band <- raster::subset(image, band)
    
    # Substituir os valores onde a máscara é TRUE (presumindo que TRUE indica nuvens)
    pixels <- values(actual_band)
    pixels[values(mask)==1] <- value
    values(actual_band) <- pixels
    
    # Salvar a banda modificada de volta na imagem
    image[[band]] <- actual_band
  }
  
  # Retorna a imagem modificada
  return(image)
}

# Função para verificar se todas as imagens têm o mesmo tamanho
verificar_dimensoes <- function(paths) {
  # Inicializar variáveis para armazenar as dimensões da primeira imagem
  inicial <- NULL
  
  # Loop para ler cada imagem e verificar suas dimensões
  for (path in paths) {
    # Carregar a imagem
    img <- raster(path)
    
    # Obter dimensões da imagem
    dimensoes <- c(nrow(img), ncol(img))
    
    # Se é a primeira imagem, definir como inicial
    if (is.null(inicial)) {
      inicial <- dimensoes
    } else {
      # Se as dimensões não forem iguais às da primeira imagem, retornar FALSE
      if (!all(inicial == dimensoes)) {
        cat("Imagem em", path, "tem dimensões diferentes: esperado", inicial, "obtido", dimensoes, "\n")
        return(FALSE)
      }
    }
  }
  
  # Se todas as imagens têm as mesmas dimensões
  cat("Todas as imagens têm as mesmas dimensões:", inicial, "\n")
  return(TRUE)
}


# CMAP CENÁRIOS C3 E C6
ptm <- proc.time()
ptm2 <- proc.time()

tempo <-  ptm2 - ptm


#------------------------------- DATA IMPORTING -------------------------------------


library(terra)


paths_2010 <- c("G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HH_gamma_model_10classes_imag2010_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HV_gamma_model_10classes_imag2010_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_bigamma_model_10classes_imag2010_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HH_a_gaussian_model_10classes_imag2010_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HV_a_gaussian_model_10classes_imag2010_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_bigaussian_a_model_10classes_imag2010_Spk3.TIF")

paths_2008 <- c("G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HH_gamma_model_10classes_imag2008_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HV_gamma_model_10classes_imag2008_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_bigamma_model_10classes_imag2008_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HH_a_gaussian_model_10classes_imag2008_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_HV_a_gaussian_model_10classes_imag2008_Spk3.TIF",
                "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C1_C2/spk3/image_likelihood_bigaussian_a_model_10classes_imag2008_Spk3.TIF")




class_likelihood_paths_C4_C5  <-c("G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/likelihoods_2005_resampled.TIF",
                                  "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/likelihoods_2006_resampled.TIF",
                                  "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/likelihoods_2007_resampled.TIF",
                                  paths_2008[4],
                                  "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/likelihoods_2009_resampled.TIF",
                                  paths_2010[4])



output_path_C4_C5 <- "G:/Meu Drive/INPE/projeto_dissertacao/7_CMAP/C4_C5/"



cloud_mask_paths_C4_C5  <- c('G:/Meu Drive/INPE/projeto_dissertacao/0_images/3_cloud_mask/rec_cloud_mask_2005_resampled.TIF',
                             'G:/Meu Drive/INPE/projeto_dissertacao/0_images/3_cloud_mask/rec_cloud_mask_2006_resampled.TIF',
                             'G:/Meu Drive/INPE/projeto_dissertacao/0_images/3_cloud_mask/rec_cloud_mask_2007_resampled.TIF',
                             0,
                             'G:/Meu Drive/INPE/projeto_dissertacao/0_images/3_cloud_mask/rec_cloud_mask_2009_resampled.TIF',
                             0)

validade_C4_C5 <- read.table("G:/Meu Drive/INPE/projeto_dissertacao/7_CMAP/C4_C5/pesos_C4_C5.txt",sep=";",h=T)


trajectory_matrix_C4_C5 <- validade_C4_C5


#-----------------------------------------------------

verificar_dimensoes(class_likelihood_paths_C4_C5)



CMAP_classifier(class_likelihood_paths_C4_C5, trajectory_matrix_C4_C5, 
                output_path_C4_C5, masking = cloud_mask_paths_C4_C5, sufix ="Gaussiano_a_HH")







