# CLOUD MASKING

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
