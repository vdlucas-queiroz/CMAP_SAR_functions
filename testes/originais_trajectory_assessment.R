calcularTransicoesImpossiveis <- function(diretorioBase,prefixo, anos, caminhosTransicoes) {
  # Inicializa a lista de rasters combinados com ponderação exponencial
  rasterCombinado <- NULL
  numTransicoes <- length(caminhosTransicoes)
  
  # Processar cada raster de entrada, aplicando uma ponderação exponencial decrescente
  for (i in 1:(numTransicoes + 1)) {
    # Carregar o raster correspondente ao ano
    rasterAnual <- raster(paste0(diretorioBase, prefixo, anos[i], ".TIF"))
    # Aplicar ponderação decrescente
    rasterPonderado <- 10^(6 - i) * rasterAnual
    
    # Combinar rasters ponderados
    if (is.null(rasterCombinado)) {
      rasterCombinado <- rasterPonderado
    } else {
      rasterCombinado <- rasterCombinado + rasterPonderado
    }
  }
  
  
  # Armazenar informações sobre as classes de cada matriz de transição
  numClasses <- vector(length = (numTransicoes + 1))
  classes <- list(length = numTransicoes + 1)
  
  # Ler e armazenar dados das matrizes de transição
  for (z in 1:numTransicoes) {
    matrizTransicao <- read.table(caminhosTransicoes[z], sep = ";", header = FALSE)
    
    if (z == 1) {
      numClasses[1] <- dim(matrizTransicao)[1]
      classes[[1]] <- 1:dim(matrizTransicao)[1]
    }
    numClasses[z + 1] <- dim(matrizTransicao)[2]
    classes[[z + 1]] <- 1:dim(matrizTransicao)[2]
  }
  
  # Criar tabela de todas as combinações possíveis de transições
  tabelaTransicoes <- expand.grid(classes[])
  pesos <- vector(length = dim(tabelaTransicoes)[1])
  
  # Atribuir pesos às transições conforme definido nas matrizes
  for (z in 1:numTransicoes) {
    matrizTransicao <- read.table(caminhosTransicoes[z], sep = ";", header = FALSE)
    for (linha in 1:dim(matrizTransicao)[1]) {
      for (coluna in 1:dim(matrizTransicao)[2]) {
        indices <- which(tabelaTransicoes[, z] == linha & tabelaTransicoes[, z + 1] == coluna)
        pesos[indices] <- matrizTransicao[linha, coluna]
      }
    }
    tabelaTransicoes <- cbind(tabelaTransicoes, pesos)
  }
  
  # Calcular o produto final dos pesos para cada combinação de transição
  if (numTransicoes > 1) {
    pesosFinais <- apply(tabelaTransicoes[, ((numTransicoes + 2):dim(tabelaTransicoes)[2])], 1, prod)
    tabelaTransicoes <- cbind(tabelaTransicoes, pesoFinal = pesosFinais)
  }
  
  # Filtrar transições com peso final zero (impossíveis)
  transicoesImpossiveis <- tabelaTransicoes[which(tabelaTransicoes$pesoFinal == 0), 1:(numTransicoes + 1)]
  transicoesImpossiveis$codificacaoTransicoes <- 0
  
  # Codificar transições impossíveis para identificação única
  for (i in 1:(numTransicoes + 1)) {
    transicoesImpossiveis$codificacaoTransicoes <- transicoesImpossiveis$codificacaoTransicoes + 10^(6 - i) * transicoesImpossiveis[, i]
  }
  
  # Criar um raster de saída identificando transições impossíveis
  rasterFinal <- rasterCombinado * 0
  indicesImpossiveis <- which(rasterCombinado[] %in% as.vector(transicoesImpossiveis$codificacaoTransicoes))
  rasterFinal[indicesImpossiveis] <- 1
  
  # Escrever o raster resultante para o arquivo
  writeRaster(rasterFinal, file = paste0(diretorioBase, "trajetorias_impossiveis.tif"),overwrite = TRUE)
  
  # Calcular a porcentagem de transições impossíveis
  contagemTransicoes <- table(as.data.frame(rasterFinal))
  porcentagemImpossiveis <- contagemTransicoes[2] / sum(contagemTransicoes) * 100
  
  return(porcentagemImpossiveis)
}


# ------------C3
caminhosTransicoes_C3  <-c("G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2005-2006.txt",
                       "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2006-2007.txt",
                       "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2007-2008.txt",
                      "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2008-2009.txt",
                      "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2009-2010.txt")

diretorioBase_C3 <- "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/"
anos_C3 <- 2005:2010

resultado_C3 <- calcularTransicoesImpossiveis(diretorioBase_C3, anos_C3, caminhosTransicoes_C3)
print(resultado_C3) # 57.47937 

# ------------C6
caminhosTransicoes_C6  <- c("G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2005-2006.txt",
                           "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2006-2007.txt",
                           "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_matrizes/01/2008-2009.txt")


diretorioBase_C6 <- "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/" #tenho que pegar o maxver
anos_C6 <- c(2005,2006,2007,2009)
resultado_C6 <- calcularTransicoesImpossiveis(diretorioBase_C6, anos_C6, caminhosTransicoes_C6)
print(resultado_C6) # 52.20572

#--------------------------------------------------------------------------------------------------------

# Function to analize disagreement between two classified rasters
compareRasterClassification <- function(baseDirectory_ML, baseDirectory_CMAP, years, maskPath) {
  
  cloudMask <- raster(maskPath)
  # Initialization of a variable to accumulate differences across years
  accumulatedDifferences <- NULL
  
  # Loop over each year to process the corresponding rasters
  for (i in 1:length(years)) {
    # Construct paths to the ML and CMAP rasters
    mlRasterPath <- paste0(baseDirectory_ML,"Classification_", years[i], ".TIF")
    cmapRasterPath <- paste0(baseDirectory_CMAP,  "CMAP_", i, ".TIF")
    
    # Load the rasters
    mlRaster <- raster(mlRasterPath)
    cmapRaster <- raster(cmapRasterPath)
    
    # Calculate differences
    difference <- mlRaster - cmapRaster
    differenceMask <- mlRaster * 0
    differenceMask[which(difference[] != 0)] <- 1
    
    # Apply cloud mask to the difference mask
    cloudAdjustedDifference <- differenceMask
    cloudAdjustedDifference[which(cloudMask[] == 1)] <- 2
    
    # Calculate and store the difference tables for each year
    differenceTable <- table(as.data.frame(difference))
    adjustedDifferenceTable <- table(as.data.frame(cloudAdjustedDifference))
    
    # Accumulate the differences in a trajectory raster
    if (is.null(accumulatedDifferences)) {
      accumulatedDifferences <- differenceMask
    } else {
      accumulatedDifferences <- accumulatedDifferences + differenceMask
    }
    
    # Print current year and percentage of significant differences
    print(years[i])
    if ("1" %in% rownames(adjustedDifferenceTable)) {
      print(100 * adjustedDifferenceTable["1"] / (adjustedDifferenceTable["0"] + adjustedDifferenceTable["1"]))
    } else {
      print(0)
    }
  }
  
  # Adjust accumulated differences with cloud mask
  accumulatedDifferences[which(cloudMask[] == 1)] <- -1
  finalDifferenceTable <- table(as.data.frame(accumulatedDifferences))
  
  # Print final trajectory analysis
  cat("Total disagreement:")
  if ("1" %in% rownames(finalDifferenceTable)) {
    print(100 * sum(finalDifferenceTable[rownames(finalDifferenceTable) > 0]) / sum(finalDifferenceTable[rownames(finalDifferenceTable) > -1]))
  } else {
    print(0)
  }
  
  return(list(accumulatedDifferences = accumulatedDifferences, finalPercentage = finalDifferenceTable))
}

# Example usage of the function
result_C3 <- compareRasterClassification(
  baseDirectory_CMAP = "G:/Meu Drive/INPE/projeto_dissertacao/7_CMAP/C3/",
  baseDirectory_ML = "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/",
  years = 2005:2010,
  maskPath = "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_mascara_nuvens/all.tif"
)

result_C6 <- compareRasterClassification(
  baseDirectory_CMAP = "G:/Meu Drive/INPE/projeto_dissertacao/7_CMAP/C6/",
  baseDirectory_ML = "G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C3/",
  years = c(2005,2006,2007,2009),
  maskPath = "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_mascara_nuvens/all.tif"
)



path <- c("1_HH_gama","2_HV_gama","3_par_intensidade","4_HH_a_gaussiana","5_HV_a_gaussiana","6_gaussiana_bivariada")
  
for (i in 1:length(path)){
  
cat(path[i])  

result_C4_C5 <- compareRasterClassification(
  baseDirectory_CMAP = paste0("G:/Meu Drive/INPE/projeto_dissertacao/7_CMAP/C4_C5/", path[i],"/"),
  baseDirectory_ML = paste0("G:/Meu Drive/INPE/projeto_dissertacao/6_cenarios/resultados_C4_C5/", path[i],"/"),
  years = 2005:2010,
  maskPath = "G:/Meu Drive/INPE/projeto_dissertacao/2_documents/documentos_reis/CMAP1/0_mascara_nuvens/all.tif"
)

}


