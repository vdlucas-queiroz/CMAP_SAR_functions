#-------------------------Auxiliary functions---------------------------------
#Libraries

#--------------------- Conversão de img p/ dataframe ------------------------

#' Convert an Image Array to a Data Frame
#'
#' This function converts a 3D image array into a data frame, where each column represents a band.
#'
#' @param image A 3D array representing the image with dimensions (rows, columns, bands).
#' @param band_names A character vector with the names of the bands.
#' @return A data frame where each column corresponds to a band in the image.
#' @export
#'
#' @examples
#' library(raster)
#' df_img <- data.frame(Band1 = runif(100), Band2 = runif(100), Band3 = runif(100))
#' template_raster <- stack(replicate(3, raster(matrix(runif(100), 10, 10))))
#' band_names <- c("Band1", "Band2", "Band3")
#' df_img <- img2df(template_raster, band_names)
img2df <- function(image,band_names){

  nrow <- dim(image)[1]
  ncol <- dim(image)[2]
  nband <- dim(image)[3]

  data_vec <- vector("list", nband)
  names(data_vec) <- band_names

  for(i in 1:nband){

    data_vec[[i]] <- as.vector(image[[i]])
  }


  df_img <- as.data.frame(data_vec)

  return(df_img)
}



#' Convert a Data Frame to an Image Array
#'
#' This function converts a data frame back into a 3D image array, maintaining the original spatial dimensions and bands.
#'
#' @param df_img A data frame where each column represents a band of the image.
#' @param image A template image from which the spatial dimensions and CRS are taken.
#' @return A 3D image array reconstructed from the data frame.
#' @import raster
#' @export
#' @examples
#' library(raster)
#' df_img <- data.frame(Band1 = runif(100), Band2 = runif(100), Band3 = runif(100))
#' template_raster <- stack(replicate(3, raster::raster(matrix(runif(100), 10, 10))))
#' img_f <- df2img(df_img, template_raster)
df2img <- function(df_img, image) {
  # Extrair dimensões da imagem de referência
  nrows <- dim(image)[1]
  ncols <- dim(image)[2]
  nbands <- ncol(df_img)  # Número de colunas no dataframe indica o número de bandas

  # Verificar se temos apenas uma banda
  if (nbands == 1) {
    data_vector <- as.vector(as.numeric(df_img[[1]]))

    # Criar o RasterLayer
    img_f <- raster(nrows=nrows, ncols=ncols)  # Inicializa um RasterLayer vazio
    values(img_f) <- data_vector  # Preenche o RasterLayer com valores
    extent(img_f) <- extent(image)  # Define a mesma extensão da imagem de referência
    projection(img_f) <- crs(image)  # Define o mesmo CRS da imagem de referência

  } else {
    # Tratar como array tridimensional para múltiplas bandas
    array_t <- array(0, dim = c(nrows, ncols, nbands))
    # Preencher o array com os dados de cada banda
    for (i in 1:nbands) {
      array_t[,,i] <- matrix(df_img[[i]], nrow = nrows, ncol = ncols, byrow = TRUE)
    }
    # Criar um RasterBrick
    img_f <- brick(array_t)
    projection(img_f) <- crs(image)
    extent(img_f) <- extent(image)
    crs(img_f) <- crs(image)
  }

  return(img_f)
}


#---------------------- Construção do dataset amostral------------------------

#' Create a Data Frame of Samples from Attribute Vectors and Classes
#'
#' This function creates a data frame by combining attribute vectors with corresponding classes.
#'
#' @param vector_attributes A list of vectors, where each vector contains attributes for a particular class.
#' @param classes A vector of class labels corresponding to each attribute vector in `vector_attributes`.
#' @param csv_name A string specifying the name of the CSV file to save the data frame. Default is "df_samples.csv". (Currently not used, as the write operation is commented out)
#' @return A data frame with combined attributes and class labels.
#' @export
#'
#' @examples
#' attributes1 <- c(1.1, 2.2, 3.3)
#' attributes2 <- c(4.4, 5.5, 6.6)
#' vector_attributes <- list(attributes1, attributes2)
#' classes <- c("Class1", "Class2")
#' df_samples <- df_samples_creation(vector_attributes, classes)

df_samples_creation <- function(vector_attributes, classes, csv_name="df_samples.csv"){

  df_samples <- data.frame()

  for(j in 1:length(classes)){
    attribute <- data.frame(Class = classes[j], vector_attributes[[j]])
    df_samples <- rbind(df_samples, attribute)
  }

  #write.csv(df_samples, csv_name)

  return(df_samples)

}

#' Downsample Data Samples Based on a Proportion Value
#'
#' This function performs downsampling on a data frame of samples to ensure that the number of samples per class does not exceed a specified proportion of the smallest class.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param ratio A numeric value indicating the proportion of the smallest class size to which other classes should be downsampled. Default is 1.
#' @return A data frame with downsampled samples for each class.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100), Value = rnorm(200))
#' downsampled_df <- downsample_data(df_samples, ratio = 0.5)
# fazer downsample das amostras dado um valor de proporção
downsample_data <- function(df_samples, ratio = 1) {
  # Calcula o número mínimo de amostras entre as classes
  min_samples <- min(table(df_samples$Class))

  # Calcula o número máximo de amostras permitido por classe com base no ratio
  max_samples <- min_samples * ratio

  # Função para fazer o downsampling
  downsample_class <- function(data) {
    if (nrow(data) > max_samples) {
      data[sample(nrow(data), max_samples), ]
    } else {
      data
    }
  }

  # Divide o dataframe por classe e aplica o downsampling
  downsampled_df <- do.call(rbind, by(df_samples, df_samples$Class, downsample_class))

  return(downsampled_df)
}



#---------------------- Generalização hierárquica das amostras ------------------------
#' Generalize Classes in a Data Frame of Samples
#'
#' This function generalizes classes in a data frame of samples based on hierarchical levels.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param hierarchical_levels A list of hierarchical levels, where each level is a list of classes to be generalized.
#' @param level An integer indicating the hierarchical level to use for generalization. Default is 1.
#' @return A data frame with generalized class labels.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = c("Class1", "Class2", "Class3"), Value = rnorm(3))
#' hierarchical_levels <- list(list(c("Class1", "Class2"), c("Class3")))
#' generalized_df <- class_generalization(df_samples, hierarchical_levels, level = 1)
#'
class_generalization <- function(df_samples, hierarchical_levels, level=1) {

  for (i in 1:length(hierarchical_levels[[level]])) {
    classes_level <- hierarchical_levels[[level]][[i]]
    new_class_name <- paste(classes_level, collapse = "+")
    df_samples$Class <- ifelse(df_samples$Class %in% classes_level, new_class_name, df_samples$Class)
  }

  return(df_samples)
}

#' Aggregate Classes in a Contingency Matrix
#'
#' This function aggregates specified classes in a contingency matrix into a single class.
#'
#' @param conf_matrix A contingency matrix with class labels as row and column names.
#' @param classes_to_aggregate A vector of class labels to be aggregated into a single class.
#' @return A contingency matrix with the specified classes aggregated.
#' @export
#'
#' @examples
#' # Creating a sample contingency matrix
#'confusion_matrix <- matrix(
#'  c(285, 3, 1, 86, 41,
#'    0, 488, 74, 0, 1,
#'    0, 1386, 223, 1, 17,
#'    1, 51, 7, 86, 15,
#'    22, 357, 73, 64, 185),
#'  nrow = 5, byrow = TRUE,
#'  dimnames = list(
#'    Prediction = c("A", "B", "C", "D", "E"),
#'    Reference = c("A", "B", " C", "D", "E")
#'  )
#')
#' aggregated_conf_matrix <- aggregate_classes(confusion_matrix, c("D", "E"))
aggregate_classes <- function(conf_matrix, classes_to_aggregate) {

  if (length(classes_to_aggregate) < 2) {
    stop("You need to specify at least two classes to aggregate.")
  }

  # Ensure the classes to aggregate exist in the matrix
  missing_classes <- setdiff(classes_to_aggregate, rownames(conf_matrix))
  if (length(missing_classes) > 0) {
    stop("The following classes are not in the confusion matrix: ", paste(missing_classes, collapse = ", "))
  }
  new_class_name <- paste(classes_to_aggregate, collapse = "+")

  # Soma as linhas e colunas das classes selecionadas
  aggregated_row <- colSums(conf_matrix[classes_to_aggregate, ])
  aggregated_col <- rowSums(conf_matrix[, classes_to_aggregate])

  # Combina os valores agregados na nova linha e coluna
  new_row <- c(aggregated_row[!names(aggregated_row) %in% classes_to_aggregate], sum(aggregated_row[classes_to_aggregate]))
  new_col <- c(aggregated_col[!names(aggregated_col) %in% classes_to_aggregate], sum(aggregated_col[classes_to_aggregate]))


  # Cria uma nova matriz sem as classes a serem agregadas
  reduced_conf_matrix <- conf_matrix[!rownames(conf_matrix) %in% classes_to_aggregate, ]
  reduced_conf_matrix <- reduced_conf_matrix[, !colnames(conf_matrix) %in% classes_to_aggregate]


  # Adiciona a nova linha e coluna à matriz reduzida
  reduced_conf_matrix <- rbind(reduced_conf_matrix, new_row)
  reduced_conf_matrix <- cbind(reduced_conf_matrix, new_col)

  # Nomeia a nova linha e coluna com o nome da classe agregada
  rownames(reduced_conf_matrix)[nrow(reduced_conf_matrix)] <- new_class_name
  colnames(reduced_conf_matrix)[ncol(reduced_conf_matrix)] <- new_class_name

  conf_matrix <- as.matrix(reduced_conf_matrix)
  long_df <- as.data.frame(as.table(conf_matrix))

  # Renomeie as colunas para 'Prediction' e 'Reference'
  names(long_df) <- c('Prediction', 'Reference', 'Frequency')

  # Agora, crie uma tabela de contingência a partir do dataframe longo
  conf_table <- xtabs(Frequency ~ Prediction + Reference, data=long_df)

  return(conf_table)
}



#----------------- Análise exploratória das amostras -------------------------
#' Perform Data Analysis and Plotting
#'
#' This function performs data analysis and generates plots based on the input data and specified parameters.
#'
#' @param df_samples A data frame containing the samples and their classes.
#' @param classes A vector of class labels.
#' @param p_function The probability function to use ('gaussian' or 'gamma').
#' @param L Optional parameter for the 'gamma' function.
#' @param title The title for the plots.
#' @return The separability metric.
#' @import ggplot2
#' @import reshape2
#' @import scales
#' @export
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 50), Value = rnorm(100))
#' classes <- c("Class1", "Class2")
#' separability <- data_analysis(df_samples, classes, p_function = 'gaussian', title = "Amplitude")
data_analysis <- function(df_samples, classes, p_function = 'gaussian', L = NULL, title = NULL){

  # Attempt to load fonts, handle errors gracefully

  df_samples$Class = as.factor(df_samples$Class)

  data_melt = melt(df_samples)

  if (p_function == 'gamma'){
    plotting <- ggplot(data_melt, aes(Class, value, fill= Class)) +
      geom_boxplot() +
      facet_wrap(~variable, scale="free") +
      theme(panel.grid.major = element_line(colour = "#d3d3d3"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            text = element_text(),  # Use Tahoma font if available
            axis.title = element_text(face="bold", size = 10),
            axis.text.x = element_text(colour="white", size = 0),
            axis.text.y = element_text(colour="black", size = 10),
            axis.line = element_line(linewidth=1, colour = "black")) +
      theme(plot.margin = unit(c(1,1,1,1), "lines")) +
      xlab("Classe") +
      ylab(expression(bold('Values'))) +
      labs(fill = "Classe")

    print(plotting)
    separability = bhattacharyya_distance_matrix(df_samples, classes, p_function, L)

  } else {
    plotting <- ggplot(data_melt, aes(Class, value, fill= Class)) +
      geom_boxplot() +
      facet_wrap(~variable, scale="free") +
      theme(panel.grid.major = element_line(colour = "#d3d3d3"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            text = element_text(),  # Use Tahoma font if available
            axis.title = element_text(face="bold", size = 10),
            axis.text.x = element_text(colour="white", size = 0),
            axis.text.y = element_text(colour="black", size = 10),
            axis.line = element_line(linewidth=1, colour = "black")) +
      theme(plot.margin = unit(c(1,1,1,1), "lines")) +
      xlab("Classe") +
      ylab(expression(bold('Values'))) +
      labs(fill = "Classe")

    print(plotting)
    separability = bhattacharyya_distance_matrix(df_samples, classes)
  }

  return(separability)
}

#-------------------Separação de Amostras e Classes-----------------------------
#Função para separar o dataset de entrada em vários subdatasets referente a cada classe

#' Split Samples into Class-specific Matrices
#'
#' This function splits a data frame of samples into separate matrices for each specified class.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to split the samples into.
#' @return A list of matrices, each containing the samples for a specific class.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 5),
#'                         Feature1 = rnorm(10), Feature2 = rnorm(10))
#' classes <- c("Class1", "Class2")
#' class_matrices <- split_classes(df_samples, classes)

split_classes<-function(df_samples,classes){

  # Verifica se a coluna 'Class' existe no dataframe
  if (!"Class" %in% names(df_samples)) {
    stop("The dataframe must contain a 'Class' column.")
  }
  df_samples$Class <- as.factor(df_samples$Class)

  # Verifica se os valores na coluna 'Class' estão corretos e são fatores
  if (!all(classes %in% levels(df_samples$Class))) {
    stop("All specified classes must be present as levels in the 'Class' column of the dataframe.")
  }

  # Assegura que a coluna 'Class' é um fator


  classes_labels <- df_samples$Class
  classes_matrix <- as.matrix(df_samples[, -which(names(df_samples) == "Class")])

  matrix <- vector("list", length(classes))
  i = 1
  for (class in classes){
    matrix[[i]] <- classes_matrix[classes_labels == class, ]
    i = i + 1
  }

  return(matrix)
}

# split_classes <- function(df_samples) {
#   split_data <- split(df_samples[, -which(names(df_samples) == "Class")], df_samples$Class)
#
#   class_matrices <- lapply(split_data, as.matrix)
#
#   return(class_matrices)
# }

# Função para separar o dataset de entrada em treinamento e validação

# split_samples <- function(df_samples, ratio){
#
#   sample_select = sample.split(df_samples$Class, SplitRatio = ratio) # TRUE OR FALSE
#
#
#   train_samples = df_samples[sample_select ,]
#   valid_samples  = df_samples[sample_select  == F,]
#
#   #write.csv(train_samples , 'train_samples.csv')
#   #write.csv(valid_samples , 'valid_sampels.csv')
#
#   return(list(train_samples ,valid_samples))
#
# }

#' Split Samples into Training and Validation Sets
#'
#' This function splits a data frame of samples into training and validation sets based on a specified ratio or a fixed number of samples per class.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param ratio A numeric value indicating the proportion of the data to include in the training set. Ignored if `samples_per_class` is provided.
#' @param samples_per_class An optional integer specifying the fixed number of samples to use for training from each class. If provided, `ratio` is ignored.
#' @return A list with two elements: `train_samples`, a data frame of training samples, and `valid_samples`, a data frame of validation samples.
#' @importFrom caret createDataPartition
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' # Split with a ratio
#' split_result <- split_samples(df_samples, ratio = 0.7)
#' train_samples <- split_result$train_samples
#' valid_samples <- split_result$valid_samples
#'
#' # Split with a fixed number of samples per class
#' split_result_fixed <- split_samples(df_samples, samples_per_class = 50)
#' train_samples_fixed <- split_result_fixed$train_samples
#' valid_samples_fixed <- split_result_fixed$valid_samples
split_samples <- function(df_samples, ratio, samples_per_class = NULL) {
  if (!is.null(samples_per_class)) {
    # Inicializa os data frames de treinamento e validação
    train_samples <- data.frame()
    valid_samples <- data.frame()

    # Processa cada classe individualmente
    for (class in unique(df_samples$Class)) {
      class_samples <- df_samples[df_samples$Class == class, ]
      # Se o número disponível de amostras é menor ou igual ao desejado para treinamento, todas vão para treinamento
      if (nrow(class_samples) <= samples_per_class) {
        train_samples <- rbind(train_samples, class_samples)
      } else {
        # Caso contrário, seleciona amostras para treinamento e as restantes para validação
        train_indices <- sample(nrow(class_samples), samples_per_class)
        train_samples <- rbind(train_samples, class_samples[train_indices, , drop = FALSE])
        valid_samples <- rbind(valid_samples, class_samples[-train_indices, , drop = FALSE])
      }
    }
  } else {
    # Comportamento padrão com proporção
    trainIndex <- caret::createDataPartition(df_samples$Class, p = ratio, list = TRUE, times = 1)[[1]]
    train_samples <- df_samples[trainIndex, ]
    valid_samples <- df_samples[-trainIndex, ]
  }

  return(list(train_samples = train_samples, valid_samples = valid_samples))
}

#Sintaxe
# split_samples(df_samples,0.7)



#------------------- Distâncias estocásticas e dendogramas  -------------------------
# Função para calcular a distância de Bhattacharyya e Jeffries-Matusita entre duas classes


# Distância de Bhattacharryya para duas classes considerando distribuição gaussiana

#' Calculate Gaussian Bhattacharyya Distance Between Two Classes
#'
#' This function calculates the Bhattacharyya distance between two classes in a data frame of samples using a Gaussian distribution.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param class1 A character string specifying the label of the first class.
#' @param class2 A character string specifying the label of the second class.
#' @return A numeric value representing the Bhattacharyya distance between the two classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' distance <- gaussian_bhattacharyya_distance(df_samples, "Class1", "Class2")
gaussian_bhattacharyya_distance <- function(df_samples,class1,class2) {


  class_labels <- df_samples$Class

  class_1_matrix <- as.matrix(df_samples[class_labels == class1, -1 ])
  class_2_matrix <- as.matrix(df_samples[class_labels == class2, -1])


  mean_class1 <- colMeans(class_1_matrix)
  mean_class2 <- colMeans(class_2_matrix)

  # as.vector(mean_class1)
  # as.vector(mean_class2)

  cov_matrix_class1 <- cov(class_1_matrix)
  cov_matrix_class2 <- cov(class_2_matrix)


  # Calcula os determinantes das matrizes de covariância
  det_cov_class1 <- det(cov_matrix_class1)
  det_cov_class2 <- det(cov_matrix_class2)

  a = 0.125*t(mean_class1-mean_class2)
  b = solve((cov_matrix_class1+cov_matrix_class2)/2)
  c = (mean_class1-mean_class2)
  d = 0.5*log(det((cov_matrix_class1+cov_matrix_class2)/2)/
                (sqrt(det_cov_class1*det_cov_class2)))

  distance = a%*%b%*%c + d

  return(distance)

}

# Distância de Bhattacharryya para duas classes considerando distribuição gama

#' Calculate Gamma Bhattacharyya Distance Between Two Classes
#'
#' This function calculates the Bhattacharyya distance between two classes in a data frame of samples using a Gamma distribution.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param class1 A character string specifying the label of the first class.
#' @param class2 A character string specifying the label of the second class.
#' @param L A numeric value used in the calculation of the Bhattacharyya distance.
#' @return A numeric value representing the Bhattacharyya distance between the two classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rgamma(200, shape = 2, rate = 1))
#' distance <- gamma_bhattacharyya_distance(df_samples, "Class1", "Class2", L = 2)
gamma_bhattacharyya_distance <- function(df_samples,class1,class2, L) {


  class_labels <- df_samples$Class
  class_matrix <- as.matrix(df_samples[,-1])


  class_1_matrix <- class_matrix[class_labels == class1, ]
  class_2_matrix <- class_matrix[class_labels == class2, ]

  lambda_1 = mean(class_1_matrix)
  lambda_2 = mean(class_2_matrix)


  distance = log((lambda_1+lambda_2)^L/
                   ((2^L)*(lambda_1*lambda_2)^(L/2)))

  return(distance)

}

#' Calculate Bhattacharyya Distance Between Two Classes
#'
#' This function calculates the Bhattacharyya distance between two classes in a data frame of samples using either a Gaussian or Gamma distribution.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param class1 A character string specifying the label of the first class.
#' @param class2 A character string specifying the label of the second class.
#' @param p_function A character string specifying the probability distribution function to be used ('gaussian' or 'gamma'). Default is 'gaussian'.
#' @param L An optional numeric value used in the calculation of the Bhattacharyya distance for the 'gamma' distribution.
#' @return A numeric value representing the Bhattacharyya distance between the two classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' distance_gaussian <- bhattacharyya_distance(df_samples, "Class1", "Class2", p_function = 'gaussian')
#'
#' df_samples_gamma <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                                Feature1 = rgamma(200, shape = 2, rate = 1))
#' distance_gamma <- bhattacharyya_distance(df_samples_gamma, "Class1", "Class2", p_function = 'gamma', L = 2)

# função para calcular a distância de Bhattacharyya entre as duas classes
bhattacharyya_distance <- function(df_samples,class1,class2, p_function = 'gaussian', L = NULL){

  if(p_function == 'gamma'){

    B = gamma_bhattacharyya_distance(df_samples,class1,class2, L)


  }else{
    B = gaussian_bhattacharyya_distance(df_samples,class1,class2)

    }

  return(B)
}

#' Calculate Bhattacharyya Distance Matrix for All Pairs of Classes
#'
#' This function calculates the Bhattacharyya distance matrix for all pairs of classes in a data frame of samples using either a Gaussian or Gamma distribution.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the distance matrix.
#' @param p_function A character string specifying the probability distribution function to be used ('gaussian' or 'gamma'). Default is 'gaussian'.
#' @param L An optional numeric value used in the calculation of the Bhattacharyya distance for the 'gamma' distribution.
#' @return A distance matrix representing the Bhattacharyya distances between all pairs of classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2", "Class3"), each = 100),
#'                          Feature1 = rnorm(300), Feature2 = rnorm(300))
#' classes <- c("Class1", "Class2", "Class3")
#' distance_matrix_gaussian <- bhattacharyya_distance_matrix(df_samples, classes, p_function = 'gaussian')
#'
#' df_samples_gamma <- data.frame(Class = rep(c("Class1", "Class2", "Class3"), each = 100),
#'                                Feature1 = rgamma(300, shape = 2, rate = 1))
#' distance_matrix_gamma <- bhattacharyya_distance_matrix(df_samples_gamma, classes, p_function = 'gamma', L = 2)
# função para calcular a distância de Bhattacharyya entre todos os pares de classes
bhattacharyya_distance_matrix <- function(df_samples,classes, p_function = 'gaussian', L = NULL){

  if (dim(df_samples)[2] == 1){

    return()

  }
  else{
    b_distance_matrix <- matrix(nrow = length(classes),ncol = length(classes),
                                 dimnames = list(classes,classes))

    for(i in 1:length(classes)){
      for(j in 1:length(classes)){
        if(i==j){
          b_distance_matrix[i,j] = 0
        }
        else{
          b_distance_matrix[i,j]=bhattacharyya_distance(df_samples,classes[i],classes[j],p_function, L)
        }
      }
    }
    return(as.dist(b_distance_matrix))
  }
}

#' Calculate Jeffries-Matusita Distance Between Two Classes
#'
#' This function calculates the Jeffries-Matusita distance between two classes in a data frame of samples.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param class1 A character string specifying the label of the first class.
#' @param class2 A character string specifying the label of the second class.
#' @return A numeric value representing the Jeffries-Matusita distance between the two classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' jm_distance <- jeffryes_matusita_distance(df_samples, "Class1", "Class2")
# função para calcular a distância de Jeffryes_Matusita entre as duas classes
jeffryes_matusita_distance <- function(df_samples,class1,class2){

    B = bhattacharyya_distance(df_samples,class1,class2)

    jm_distance = sqrt(2*(1 - exp(-B)))

  return(jm_distance)
}


#' Calculate Jeffries-Matusita Distance Matrix for All Pairs of Classes
#'
#' This function calculates the Jeffries-Matusita distance matrix for all pairs of classes in a data frame of samples.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the distance matrix.
#' @return A distance matrix representing the Jeffries-Matusita distances between all pairs of classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2", "Class3"), each = 100),
#'                          Feature1 = rnorm(300), Feature2 = rnorm(300))
#' classes <- c("Class1", "Class2", "Class3")
#' jm_distance_matrix <- jeffryes_matusita_distance_matrix(df_samples, classes)
# função para calcular a distância de Jeffryes_Matusita entre todos os pares de classes
jeffryes_matusita_distance_matrix <- function(df_samples,classes){


  if (dim(df_samples)[2] == 1){

    return()

  }
  else{
    jm_distance_matrix <- matrix(nrow = length(classes),ncol = length(classes),
                                 dimnames = list(classes,classes))

    for(i in 1:length(classes)){
      for(j in 1:length(classes)){
        if(i==j){
          jm_distance_matrix[i,j] = 0
        }
        else{
          jm_distance_matrix[i,j]=jeffryes_matusita_distance(df_samples,classes[i],classes[j])
        }
      }
    }
    return(as.dist(jm_distance_matrix))
  }
}

#Sintaxe
#JM_Matrix = jeffryes_matusita_distance_matrix(df_samples,classes,p_function = 'gaussian', L = 2.312711)


#' Calculate Mean Jeffries-Matusita Distance Between All Classes
#'
#' This function calculates the mean Jeffries-Matusita distance between all pairs of classes in a data frame of samples.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the distance calculation.
#' @return A numeric value representing the mean Jeffries-Matusita distance between all pairs of classes.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2", "Class3"), each = 100),
#'                          Feature1 = rnorm(300), Feature2 = rnorm(300))
#' classes <- c("Class1", "Class2", "Class3")
#' mean_distance <- mean_jm_distance(df_samples, classes)
#' print(mean_distance)
mean_jm_distance <- function(df_samples, classes) {
  # sub_df_samples <- df_samples[,  c("Class", features), drop = FALSE]
  dist_matrix = jeffryes_matusita_distance_matrix(df_samples, classes)
  print(dist_matrix)
  mean_distance = mean(dist_matrix[!is.na(dist_matrix)], na.rm = TRUE)  # Ignora NA e obtém a menor distância
  return(mean_distance)
}



#' Exhaustive Search for Optimal Feature Set Based on Jeffries-Matusita Distance
#'
#' This function performs an exhaustive search to find the optimal set of features that maximizes the mean Jeffries-Matusita distance between classes in a data frame of samples.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the distance calculation.
#' @param all_features A vector of all possible feature names to consider in the search.
#' @return A list containing two data frames:
#' \itemize{
#'   \item \code{distances}: A data frame with each feature set and their corresponding mean Jeffries-Matusita distance.
#'   \item \code{results}: A data frame with the optimal feature sets, their gains, and mean Jeffries-Matusita distances.
#' }
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2", "Class3"), each = 100),
#'                          Feature1 = rnorm(300), Feature2 = rnorm(300), Feature3 = rnorm(300))
#' classes <- c("Class1", "Class2", "Class3")
#' all_features <- c("Feature1", "Feature2", "Feature3")
#' result <- exhaustive_search_jm_optimal(df_samples, classes, all_features)
#' print(result)
# conjunto de atributos ótimo
exhaustive_search_jm_optimal <- function(df_samples, classes, all_features) {
  previous_max_distance <- 0
  optimal_features <- NULL
  gain_threshold <- 0.01
  results <- data.frame(Features = character(), Gain = numeric(), JM_dist = numeric(), stringsAsFactors = FALSE)
  distances <- data.frame(Features = character(), JM_dist = numeric())
  base_features <- NULL  # Inicialmente sem atributos base

  for (num_features in 1:length(all_features)) {
    if (!is.null(base_features)) {
      # Gerar combinações que incluam os atributos base já selecionados
      combinations <- lapply(combn(setdiff(all_features, base_features), num_features - length(base_features), simplify = FALSE), function(x) c(base_features, x))
    } else {
      combinations <- combn(all_features, num_features, simplify = FALSE)
    }

    max_mean_distance_current <- -Inf
    best_feature_set_current <- NULL

    for (feature_set in combinations) {
      feature_set <- unlist(feature_set)  # Garante que o conjunto seja um vetor
      subset_df_samples <- df_samples[, c("Class", feature_set), drop = FALSE]
      distance <- mean_jm_distance(subset_df_samples, classes)
      distances <- rbind(distances, data.frame(Features = toString(feature_set), JM_dist = distance))

      if (!is.na(distance) && distance > max_mean_distance_current) {
        max_mean_distance_current <- distance
        best_feature_set_current <- feature_set
      }
    }

    gain <- if (previous_max_distance > 0) {  # Evita divisão por zero
      abs(max_mean_distance_current - previous_max_distance) / previous_max_distance
    } else {
      0  # No gain in the first round
    }
    results <- rbind(results, data.frame(Features = toString(best_feature_set_current), Gain = gain, JM_dist = max_mean_distance_current))

    if (gain <= gain_threshold && num_features > 1) {
      optimal_features <- best_feature_set_current
      #break
    }

    if (num_features == 1 || max_mean_distance_current > previous_max_distance) {
      optimal_features <- best_feature_set_current
      base_features <- best_feature_set_current  # Define os atributos base para as próximas iterações
    }

    previous_max_distance <- max_mean_distance_current
  }

  return(list(distances, results))
}


# ----------------------------- Gráficos -----------------------------------------------------


# Função para gerar dendograma em relação a Matriz de distâncias calculada previamente

#' Plot Dendrogram Based on Distance Matrix
#'
#' This function generates a dendrogram from a previously calculated distance matrix.
#'
#' @param dist_matrix A distance matrix used for hierarchical clustering.
#' @param threshold A numeric value specifying the threshold for cutting the dendrogram. Default is 1.
#' @param method A character string specifying the clustering method. Default is "ward.D2".
#' @param name A character string specifying the name/title of the dendrogram plot.
#' @return An object of class \code{hclust} which describes the tree produced by the clustering process.
#' @export
plot_dendrogram <- function(dist_matrix, threshold = 1, method = "ward.D2", name = NULL) {

  # Definindo a fonte para Georgia e negrito para os labels
  par(family = "Georgia", font.lab = 4, cex.lab = 1.2, cex.axis = 1.2, cex.main = 1,  ann = FALSE) # 'ann = FALSE' desativa anotações automáticas

  hc <- hclust(dist_matrix, method = method)
  plot(hc, main = name, hang = -1, xlab = "Classes", ylab = "Distância de Bhattacharyya", ann = FALSE)
  # abline(h = threshold, col = "red", lty = 2) # Limiar

  # Adicionando manualmente o título e labels personalizados
  title(main = name, xlab = "Classes", ylab = "Distância de Bhattacharyya", font.main = 2, font.lab = 2)

  # Resetar parâmetros gráficos após o plot
  par(family = "", font.lab = 1, ann = TRUE)

  return(hc)
}


#' Extract Hierarchy from Hierarchical Clustering Object
#'
#' This function extracts the hierarchy of clusters from an object of class \code{hclust}.
#'
#' @param hclust_obj An object of class \code{hclust} produced by hierarchical clustering.The leafs of this object have to be labeled with strings.
#' @return A list where each element represents a level of the hierarchy and contains the clusters at that level.
#' @export
#'
#' @examples
#' dist_matrix <- as.dist(matrix(runif(100), nrow = 10, dimnames = list(LETTERS[1:10], LETTERS[1:10])))
#' hc <- hclust(dist_matrix, method = "ward.D2")
#' hierarchy <- extract_hierarchy(hc)
#' print(hierarchy)
extract_hierarchy <- function(hclust_obj) {
  # Número total de observações
  n <- length(hclust_obj$labels)

  # Inicializa os grupos com as labels originais
  groups <- list()
  for (i in 1:n) {
    groups[[i]] <- hclust_obj$labels[i]
  }

  # Lista para manter o controle dos grupos em cada nível
  levels <- list()

  # Processa cada passo de mesclagem
  for (i in 1:(n - 1)) {
    # Os índices dos grupos sendo mesclados
    merge_step <- hclust_obj$merge[i, ]
    ind1 <- merge_step[1]
    ind2 <- merge_step[2]

    # Se o índice for negativo, refere-se a uma observação única, caso contrário, refere-se a um grupo já formado
    if (ind1 < 0) {
      members1 <- groups[[-ind1]]
    } else {
      members1 <- levels[[ind1]]
    }

    if (ind2 < 0) {
      members2 <- groups[[-ind2]]
    } else {
      members2 <- levels[[ind2]]
    }

    # Junta os dois grupos
    new_group <- c(members1, members2)

    # Salva o novo grupo no próximo nível
    levels[[i]] <- new_group
  }

  # Nomeia os níveis
  names(levels) <- paste0("lvl_", seq_along(levels))

  # Retorna a hierarquia dos níveis
  return(levels)
}

#Sintaxe
#plot_dendrogram(JM_Matrix,threshold = 1, method = "ward.D2")

# Gráfico BALLS

#' Create Bubble Plot from Confusion Matrix
#'
#' This function generates a bubble plot from a confusion matrix, with options for custom titles and label dictionaries.
#'
#' @param confusion_matrix A confusion matrix with class labels as row and column names.
#' @param title A character string specifying the title of the plot.
#' @param dict An optional named vector for renaming the class labels.
#' @param custom_colors A named vector for specifying the color of classes.
#' @return A ggplot object representing the bubble plot.
#' @importFrom ggplot2 ggplot aes geom_point geom_text labs theme_minimal theme element_text element_line element_blank scale_size_continuous scale_color_manual
#' @importFrom dplyr %>% group_by mutate
#' @importFrom reshape2 melt
#' @importFrom scales percent
#' @export
#'
#' @examples
#' library(reshape2)
#' library(dplyr)
#' library(ggplot2)
#' conf_matrix <- matrix(c(50, 10, 5, 2, 30, 15, 3, 2, 40),
#'                       nrow = 3, byrow = TRUE,
#'                       dimnames = list(Prediction = c("Class1", "Class2", "Class3"),
#'                                       Reference = c("Class1", "Class2", "Class3")))
#' dict <- c("Class1" = "A", "Class2" = "B", "Class3" = "C")
#' ball_graphs(conf_matrix, title = "Confusion Matrix Bubble Plot", dict = dict)
#'
#'
ball_graphs <- function(confusion_matrix, custom_colors = NULL, title = NULL, dict = NULL ) {
  confusion_matrix_long <- reshape2::melt(confusion_matrix)

  # Convert Prediction and Reference to factors
  confusion_matrix_long$Prediction <- factor(confusion_matrix_long$Prediction)
  confusion_matrix_long$Reference <- factor(confusion_matrix_long$Reference)

  if (!is.null(dict)) {
    # Apply substitutions using the factor levels
    levels(confusion_matrix_long$Prediction) <- dict[levels(confusion_matrix_long$Prediction)]
    levels(confusion_matrix_long$Reference) <- dict[levels(confusion_matrix_long$Reference)]
  }

  confusion_matrix_long <- confusion_matrix_long %>%
    dplyr::group_by(Reference) %>%
    dplyr::mutate(Relative_Frequency = value / sum(value),
                  Percent_Label = gsub("\\.", ",", scales::percent(Relative_Frequency, accuracy = 0.1)))

  # Define a default gray color palette if custom_colors is not provided
  if (is.null(custom_colors)) {
    custom_colors <- rep("gray", length(unique(confusion_matrix_long$Reference)))
    names(custom_colors) <- unique(confusion_matrix_long$Reference)
  }

  # Create the plot
  ggplot2::ggplot(confusion_matrix_long, aes(x = Reference, y = Prediction, size = Relative_Frequency)) +
    ggplot2::geom_point(aes(color = Reference), alpha = 0.6) +  # Apply color based on the Reference column
    ggplot2::geom_text(aes(label = Percent_Label), color = "black", size = 4, fontface = "bold") +  # Add formatted text
    ggplot2::scale_size_continuous(range = c(0, 30)) +  # Adjust point size
    ggplot2::labs(title = title,
                  x = "Reference",
                  y = "Prediction") +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none",  # Remove legend
                   text = ggplot2::element_text(size = 12, face = "bold"),  # Bold text
                   plot.title = ggplot2::element_text(size = 16, face = "bold"),  # Bold title
                   axis.title = ggplot2::element_text(size = 14, face = "bold"),  # Bold axis titles
                   axis.text = ggplot2::element_text(size = 12, face = "bold"),  # Bold axis text
                   panel.grid.major = ggplot2::element_line(size = 0.5, color = "gray85"),  # Major grid lines
                   panel.grid.minor = ggplot2::element_blank(),  # Remove minor grid lines
                   axis.line = ggplot2::element_line(color = "gray50", size = 0.5)) +  # Axis lines
    ggplot2::scale_color_manual(values = custom_colors)  # Manually assign colors

}


#------------------- Parameters Estimation, Probability Density and Likelihood Functions ----------------------------
#' Estimate Parameters for Each Class in a Data Frame of Samples
#'
#' This function estimates parameters for each class in a data frame of samples using different probability distribution functions.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the parameter estimation.
#' @param p_function A character string specifying the probability distribution function to be used ('gaussian', 'gamma', or 'intensity_joint_distribution'). Default is 'gaussian'.
#' @param ENL An optional numeric value used in the estimation for the 'gamma' distribution.
#' @return A list of parameter estimates for each class.
#' @export
#'
#' @examples
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' classes <- c("Class1", "Class2")
#' params_list <- parameters_estimation(df_samples, classes, p_function = 'gaussian')
parameters_estimation <- function(df_samples,classes,p_function = 'gaussian', ENL = NULL){

  num_classes <- length(classes)
  splited_classes <- split_classes(df_samples, classes)
  params_list <- list()


# Estimando parametros e gerando matrix de likelihood para cada classe
  if (p_function == 'gamma'){
    for (i in 1:num_classes){
      params <- gamma_parameters_estimation(as.matrix(splited_classes[[i]]), ENL) # ENL
      params_list[[i]] <- params
    }
  } else if (p_function == 'intensity_joint_distribution'){
    for (i in 1:num_classes){
      params <- intensity_joint_parameters_estimation(splited_classes[[i]])
      params_list[[i]] <- params
    }
  }
  else{
    for (i in 1:num_classes){
      params <- gaussian_parameters_estimation(splited_classes[[i]])
      params_list[[i]] <- params
    }
  }
  return(params_list)
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DISTRIBUTIONS

# Gaussian
# Gaussian Parameters Estimation


#' Estimate Gaussian Parameters for a Matrix of Samples
#'
#' This function estimates the parameters of a Gaussian distribution (mean vector and covariance matrix) for a given matrix of samples.
#'
#' @param matrix A numeric matrix where each row represents a sample and each column represents a feature.
#' @return A list containing the estimated mean vector and covariance matrix.
#' @export
#'
#' @examples
#' matrix <- matrix(rnorm(200), nrow = 100, ncol = 2)
#' params <- gaussian_parameters_estimation(matrix)
#' print(params)
gaussian_parameters_estimation <- function(matrix){


  mean_vector <- colMeans(as.matrix(matrix))
  cov_matrix <- cov(as.matrix(matrix))

  parameters <- list(mean = mean_vector, covariance_matrix = cov_matrix)

  return(parameters)
}

# Gaussian Likelihood

# gaussian_class_likelihood <- function(image, classes_parameters){
#
#   nrow <- nrow(image)
#   ncol <- ncol(image)
#   N <- dim(image)[3]
#   num_classes <- length(classes_parameters)
#
#   image_flatten <- matrix(image,N,nrow*ncol,1)
#   mean_vector <- list()
#   cov_matrix <- list()
#
#   for (i in 1:num_classes){
#
#     mean_vector[[i]] <- classes_parameters[[i]]$mean
#     cov_matrix [[i]] <- classes_parameters[[i]]$covariance_matrix
#   }
#
#   image_centered <- lapply(mean_vector, \(v) image_flatten - v)
#
#   det_cov <- lapply(cov_matrix , det)
#   inv_cov <- lapply(cov_matrix , solve)
#
#   rule <- list()
#   const <- 2*pi^(N/2)
#   rule <- mapply(function(i) {
#     1/(const*det_cov[[i]]^(1/2))* exp(
#       -0.5 * rowSums(crossprod(image_centered[[i]], inv_cov[[i]])*t(image_centered[[i]])))
#   }, 1:num_classes)
#
#   organized_rule <- list()
#
#   for (i in 1:num_classes){
#     organized_rule[[i]] <- matrix(unlist(rule[,i]),nrow = nrow,byrow =TRUE)
#   }
#
#   array_probabilities <- array(unlist(organized_rule), dim = c(dim(organized_rule[[1]]), num_classes))
#
#
#   extent <- image@extent
#   crs <- crs(image)
#   image_likelihood <- brick(array_probabilities,
#                                xmn = extent@xmin, xmx = extent@xmax,
#                                ymn = extent@ymin, ymx = extent@ymax, crs = crs)
#
#   return(image_likelihood)
#
# }

#' Calculate Gaussian Class Likelihood for a Data Frame of Samples
#'
#' This function calculates the Gaussian likelihood for each sample in a data frame based on the provided class parameters (mean vector and covariance matrix).
#'
#' @param dataframe A data frame where each row represents a sample and each column represents a feature.
#' @param classes_parameters A list of class parameters, where each element is a list containing a mean vector and covariance matrix for a class.
#' @return A data frame of Gaussian likelihoods for each sample, with one column per class.
#' @export
#'
#' @examples
#' dataframe <- data.frame(Feature1 = rnorm(100), Feature2 = rnorm(100))
#' class1_params <- list(mean = colMeans(dataframe), covariance_matrix = cov(dataframe))
#' class2_params <- list(mean = colMeans(dataframe + 1), covariance_matrix = cov(dataframe + 1))
#' classes_parameters <- list(class1_params, class2_params)
#' likelihood <- gaussian_class_likelihood(dataframe, classes_parameters)
#' print(likelihood)
gaussian_class_likelihood <- function(dataframe, classes_parameters) {
  num_classes <- length(classes_parameters)

  N <- ncol(dataframe)
  likelihood <- as.data.frame(matrix(nrow = nrow(dataframe), ncol = num_classes))
  names(likelihood) <- 1:num_classes

  df_matrix <- as.matrix(dataframe)

  # Adicionando um pequeno ruído para evitar matriz de covariância singular
  # for (i in 1:N) {
  #   if (var(df_matrix[, i]) == 0) {
  #     # Aplicar ruído somente na coluna i
  #     df_matrix[, i] <- df_matrix[, i] + runif(nrow(df_matrix), min = 0, max = 1e-5)
  #   }
  # }

  for (i in 1:num_classes) {
    mean_vector <- classes_parameters[[i]]$mean
    cov_matrix <- classes_parameters[[i]]$covariance_matrix


    # Centralizando os dados em torno da média
    df_centered <- sweep(df_matrix, 2, mean_vector, FUN = "-")

    # Cálculo da inversa da matriz de covariância e da constante
    inv_cov <- solve(cov_matrix)
    det_cov <- det(cov_matrix)
    const <- (2 * pi^(N / 2)) * sqrt(det_cov)

    # Cálculo o likelihood gaussiano para todos os pontos simultaneamente
    mahalanobis_distances <- rowSums((df_centered %*% inv_cov) * df_centered)
    gaussian_like <- (1/const)*exp(-0.5 * mahalanobis_distances)
    likelihood[[i]] <- gaussian_like
  }

  return(likelihood)
}



#Gamma
#Gamma Parameters Estimation

# gamma_parameters_estimation_MoM <- function(matrix){
#
#   mu_hat = mean(matrix)
#   var_hat = var(matrix)
#
#   #Gamma distribution parameters estimation by MoM
#   #alpha
#   alpha_hat <- mu_hat^2/var_hat
#
#   #beta
#   beta_hat <- mu_hat/var_hat
#   parameters <- list(alpha = alpha_hat, beta = beta_hat)
#
#   return(parameters)
#
# }

# gamma parameters estimation

#' Estimate Gamma Parameters for a Matrix of Samples
#'
#' This function estimates the parameters of a Gamma distribution (alpha and beta) for a given matrix of samples.
#'
#' @param matrix A numeric matrix where each row represents a sample and each column represents a feature.
#' @param ENL A numeric value representing the Equivalent Number of Looks (ENL).
#' @return A list containing the estimated alpha and beta parameters.
#' @export
#'
#' @examples
#' matrix <- matrix(rgamma(200, shape = 2, rate = 1), nrow = 100, ncol = 2)
#' ENL <- 2
#' params <- gamma_parameters_estimation(matrix, ENL)
#' print(params)
gamma_parameters_estimation <- function(matrix, ENL){

  mu_hat = mean(matrix)
  beta_hat = ENL/mu_hat

  parameters <- list(alpha = ENL, beta = beta_hat)

  return(parameters)

}

# gamma_parameters_estimation <- function(matrix){
#
#
#   ks_result = gamma_euler_ks_test(splitted_classes[[1]], significance_level = 0.05)
#
#   parameters <- list(alpha = ks_result$alpha, beta = ks_result$beta)
#
#   return(parameters)

# }


# Gamma Density function
# gamma_density_function <- function(pixel,alpha,beta){
#   if(pixel > 0){
#     probability = (beta^alpha)*(pixel^(alpha-1))*(exp(-beta*pixel))
#     probability = probability/gamma(alpha)
#     return(probability)
#   }else{
#   return(0)
#   }
# }

# Gamma Likelihood
# gamma_class_likelihood <- function(image, classes_parameters){
#
#   nrow <- nrow(image)
#   ncol <- ncol(image)
#   num_classes <- length(classes_parameters)
#
#   probability_arrays <- array(0, dim = c(nrow, ncol, num_classes))
#
#   # Loop sobre as classes e calculo das probabilidades
#   for (i in 1:num_classes) {
#     alpha_i <- classes_parameters[[i]]$alpha
#     beta_i <- classes_parameters[[i]]$beta
#     probability_arrays[,,i] <- apply(as.array(image), MARGIN = c(1, 2), function(pixel) gamma_density_function(pixel, alpha_i, beta_i))
#   }
#
#   extent <- image@extent
#   crs <- crs(image)
#   image_likelihood<- brick(probability_arrays ,
#                                xmn = extent@xmin, xmx = extent@xmax,
#                                ymn = extent@ymin, ymx = extent@ymax, crs = crs)
#
#   return(image_likelihood)
#
# }


#' Calculate Gamma Class Likelihood for a Data Frame of Samples
#'
#' This function calculates the Gamma likelihood for each sample in a data frame based on the provided class parameters (alpha and beta).
#'
#' @param dataframe A data frame where each row represents a sample and each column represents a feature.
#' @param classes_parameters A list of class parameters, where each element is a list containing alpha and beta for a class.
#' @return A data frame of Gamma likelihoods for each sample, with one column per class.
#' @export
#'
#' @examples
#' dataframe <- data.frame(Feature1 = rgamma(100, shape = 2, rate = 1), Feature2 = rgamma(100, shape = 2, rate = 1))
#' class1_params <- list(alpha = 2, beta = 1)
#' class2_params <- list(alpha = 2, beta = 1.5)
#' classes_parameters <- list(class1_params, class2_params)
#' likelihood <- gamma_class_likelihood(dataframe, classes_parameters)
#' print(likelihood)
gamma_class_likelihood <- function(dataframe, classes_parameters) {

  num_classes <- length(classes_parameters)
  # Lista para armazenar as probabilidades para cada classe
  likelihood <- as.data.frame(matrix(nrow = nrow(dataframe), ncol = num_classes))
  names(likelihood) <- 1:num_classes

  # Cálculo das probabilidades para cada classe
  for (i in seq_along(classes_parameters)) {
    alpha_i <- classes_parameters[[i]]$alpha
    beta_i <- classes_parameters[[i]]$beta

    gamma_like <- dgamma(as.matrix(dataframe), shape = alpha_i, rate = beta_i)
    # gamma_like <- apply(as.matrix(dataframe), c(1, 2), function(x) gamma_density_function(x, alpha_i, beta_i))

    likelihood[[i]] <- gamma_like
  }
  likelihood <- data.frame(likelihood)

  return(likelihood)
}




# Par de imagens Intensidade
# Estimação de parâmetros

#' Estimate Joint Intensity Parameters for a Pair of Images
#'
#' This function estimates the joint intensity parameters (mu1, mu2, ro2) for a pair of images.
#'
#' @param df A data frame containing two columns, each representing an image.
#' @return A list containing the estimated parameters: ro2, mu1, and mu2.
#' @export
#'
#' @examples
#' df <- data.frame(Image1 = rnorm(100), Image2 = rnorm(100))
#' params <- intensity_joint_parameters_estimation(df)
#' print(params)
intensity_joint_parameters_estimation <-  function(df) {

  sample1 <- df[, 1]
  sample2 <- df[, 2]
  mu1 <- mean(sample1)
  mu2 <- mean(sample2)


  e1 <- mean((sample1 - mu1) * (sample2 - mu2))
  e2 <- mean((sample1 - mu1)^2)
  e3 <- mean((sample2 - mu2)^2)
  ro2 <- abs(e1 / sqrt(e2 * e3))

  return(list(ro2=ro2,mu1=mu1,mu2=mu2))
}

# Density Function
# Adaptado de CORREIA,1998

# intensity_joint_class_likelihood <- function(image, class_parameters, enl) {
#   nrow <- dim(image)[1]
#   ncol <- dim(image)[2]
#   num_classes <- length(class_parameters)
#
#   # Preparando o array para as probabilidades
#   probability_arrays <- array(0, dim = c(nrow, ncol, num_classes))
#
#   # Iterar sobre cada classe para calcular a distribuição de intensidade conjunta
#   for (i in 1:num_classes) {
#     mu1 <- class_parameters[[i]]$mu1
#     mu2 <- class_parameters[[i]]$mu2
#     ro2 <- class_parameters[[i]]$ro2
#
#     # Extração das bandas da imagem
#     img1 <- matrix(image[[1]])  # Banda 1
#     img2 <- matrix(image[[2]])  # Banda 2
#
#     # Cálculo vetorizado
#     x <- (2 * enl * sqrt(ro2) / (1 - ro2)) * sqrt((img1 * img2) / (mu1 * mu2))
#     x <- pmin(x, 709)  # Evitar overflow
#
#     bess <- besselI(x, enl - 1)
#
#     a <- (enl^(enl + 1)) * (img1 * img2)^((enl - 1) / 2)
#     b <- exp(-(enl * ((img1 / mu1) + (img2 / mu2))) / (1 - ro2))
#     c <- (mu1 * mu2)^((enl + 1) / 2) * gamma(enl) * (1 - ro2) * (sqrt(ro2)^(enl - 1))
#     probability_arrays[,,i] <- (a * b * bess) / c
#   }
#
#   # Convertendo o array de probabilidades de volta para um objeto raster para cada classe
#   organized_rule <- list()
#
#   for (i in 1:num_classes){
#     organized_rule[[i]] <- matrix(unlist(probability_arrays[,,i]),nrow = nrow,byrow =TRUE)
#   }
#
#   array_probabilities <- array(unlist(organized_rule), dim = c(dim(organized_rule[[1]]), num_classes))
#
#
#   extent <- image@extent
#   crs <- crs(image)
#   image_likelihood <- brick(array_probabilities,
#                             xmn = extent@xmin, xmx = extent@xmax,
#                             ymn = extent@ymin, ymx = extent@ymax, crs = crs)
#
#   return(image_likelihood)
# }


#' Calculate Joint Intensity Likelihood for a Pair of Images
#'
#' This function calculates the joint intensity likelihood for a pair of images based on the provided class parameters.
#'
#' @param dataframe A data frame where each row represents a sample and each column represents a feature.
#' @param class_parameters A list of class parameters, where each element is a list containing mu1, mu2, and ro2 for a class.
#' @param enl A numeric value representing the Equivalent Number of Looks (ENL).
#' @return A data frame of joint intensity likelihoods for each sample, with one column per class.
#' @export
#'
#' @examples
#' dataframe <- data.frame(Image1 = rgamma(100, shape = 2, rate = 1), Image2 = rgamma(100, shape = 2, rate = 1))
#' class1_params <- list(mu1 = mean(dataframe$Image1), mu2 = mean(dataframe$Image2), ro2 = 0.5)
#' class_parameters <- list(class1_params)
#' enl <- 2
#' likelihood <- intensity_joint_likelihood(dataframe, class_parameters, enl)
#' print(likelihood)
intensity_joint_likelihood <- function(dataframe, class_parameters, enl) {
  num_classes <- length(class_parameters)

  # Converter o dataframe para matriz para operações matriciais
  df_matrix <- as.matrix(dataframe)

  # Separando em duas colunas
  img1 <- df_matrix[, 1]
  img2 <- df_matrix[, 2]

  likelihood <- as.data.frame(matrix(nrow = nrow(dataframe), ncol = num_classes))
  names(likelihood) <- 1:num_classes

  for (i in 1:num_classes) {
    mu1 <- class_parameters[[i]]$mu1
    mu2 <- class_parameters[[i]]$mu2
    ro2 <- class_parameters[[i]]$ro2

    # Considerando ro2 como o quadrado do coeficiente de correlação

    # Cálculo vetorizado
    x <- (2 * enl * sqrt(ro2) / (1 - ro2)) * sqrt((img1 * img2) / (mu1 * mu2))
    x <- pmin(x, 709)  # Limitando 'x' para evitar overflow

    bess <- besselI(x, enl - 1)

    a <- (enl^(enl + 1)) * (img1 * img2)^((enl - 1) / 2)
    b <- exp(-(enl * ((img1 / mu1) + (img2 / mu2))) / (1 - ro2))
    c <- (mu1 * mu2)^((enl + 1) / 2) * gamma(enl) * (1 - ro2) * (sqrt(ro2)^(enl - 1))

    # Calculo da probabilidade para todos os pontos simultaneamente
    prob <- (a * b * bess) / c

    likelihood[[i]] <- prob
  }

  return(likelihood)
}



#------------------------- Maximum Likelihood Classifier ------------------------

# Classificando a imagem completa
# maximum_likelihood_classifier <- function(image_probabilities) {
#
#   classif ied_image_array <- apply(as.array(image_probabilities), c(1,2), which.max)
#
#   extent <- image_probabilities@extent
#   crs <- crs(image_probabilities)
#   classified_image <- raster(classified_image_array,
#                              xmn = extent@xmin, xmx = extent@xmax,
#                              ymn = extent@ymin, ymx = extent@ymax, crs = crs)
#   return(classified_image)
#
# }


#' Maximum Likelihood Classifier
#'
#' This function classifies samples based on the maximum likelihood from a set of likelihoods.
#'
#' @param likelihoods A data frame where each column represents the likelihood of a sample belonging to a specific class.
#' @return A data frame with one column named \code{predicted_class_id}, containing the predicted class for each sample.
#' @export
#'
#' @examples
#' likelihoods <- data.frame(Class1 = c(0.1, 0.6, 0.3), Class2 = c(0.4, 0.2, 0.5), Class3 = c(0.5, 0.2, 0.2))
#' predictions <- maximum_likelihood_classifier(likelihoods)
#' print(predictions)
maximum_likelihood_classifier <- function(likelihoods) {

  predicted_class_id <- max.col(likelihoods, ties.method = "first")

  return(data.frame(predicted_class_id))
}










