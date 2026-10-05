
#---------------------------------Cross Validation--------------------------------------------------------
#' Cross Validation for Classification
#'
#' This function performs cross validation on a dataset of samples to evaluate the performance of different classifiers.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the validation process.
#' @param sample_proportion A numeric value representing the proportion of samples to be used for training. Default is 0.7.
#' @param p_function A character string specifying the probability distribution function to be used ('gaussian', 'gamma', or 'intensity_joint_distribution'). Default is 'gaussian'.
#' @param n_iteracoes An integer value specifying the number of iterations for cross validation. Default is 10.
#' @param ENL An optional numeric value used in the estimation for the 'gamma' and 'intensity_joint_distribution' distributions.
#' @param seed An optional numeric value or vector of seeds for reproducibility. If provided, the length of the vector determines the number of iterations.
#' @param aggregation An optional list of class aggregations to be applied to the confusion matrix.
#' @param external_valid_samples An optional data frame containing external validation samples.
#' @param n_samples An optional numeric value specifying the number of samples per class for validation.
#' @param fill_names An optional character vector for renaming the classes in the confusion matrix.
#' @param type A character string indicating the type of cross validation ('variable' or 'fixed'). Default is 'variable'.
#' @return A list containing the overall accuracy, F1 macro scores, producers' accuracy, users' accuracy, F1 scores, and the seeds for the best, median, and third quartile accuracies.
#' @importFrom caret createDataPartition
#' @export
#'
#' @examples
#' # Example usage of cross_validation function
#' df_samples <- data.frame(Class = rep(c("A", "B"), each = 25), matrix(rnorm(200), nrow = 50))
#' colnames(df_samples) <- c("Class", paste0("Feature", 1:4))
#' classes <- c("A", "B")
#' result <- cross_validation(df_samples, classes, sample_proportion = 0.7,
#'                            p_function = 'gaussian', n_iteracoes = 10, seed = NULL, type = 'variable')
#' print(result)

cross_validation <- function(df_samples, classes, sample_proportion = 0.7,
                             p_function = 'gaussian', n_iteracoes = 10, ENL = NULL, seed = NULL,
                             aggregation = NULL, external_valid_samples = NULL, n_samples = NULL,
                             fill_names = NULL, type = 'variable') {

  # Se um valor de seed foi fornecido, usar set.seed() para garantir reprodutibilidade
  if (!is.null(seed)) {
    set.seed(seed)
    n_iteracoes = length(seed)
  }

  # Pré-alocação de memória para dataframes


  seeds <- numeric(n_iteracoes) # Armazena as sementes usadas em cada iteração

  best_accuracy <- 0
  best_accuracy_seed <- NA
  third_quartile_seed <- NA
  median_accuracy_seed <- NA


  # Se for fixo, a variação é só na conjunto de teste # Só faz sentido ser fixo se tiver um conjunto externo de avaliação
  if (type == 'variable'){
    sample_proportion_train <- sample_proportion
    sample_proportion_test <- sample_proportion
  }else if(type == 'fixed'){
    sample_proportion_train <- 1
    sample_proportion_test <- sample_proportion
  }

  for (i in seq_len(n_iteracoes)) {
    cat('Iteração', i, "\n")

    # Aqui uso i como a semente para cada iteração, se um seed global não foi definido
    seeds[i] <- ifelse(is.null(seed), i, seed)
    set.seed(seeds[i])

    #Dividindo as amostras em treinamento e teste de forma aleatória e controlada
    splitted_points <- split_samples(df_samples, sample_proportion_train)
    train_points <- splitted_points[[1]]

    if (is.null(external_valid_samples)){
    test_points <- splitted_points[[2]]
    valid_points <- test_points[sapply(test_points, is.numeric)]
    } else{
      test_points <-  split_samples(external_valid_samples, sample_proportion_test, samples_per_class = n_samples)[[1]]
      valid_points <- test_points[sapply(test_points, is.numeric)]
    }
    # Cálculo dos parâmetros

    # params_list <- parameters_estimation(train_points, classes, p_function)
    params_list <- parameters_estimation(train_points, classes, p_function, ENL)

    likelihoods <- if (p_function == 'gamma') {
      gamma_class_likelihood(valid_points, params_list)
    }else if(p_function == 'intensity_joint_distribution'){
      intensity_joint_likelihood(valid_points, params_list, ENL)
    }else {
      gaussian_class_likelihood(valid_points, params_list)
    }

    predicted_classes <- maximum_likelihood_classifier(likelihoods)
    df_temp <- match_classes(predicted_classes,test_points,classes, replace_names = fill_names) #AQUI

    # Matriz de confusão
    confusion_matrix <- confusion_matrix_table(df_temp)
    #print(confusion_matrix)

    if(!is.null(aggregation)){
      for (j in 1:length(aggregation)) {
        confusion_matrix <- aggregate_classes(confusion_matrix, aggregation[[j]])
      }
      print(confusion_matrix)
    }

    metrics <- metrics_calculation(confusion_matrix)

    if (i == 1) {
      overall_accuracies <- numeric(n_iteracoes)
      f1_macros <- numeric(n_iteracoes)
      producers_accuracies <- data.frame(matrix(ncol = ncol(confusion_matrix), nrow = n_iteracoes))
      users_accuracies <- data.frame(matrix(ncol = ncol(confusion_matrix), nrow = n_iteracoes))
      f1_scores <- data.frame(matrix(ncol = ncol(confusion_matrix), nrow = n_iteracoes))

      colnames(producers_accuracies) <- names(metrics$producers_accuracy)
      colnames(users_accuracies) <- names(metrics$users_accuracy)
      colnames(f1_scores) <- names(metrics$f1_score)
      if (all(colnames(producers_accuracies) == colnames(users_accuracies)) &
          all(colnames(users_accuracies) == colnames(f1_scores))){
        col_names <- colnames(producers_accuracies)
      }

    }

    # Armazenar resultados nos dataframes
    overall_accuracies[i] <- metrics$overall_accuracy
    f1_macros[i] <- metrics$f1_macro
    producers_accuracies[i, ] <- unlist(metrics$producers_accuracy[col_names])
    users_accuracies[i, ] <- unlist(metrics$users_accuracy[col_names])
    f1_scores[i, ] <- unlist(metrics$f1_score[col_names])

    if ((!is.na(metrics$f1_macro) && metrics$f1_macro > best_accuracy) ) {
      best_accuracy <- metrics$f1_macro
      best_accuracy_seed <- i
    }
  }
  # Número total de iterações para determinar posições mediana e de 3º quartil
  total_iterations <- length(overall_accuracies)
  median_position <- floor(total_iterations / 2)
  third_quartile_position <- ceiling(0.75 * total_iterations)

  # Ordena as acurácias e obtém seus índices originais
  ordered_indices <- order(f1_macros)

  # Calcula os índices para a mediana e o terceiro quartil
  median_index <- ordered_indices[median_position]
  third_quartile_index <- ordered_indices[third_quartile_position]

  # Obtém as seeds correspondentes
  median_accuracy_seed <- seeds[median_index]
  third_quartile_seed <- seeds[third_quartile_index]


  return(list(
    overall_accuracy = overall_accuracies,
    f1_macro = f1_macros,
    producers_accuracy = producers_accuracies,
    users_accuracy = users_accuracies,
    f1_score = f1_scores,
    median_accuracy_seed = median_accuracy_seed,
    third_quartile_seed = third_quartile_seed,
    best_accuracy_seed = best_accuracy_seed
  ))
}



#' Calculate Evaluation Metrics from Confusion Matrix
#'
#' This function calculates evaluation metrics from a confusion matrix, including overall accuracy, F1 macro score, producers' accuracy, users' accuracy, and F1 scores.
#'
#' @param confusion_matrix A matrix representing the confusion matrix of the classification results.
#' @return A list containing the overall accuracy, F1 macro score, producers' accuracy, users' accuracy, and F1 scores.
#' @export
#'
#' @examples
#' confusion_matrix <- matrix(c(50, 10, 5, 35), nrow = 2, byrow = TRUE)
#' colnames(confusion_matrix) <- c("Class1", "Class2")
#' rownames(confusion_matrix) <- c("Class1", "Class2")
#' result <- metrics_calculation(confusion_matrix)
#' print(result)
metrics_calculation <- function(confusion_matrix) {

  confusion_mat <- as.matrix(confusion_matrix)
  # print(confusion_mat)

  #total_samples = sum(confusion_mat)
  nj = colSums(confusion_mat)

  confusion_mat <- sweep(confusion_mat, 2, nj,"/")
  # print(confusion_mat)
  total = sum(confusion_mat)
  nj = colSums(confusion_mat)
  ni = rowSums(confusion_mat)

  users_accuracy <- diag(confusion_mat)/ni
  producers_accuracy <- diag(confusion_mat)
  overall_accuracy <- sum(diag(confusion_mat)) / total
  # Substituir NaN por 0
  producers_accuracy[is.nan(producers_accuracy)] <- 0
  users_accuracy[is.nan(users_accuracy)] <- 0

  f1_scores <- numeric(length = length(ni))
  f1_scores = 2*users_accuracy*producers_accuracy/(producers_accuracy+users_accuracy)

  f1_score_macro <- mean(f1_scores)

  metrics <- list(overall_accuracy = overall_accuracy,
                  f1_macro = f1_score_macro,
    producers_accuracy = producers_accuracy,
    users_accuracy = users_accuracy,
    f1_score = f1_scores
  )

  return(metrics)

}


#' Calculate Statistics for Each Column of a Data Frame
#'
#' This function calculates the mean, standard deviation, and 95% confidence interval for each column of a data frame.
#'
#' @param df A data frame with numeric columns.
#' @return A data frame containing the mean, standard deviation, and 95% confidence interval (lower and upper) for each column.
#' @export
#'
#' @examples
#' df <- data.frame(A = rnorm(100), B = rnorm(100))
#' stats <- calc_stats(df)
#' print(stats)
calc_stats <- function(df) {
  stats_df <- data.frame(
    Mean = colMeans(df, na.rm = TRUE),
    SD = apply(df, 2, sd, na.rm = TRUE),
    ci_lw  = apply(df, 2, quantile, probs = 0.025, na.rm = TRUE),
    ci_up = apply(df, 2, quantile, probs = 0.975, na.rm = TRUE)
  )
  return(stats_df)
}

#' Calculate Statistics for a Numeric Vector
#'
#' This function calculates the mean, standard deviation, and 95% confidence interval for a numeric vector.
#'
#' @param vec A numeric vector.
#' @return A data frame containing the mean, standard deviation, and 95% confidence interval (lower and upper) for the vector.
#' @export
#'
#' @examples
#' vec <- rnorm(100)
#' stats <- calc_stats_vector(vec)
#' print(stats)
calc_stats_vector <- function(vec) {
  stats_vec <- data.frame(
    Mean = mean(vec, na.rm = TRUE),
    SD = sd(vec, na.rm = TRUE),
    ci_lw = quantile(vec, probs = 0.025, na.rm = TRUE),
    ci_up = quantile(vec, probs = 0.975, na.rm = TRUE)
  )
  return(stats_vec)
}


#' Calculate Cross-Validation Statistics
#'
#' This function calculates statistics (mean, standard deviation, and 95% confidence interval) for various metrics obtained from cross-validation.
#'
#' @param metrics_list A list containing data frames or vectors of metrics from cross-validation, including overall accuracy, F1 macro score, producers' accuracy, users' accuracy, and F1 scores.
#' @return A list containing the overall statistics, F1 macro statistics, and statistics per class.
#' @export
#'
#' @examples
#' # Example data
#' metrics_list <- list(
#'   overall_accuracy = rnorm(10, 0.8, 0.05),
#'   f1_macro = rnorm(10, 0.75, 0.06),
#'   producers_accuracy = data.frame(Class1 = rnorm(10, 0.8, 0.05), Class2 = rnorm(10, 0.7, 0.06)),
#'   users_accuracy = data.frame(Class1 = rnorm(10, 0.85, 0.05), Class2 = rnorm(10, 0.75, 0.06)),
#'   f1_score = data.frame(Class1 = rnorm(10, 0.82, 0.05), Class2 = rnorm(10, 0.77, 0.06))
#' )
#' stats_results <- cross_validation_stats(metrics_list)
#' print(stats_results)
cross_validation_stats <- function(metrics_list) {
  # Extrair dataframes de métricas

  overall_accuracy_df <- metrics_list$overall_accuracy
  f1_score_macro_df <- metrics_list$f1_macro


  overall_stats <- calc_stats_vector(overall_accuracy_df)
  f1_macro_stats <- calc_stats_vector(f1_score_macro_df)

  producers_accuracy_df <- metrics_list$producers_accuracy
  users_accuracy_df <- metrics_list$users_accuracy
  f1_score_df <- metrics_list$f1_score

  producers_stats <- calc_stats(producers_accuracy_df)
  users_stats <- calc_stats(users_accuracy_df)
  f1_stats <- calc_stats(f1_score_df)

  names(producers_stats) <- paste0(names(producers_stats), "_PA")
  names(users_stats) <- paste0(names(users_stats), "_UA")
  names(f1_stats) <- paste0(names(f1_stats), "_F1")

  # Concatenando os dataframes por colunas (assumindo que todos têm as mesmas linhas nas mesmas ordens)
  combined_df <- cbind(producers_stats, users_stats, f1_stats)

  # Combina os resultados em uma lista
  stats_results <- list(
    Overall_Stats = overall_stats,
    F1_Macro_Stats = f1_macro_stats,
    Stats_per_class = combined_df
  )


  return(stats_results)
}



#---------------------------------Confusion Matrix Table--------------------------------------------------------
#
#' Generate Confusion Matrix and Accuracy Metrics from Predictions
#'
#' This function takes a dataframe with predicted and actual classes to generate a confusion matrix and calculate
#' various accuracy metrics. It returns a list containing data frames for overall accuracy, the confusion matrix,
#' producer's accuracy, and user's accuracy.
#'
#' @param df_temp A dataframe with at least two columns: `predicted_class` and `Class`,
#'        where `predicted_class` contains the model's predictions and `Class` contains the true class labels.
#'
#' @details
#' The function first creates a confusion matrix using the `confusionMatrix` function from the `caret` package,
#' comparing the predicted classes against the actual classes. It then calculates the overall accuracy, producer's
#' accuracy, and user's accuracy from this confusion matrix. Each of these metrics is returned as a separate dataframe
#' within a list, facilitating further analysis or reporting.
#'
#' @return A list of data frames, where each data frame represents a different set of metrics:
#'         overall accuracy, the confusion matrix itself, producer's accuracy, and user's accuracy.
#' @export
#' @importFrom caret confusionMatrix
#' @examples
#' df_temp <- data.frame(predicted_class = sample(c("A", "B"), 100, replace = TRUE),
#'                       Class = sample(c("A", "B"), 100, replace = TRUE))
#' confusion_df <- confusion_matrix_table(df_temp)
#' print(confusion_df)
confusion_matrix_table <- function(df_temp){

  # Criar matriz de confusão
  confusion_matrix <- confusionMatrix(as.factor(df_temp$predicted_class), as.factor(df_temp$Class))
  # metrics <- metrics_calculation(confusion_matrix) #metrics calculation foi definida anteriormente
  #
  # oa_df <- as.data.frame(metrics$overall_accuracy)
  # f1_macro_df <- as.data.frame(metrics$f1_macro)
   confusion_df <- as.data.frame.matrix(confusion_matrix$table)
  # producer_accuracy_df <- as.data.frame(metrics$producers_accuracy)
  # user_accuracy_df <- as.data.frame(metrics$users_accuracy)
  # f1_score_df <- as.data.frame(metrics$f1_score)


  #return(list(table = confusion_df, OA = oa_df,F1_M = f1_macro_df, PA = producer_accuracy_df,UA = user_accuracy_df,F1 = f1_score_df))
  return(table = confusion_df)
}



#---------------------------------Match Classes-------------------------------------------------------------------------
#' Match Predicted Classes with Actual Classes and Convert to Class Names
#'
#' This function takes numeric predicted class indices and a test dataset, and maps these indices
#' to their corresponding class names based on a provided vector of class names. It returns a dataframe
#' where both predicted and actual class indices are replaced with class names, and both are converted
#' to factors with levels corresponding to all unique class names present.
#'
#' @param predicted_classes A numeric vector or a column in a dataframe that contains the predicted class indices.
#' @param test_points A dataframe that must contain at least one column named 'Class', which includes the actual class indices.
#' @param classes A character vector containing the names of the classes. The order of the names in this vector
#'        should correspond to the numeric indices of the classes (i.e., the first name corresponds to index 1, and so on).
#' @param replace_names A dictionary to replace the names
#' @details
#' The function first combines the `predicted_classes` with the actual class indices from `test_points` into a new dataframe.
#' Then, it replaces the numeric indices in the `predicted_classes` column with their corresponding class names from the `classes` vector.
#' After mapping the numeric indices to class names, it sorts and finds all unique class names to set the factor levels for both
#' predicted and actual classes, ensuring consistent representation and facilitating comparison or further analysis.
#'
#' The resulting dataframe has two columns: 'predicted_class' and 'Class', both as factors with the same levels (the unique class names).
#' This setup is particularly useful for subsequent data analysis tasks that require comparing predicted and actual classes, such as
#' generating confusion matrices or calculating classification metrics.
#'
#' @import dplyr
#' @importFrom dplyr mutate recode
#'
#' @return A dataframe with two columns: 'predicted_class' and 'Class'. Both columns are converted to factors with levels corresponding
#'         to all unique class names derived from the numeric indices provided in the input. This facilitates direct comparison
#'         and analysis of classification results.
#'
match_classes <- function(predicted_classes,test_points,classes, replace_names = NULL ){


  df_temp <- cbind(predicted_classes,test_points['Class'])
  #df_temp$Class <- match(df_temp$Class, classes)
  df_temp$predicted_class <- classes[df_temp$predicted_class]

  if (!is.null(replace_names)) {
    df_temp = replace_class_names(df_temp,'predicted_class',replace_names)

  }

  all_classes <- sort(unique(c(df_temp$predicted_class, df_temp$Class)))

  df_temp$predicted_class <- factor(df_temp$predicted_class, levels = all_classes)
  df_temp$Class <- factor(df_temp$Class, levels = all_classes)

  return(df_temp)
}


#' Replace Class Names
#'
#' This function replaces the class names in a specific column of a dataframe according to a provided dictionary.
#'
#' @param data A dataframe containing the column whose values will be replaced.
#' @param column A string specifying the name of the column whose values will be replaced.
#' @param dictionary A named vector where the names represent the old values and the values represent the new class names.
#' @return The dataframe with the specified column containing the new class names.
#' @import dplyr
#' @importFrom dplyr mutate recode
#' @examples
#' data <- data.frame(Class = c("A", "B", "C"))
#' dictionary <- c("A" = "Class1", "B" = "Class2", "C" = "Class3")
#' data <- replace_class_names(data, "Class", dictionary)
#' @export
replace_class_names <- function(data, column, dictionary) {
  # Assegura que a coluna especificada existe no dataframe
  if (!column %in% names(data)) {
    stop("A coluna especificada não existe no dataframe.")
  }

  # Substitui os nomes das classes conforme o dicionário
  data <- data %>%
    mutate(!!column := recode(!!as.name(column), !!!dictionary))

  return(data)
}


