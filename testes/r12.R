
# Função para realizar o teste de Kolmogorov-Smirnov com a distribuição Gama de Euler


#' Gamma Euler Distribution Kolmogorov-Smirnov Test
#'
#' This function performs a Kolmogorov-Smirnov test to check if the data follows Euler's Gamma distribution, estimates the parameters of the Gamma distribution, and optionally plots the histogram with the theoretical curve.
#'
#' @param data A numeric vector of data to be tested.
#' @param significance_level A numeric value representing the significance level for the test. Default is 0.05.
#' @param method A character string specifying the method used for parameter estimation. Default is 'mle' (maximum likelihood estimation).
#' @param plot A logical value indicating whether to plot the histogram with the theoretical Gamma distribution curve. Default is FALSE.
#' @return A list containing the estimated parameters (alpha and beta) and the p-value of the Kolmogorov-Smirnov test.
#' @import fitdistrplus
#' @export
#' @examples
#' library(fitdistrplus)
#' library(ggplot2)
#' data <- rgamma(100, shape = 2, rate = 1)
#' result <- gamma_euler_ks_test(data, significance_level = 0.05, method = 'mle', plot = TRUE)
#' print(result)
gamma_euler_ks_test <- function(data, significance_level = 0.05, method = 'mle', plot=FALSE) {
  fit <- fitdistrplus::fitdist(data, distr = "gamma", method = method)

  # Teste de Komolgorov-Smirnov
  ks_result <-  ks.test(data, "pgamma", shape = fit$estimate[1], rate = fit$estimate[2])

  # Exibição dos resultados
  cat("Euler's Gamma Distribution Parameters:\n")
  cat("Alpha:", fit$estimate[1], "\n")
  cat("Beta:", fit$estimate[2], "\n\n")

  cat("Resultado do teste de Kolmogorov-Smirnov:\n")
  print(ks_result)

  if (plot == TRUE){
    #Plotagem
    hist(data, breaks = 20, freq = FALSE, col = "lightblue", main = "Histograma e Curva Teórica da Distribuição Gama",
         xlab = "Valores", ylab = "Densidade")
    curve(dgamma(x, shape = fit$estimate[1], rate = fit$estimate[2]),
          add = TRUE, col = "red", lwd = 2)
    legend("topright", legend = c("Dados", "Curva Teórica"), col = c("lightblue", "red"), lwd = 2)
  }


  # Verificando se o valor-p é menor que o nível de significância
  if (ks_result$p.value <= significance_level) {
    cat("\nRejeitar a hipótese nula. Os dados não seguem a distribuição Gama de Euler.\n")
  } else {
    cat("\nNão há evidências para rejeitar a hipótese nula. Os dados podem seguir uma distribuição Gama de Euler.\n")
  }
  cat("--------------------------------------------\n")
  return(list(alpha = fit$estimate[1], beta = fit$estimate[2],
              p_value = as.numeric(ks_result$p.value)))
}

#Sintaxe
#fit_result<-gamma_euler_ks_test(splited_classes[[1]])


# FITTING

#' Fit Gamma Distribution to Sample Classes and Perform KS Test
#'
#' This function fits a Gamma distribution to each class in a data frame of samples, performs a Kolmogorov-Smirnov test, and optionally plots the results.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the fitting process.
#' @param significance_level A numeric value representing the significance level for the KS test. Default is 0.05.
#' @param method A character string specifying the method used for parameter estimation in the Gamma distribution. Default is 'mle' (maximum likelihood estimation).
#' @param plot A logical value indicating whether to plot the histogram with the theoretical Gamma distribution curve for each class. Default is FALSE.
#' @return A data frame with the results of the Gamma fitting and KS test for each class, including the estimated parameters (Alpha and Beta), p-value, and a decision to accept or reject the null hypothesis.
#' @import ggplot2
#' @import dplyr
#' @importFrom fitdistrplus fitdist
#'
#'
#' @export
#'
#' @examples
#' library(fitdistrplus)
#' library(ggplot2)
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100), Value = rgamma(200, shape = 2, rate = 1))
#' classes <- c("Class1", "Class2")
#' result_df <- gamma_fitting(df_samples, classes, significance_level = 0.05, method = 'mle', plot = FALSE)
#' print(result_df)
gamma_fitting <- function(df_samples, classes, significance_level = 0.05, method = 'mle', plot = FALSE) {
  num_classes = length(classes)
  splited_classes <- split_classes(df_samples,classes)
  result_df <- data.frame()

  for (i in 1:num_classes) {
    test_result <- gamma_euler_ks_test(splited_classes[[i]], significance_level,method = method)
    result_df <- rbind(result_df, c(Class = classes[i], Alpha = test_result$alpha, Beta = test_result$beta, p_value = test_result$p_value))

    if (plot == TRUE){
    # Preparar os dados para o ggplot
    df_for_plot <- data.frame(Values = splited_classes[[i]])

    # Gráfico
    g <- ggplot(df_for_plot, aes(x = Values)) +
      geom_histogram(aes(y = ..density..), binwidth = 0.01, fill = "aliceblue", color = "black") +
      stat_function(fun = dgamma, args = list(shape = test_result$alpha, rate = test_result$beta),
                    color = "red", linetype = "solid", linewidth = 1) +
      labs(title = paste("Gamma fitting - Class", classes[i]),
           x = "Values",
           y = "Density") +
      theme_minimal() +
      theme(text = element_text(size = 12, family = "Cambria"),
            plot.title = element_text(hjust = 0.5, family = "Cambria"),
            axis.title = element_text(family = "Cambria"),
            axis.text = element_text(family = "Cambria"),
            legend.position = "none") +
      annotate("text", x = Inf, y = Inf, label = sprintf("italic(alpha) == %.2f", test_result$alpha),
               hjust = 1.2, vjust = 1, size = 4, color = "darkblue", parse = TRUE) +
      annotate("text", x = Inf, y = Inf, label = sprintf("italic(beta) == %.2f", test_result$beta),
               hjust = 1.15, vjust = 2.5, size = 4, color = "darkblue", parse = TRUE) +
      annotate("text", x = Inf, y = Inf, label = sprintf("italic('p-value') == %.4f", test_result$p_value),
               hjust = 1.1, vjust = 4, size = 4, color = "darkblue", parse = TRUE) +
      scale_x_continuous(expand = expansion(mult = c(0.05, 0.1))) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.05)))


    # Exibe o gráfico
    print(g)}

  }

  # Criação do dataframe de resultados
  result_df <- as.data.frame(result_df)
  names(result_df) <- c("Class", "Alpha", "Beta", "p_value")
  result_df$H0 <- ifelse(as.numeric(result_df$p_value) > significance_level, "accept", "reject")

  print(result_df)

  return(result_df)
}

#Equivalent number of looks

#' Find the Best ENL Based on Gamma Fitting Results
#'
#' This function finds the best Equivalent Number of Looks (ENL) based on the results of fitting a Gamma distribution. It returns the alpha parameter of the class with the highest p-value that accepted the null hypothesis.
#'
#' @param fitting_result_df A data frame containing the results of Gamma fitting, including columns for Class, Alpha, Beta, p_value, and H0.
#' @return A numeric value representing the best ENL (alpha) or NA if no class accepted the null hypothesis.
#' @export
#'
#' @examples
#' fitting_result_df <- data.frame(Class = c("Class1", "Class2"),
#'                                 Alpha = c(2.5, 3.0),
#'                                 Beta = c(1.5, 2.0),
#'                                 p_value = c(0.05, 0.2),
#'                                 H0 = c("reject", "accept"))
#' best_enl <- best_ENL(fitting_result_df)
#' print(best_enl)
best_ENL <- function(fitting_result_df){

    df_accept <- subset(fitting_result_df, H0 == "accept")

  # Verificar se existem linhas após o filtro
  if(nrow(df_accept) == 0) {
    return(NA) # Retorna NA se não houver linhas com 'accept'
  }

  # Looking for the row with highest p_value
  max_pvalue_row <- df_accept[which.max(df_accept$p_value), ]

  # Saving and showing Alpha
  #cat("ENL: ",max_pvalue_row$Alpha )
  return(as.numeric(max_pvalue_row$Alpha))
}
#------------------------------------------------------------------------------

# Gamma FUNÇÃO ORIGINAL
# gamma_fitting <- function(df_samples,classes, significance_level = 0.05, method = 'mle'){
#
#   num_classes = length(classes)
#   splited_classes <- split_classes(df_samples,classes)
#   result_df <- data.frame()
#   par(mfrow = c(ceiling(num_classes / 2), 2), mar = c(4, 4, 2, 1))
#
#   for (i in 1:num_classes){
#     msg <- sprintf("Classe %s", classes[i])
#     print(msg)
#     test_result <- gamma_euler_ks_test(splited_classes[[i]], significance_level, method)
#     result_df <- rbind(result_df, c(class = classes[i], alpha = test_result$alpha, beta = test_result$beta, p_value = test_result$p_value))
#
#     # Histogram
#     hist(splited_classes[[i]], breaks = 20, freq = FALSE, col = "lightblue",
#          main = paste("Fitting - Class", classes[i]),
#          xlab = "Values", ylab = "Density")
#
#     # Gamma Density
#     curve(dgamma(x, shape = test_result$alpha, rate = test_result$beta),
#           add = TRUE, col = "red", lwd = 2)
#
#     # Information
#     legend("topright", legend = c("Data", "Gamma Distribution"), col = c("lightblue", "red"), lwd = 2)
#
#   }
#
#   # Dataframe creation
#   names(result_df) <- c("Class", "Alpha","Beta","p_value")
#   result_df$H0 <- ifelse(as.numeric(result_df$p_value) > significance_level, "accept", "reject")
#   print(result_df)
#   return(result_df)
#
# }
#



#' Iterative Gamma Fitting to Find the Best ENL
#'
#' This function performs iterative fitting of the Gamma distribution to find the best Equivalent Number of Looks (ENL) based on the results of a Kolmogorov-Smirnov test. It returns the best ENL along with the fitting results.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the fitting process.
#' @param significance_level A numeric value representing the significance level for the KS test. Default is 0.05.
#' @param method A character string specifying the method used for parameter estimation in the Gamma distribution. Default is 'mle' (maximum likelihood estimation).
#' @return A list containing the best ENL, the fitting result for the best ENL, and a data frame with ENL and acceptance count.
#' @importFrom dplyr filter arrange
#' @import ggplot2
#' @export
#'
#' @examples
#' library(fitdistrplus)
#' library(ggplot2)
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100), Value = rgamma(200, shape = 2, rate = 1))
#' classes <- c("Class1", "Class2")
#' result <- iterative_gamma_fitting(df_samples, classes, significance_level = 0.05, method = 'mle')
#' print(result)
iterative_gamma_fitting <- function(df_samples, classes, significance_level = 0.05, method = 'mle') {
  fitting_result_df <- gamma_fitting(df_samples, classes, significance_level, method = method)
  fitting_result_df_accept <- subset(fitting_result_df, H0 == "accept" & !is.na(p_value))

  # Estrutura para armazenar os resultados de aceitação e os resultados de ajuste específicos
  enl_acceptance_df <- data.frame(ENL = numeric(), Count = integer())
  specific_fitting_results_list <- list()

  if (nrow(fitting_result_df_accept) > 0) {
    fitting_result_df_accept <- fitting_result_df_accept[order(fitting_result_df_accept$p_value, decreasing = TRUE), ]

    for (enl_row in 1:nrow(fitting_result_df_accept)) {
      current_enl <- as.numeric(fitting_result_df_accept$Alpha[enl_row])
      specific_fitting_result <- gamma_fitting_specific(df_samples, classes, current_enl, significance_level)
      num_accepts <- sum(specific_fitting_result$H0 == "accept")

      # Armazena o resultado de ajuste específico na lista
      specific_fitting_results_list[[as.character(current_enl)]] <- specific_fitting_result

      enl_acceptance_df <- rbind(enl_acceptance_df, data.frame(ENL = current_enl, Count = num_accepts))
    }

    # Encontrar o ENL vencedor
    winning_enl <- enl_acceptance_df$ENL[which.max(enl_acceptance_df$Count)]
    winning_fitting_result <- specific_fitting_results_list[[as.character(winning_enl)]]

    # Mostrar o resultado de ajuste específico para o ENL vencedor
    print(paste("Resultados de ajuste específico para o ENL vencedor (ENL =", winning_enl, "):"))
    print(winning_fitting_result)

    return(list(
      Best_ENL = winning_enl,
      Fitting_Result = winning_fitting_result,
      Result_counting = enl_acceptance_df
    ))
  } else {
    cat("Nenhum resultado válido para processamento.\n")
    return(NULL)
  }
}


# Ajuste específico para ENL dado
#' Specific Gamma Fitting for Given ENL
#'
#' This function performs a specific fitting of the Gamma distribution for a given Equivalent Number of Looks (ENL) to each class in a data frame of samples.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the fitting process.
#' @param best_enl A numeric value representing the best ENL to be used for fitting.
#' @param significance_level A numeric value representing the significance level for the KS test. Default is 0.05.
#' @return A data frame with the results of the specific Gamma fitting for each class, including the shape and rate parameters, p-value, and a decision to accept or reject the null hypothesis.
#' @export
#'
#' @examples
#' library(fitdistrplus)
#' library(ggplot2)
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100), Value = rgamma(200, shape = 2, rate = 1))
#' classes <- c("Class1", "Class2")
#' best_enl <- 2
#' result_df_specific <- gamma_fitting_specific(df_samples, classes, best_enl, significance_level = 0.05)
#' print(result_df_specific)
gamma_fitting_specific <- function(df_samples, classes, best_enl, significance_level = 0.05) {
  result_df_specific <- data.frame()

  splited_classes <- split_classes(df_samples, classes)

  for (i in 1:length(classes)) {

    class_mean <- mean(splited_classes[[i]])

    shape_param <- best_enl
    rate_param <- best_enl / class_mean

    ks_test_result <- ks.test(splited_classes[[i]], "pgamma", shape = shape_param, rate = rate_param)

    result_df_specific <- rbind(result_df_specific, c(Class = classes[i], Alpha = shape_param, Beta = rate_param, p_value = ks_test_result$p.value))
  }

  names(result_df_specific) <- c("Class", "Shape", "Rate", "p_value")
  result_df_specific$H0 <- ifelse(as.numeric(result_df_specific$p_value) > significance_level, "accept", "reject")

  return(result_df_specific)
}


# -------------------------------------------------------------------------------
# NORMAL

# Função para realizar o teste de Kolmogorov-Smirnov com a distribuição Gaussiana Univariada

#' Perform Kolmogorov-Smirnov Test for Normality
#'
#' This function performs a Kolmogorov-Smirnov test to check if the data follows a Normal distribution and estimates the parameters.
#'
#' @param data A numeric vector of data to be tested.
#' @param significance_level A numeric value representing the significance level for the test. Default is 0.05.
#' @param method A character string specifying the method used for parameter estimation. Default is 'mle' (maximum likelihood estimation).
#' @return A list containing the estimated mean, standard deviation, and the p-value of the KS test.
#' @importFrom fitdistrplus fitdist
#' @export
#'
#' @examples
#' data <- rnorm(100)
#' result <- normal_ks_test(data, significance_level = 0.05, method = 'mle')
#' print(result)
normal_ks_test <- function(data, significance_level = 0.05, method = 'mle') {
  fit <- fitdist(data, distr = "norm", method = method)

  # Teste de Kolmogorov-Smirnov
  ks_result <- ks.test(data, "pnorm", mean = fit$estimate["mean"], sd = fit$estimate["sd"])

  # Exibição dos resultados
  cat("Parâmetros da Distribuição Gaussiana:\n")
  cat("Média:", fit$estimate["mean"], "\n")
  cat("Desvio Padrão:", fit$estimate["sd"], "\n\n")

  cat("Resultado do teste de Kolmogorov-Smirnov:\n")
  print(ks_result)

  # Verificando se o valor-p é menor que o nível de significância
  if (ks_result$p.value <= significance_level) {
    cat("\nRejeitar a hipótese nula. Os dados não seguem a distribuição Gaussiana.\n")
  } else {
    cat("\nNão há evidências para rejeitar a hipótese nula. Os dados podem seguir uma distribuição Gaussiana.\n")
  }
  cat("--------------------------------------------\n")
  return(list(mean = fit$estimate["mean"], sd = fit$estimate["sd"],
              p_value = as.numeric(ks_result$p.value)))
}

# FITTING


#' Fit Normal Distribution to Sample Classes and Perform KS Test
#'
#' This function fits a normal distribution to sample classes and performs the KS test.
#'
#' @param df_samples A data frame with sample data.
#' @param classes A vector of class labels.
#' @param significance_level The significance level for the KS test.
#' @param method The method for fitting the distribution (default is 'mle').
#' @param plot Logical. If TRUE, plots the fitted distribution.
#' @return A data frame with the fitting results.
#' @export
#' @examples
#' library(fitdistrplus)
#' library(ggplot2)
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100), Value = rnorm(200))
#' classes <- c("Class1", "Class2")
#' result_df <- normal_fitting(df_samples, classes, significance_level = 0.05, method = 'mle', plot = TRUE)
normal_fitting <- function(df_samples, classes, significance_level = 0.05, method = 'mle', plot = FALSE) {
  num_classes <- length(classes)
  splited_classes <- split_classes(df_samples, classes)
  result_df <- data.frame()

  for (i in 1:num_classes) {
    msg <- sprintf("Classe %s", classes[i])
    print(msg)
    test_result <- normal_ks_test(splited_classes[[i]], significance_level, method)
    result_df <- rbind(result_df, c(class = classes[i], mean = test_result$mean, sd = test_result$sd, p_value = test_result$p_value))

    if (plot == TRUE) {
      # Preparar os dados para o ggplot
      df_for_plot <- data.frame(Values = splited_classes[[i]])

      # Gráfico
      g <- ggplot(df_for_plot, aes(x = Values)) +
        geom_histogram(aes(y = ..density..), binwidth = 0.01, fill = "aliceblue", color = "black") +
        stat_function(fun = dnorm, args = list(mean = test_result$mean, sd = test_result$sd),
                      color = "red", linetype = "solid", linewidth = 1) +
        labs(title = paste("Normal fitting - Class", classes[i]),
             x = "Values",
             y = "Density") +
        theme_minimal() +
        theme(legend.position = "none") +
        annotate("text", x = Inf, y = Inf, label = sprintf("italic(mu) == %.2f", test_result$mean),
                 hjust = 1.2, vjust = 1, size = 4, color = "darkblue", parse = TRUE) +
        annotate("text", x = Inf, y = Inf, label = sprintf("italic(sigma) == %.2f", test_result$sd),
                 hjust = 1.17, vjust = 2.7, size = 4, color = "darkblue", parse = TRUE) +
        annotate("text", x = Inf, y = Inf, label = sprintf("italic('p-value') == %.4f", test_result$p_value),
                 hjust = 1.1, vjust = 4, size = 4, color = "darkblue", parse = TRUE) +
        scale_x_continuous(expand = expansion(mult = c(0.05, 0.1))) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

      # Exibe o gráfico
      print(g)
    }
  }

  # Criação do dataframe
  names(result_df) <- c("Class", "Mean", "SD", "p_value")
  result_df$H0 <- ifelse(as.numeric(result_df$p_value) > significance_level, "accept", "reject")
  print(result_df)
  return(result_df)
}


# Função para realizar o teste de Mardia para normalidade multivariada


#' Perform Mardia Test for Multivariate Normality
#'
#' This function performs Mardia's test to check if the data follows a Multivariate Normal distribution.
#'
#' @param data A data frame or matrix of data to be tested.
#' @param significance_level A numeric value representing the significance level for the test. Default is 0.05.
#' @return A list containing the p-values for skewness and kurtosis.
#' @importFrom MVN mvn
#' @export
#'
#' @examples
#' data <- matrix(rnorm(300), ncol = 3)
#' result <- normal_mardia_test(data, significance_level = 0.05)
#' print(result)
normal_mardia_test <- function(data, significance_level = 0.05) {
  # Teste de Mardia
  mardia_result <- MVN::mvn(data, mvnTest = "mardia", alpha = significance_level)

  # Exibição dos resultados
  cat("Resultado do teste de Mardia:\n")
  print(mardia_result$multivariateNormality)

  mardia_df = mardia_result$multivariateNormality

  # Verificando se o valor-p é menor que o nível de significância
  if (mardia_df$Result[3] == 'YES') {
    cat("\nNão há evidências para rejeitar a hipótese nula. Os dados podem seguir uma distribuição Gaussiana Multivariada.\n")

  } else {
    cat("\nRejeitar a hipótese nula. Os dados não seguem a distribuição Gaussiana Multivariada.\n")
  }
  cat("--------------------------------------------\n")
  #return(mardia_result$multivariateNormality)
  return(list(p_value_skewness = levels(mardia_df$`p value`)[1],
              p_value_kurtosis = levels(mardia_df$`p value`)[2] ))
}


# Ajuste para dados multivariados, utilizando o teste de Mardia

#' Fit Multivariate Normal Distribution to Sample Classes and Perform Mardia Test
#'
#' This function fits a Multivariate Normal distribution to each class in a data frame of samples and performs Mardia's test for multivariate normality.
#'
#' @param df_samples A data frame containing samples with a column named 'Class' indicating the class of each sample.
#' @param classes A vector of class labels to include in the fitting process.
#' @param significance_level A numeric value representing the significance level for the Mardia test. Default is 0.05.
#' @return A data frame with the results of the Multivariate Normal fitting and Mardia test for each class, including the p-values for skewness and kurtosis, and a decision to accept or reject the null hypothesis.
#' @importFrom MVN mvn
#' @export
#'
#' @examples
#' library(MVN)
#' df_samples <- data.frame(Class = rep(c("Class1", "Class2"), each = 100),
#'                          Feature1 = rnorm(200), Feature2 = rnorm(200))
#' classes <- c("Class1", "Class2")
#' result_df <- normal_fitting_mv(df_samples, classes, significance_level = 0.05)
#' print(result_df)
normal_fitting_mv <- function(df_samples, classes, significance_level = 0.05){

  num_classes = length(classes)
  splited_classes <- split_classes(df_samples,classes)
  result_df <- data.frame()

  for (i in 1:num_classes){
    msg <- sprintf("Classe %s", classes[i])
    print(msg)
    test_result <- normal_mardia_test(splited_classes[[i]], significance_level)
    result_df <- rbind(result_df, c(Class = classes[i],
                                    p_value_skewness = test_result$p_value_skewness,
                                    p_value_kurtosis = test_result$p_value_kurtosis))

  }

  # Criação do dataframe
  names(result_df) <- c("Class","p_value_skewness","p_value_kurtosis")
  result_df$H0 <- ifelse(as.numeric(result_df$p_value_skewness) > significance_level & as.numeric(result_df$p_value_kurtosis) > significance_level, "accept", "reject")
  print(result_df)
  return(result_df)
}



