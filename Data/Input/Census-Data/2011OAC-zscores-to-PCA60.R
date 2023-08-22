# Libraries ---------------------------------------------------------------

library(logr)
library(testthat)
library(tidyverse)
library(lubridate)
library(knitr)
library(skimr)


# Clean-up environment ----------------------------------------------------

rm(list = ls())



# Information for logging -------------------------------------------------

# Log file
logr_file <- 
  file.path( 
    paste0(
      "GreaterLondon_2011_OAC_Raw_uVariables--zscores-to-PCA60--",
      # Timestamp
      now() %>% str_replace_all("\\s", "-") %>% str_replace_all(":", ""),
      ".log"
    )
  )

# Log object
logr_object <- 
  log_open(
    logr_file,
    autolog = TRUE, 
    show_notes = FALSE
  )


# Load data ---------------------------------------------------------------

oac_zscores <-
  read_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--zscores-with-colnames.csv")

oac_zscores %>% 
  skim() %>% 
  put()


# Principal Component Analysis --------------------------------------------

oac_zscores_pca <-
  oac_zscores %>%
  select(-OA11CD) %>%
  prcomp() 

oac_zscores_pca %>% 
  summary() %>%
  put()
  
oac_zscores_with_pca <- 
  oac_zscores %>%
  dplyr::bind_cols(
    oac_zscores_pca %$% x %>% as.data.frame()
  )

oac_zscores_with_pca %>% 
  skim() %>% 
  put()

oac_pca60 <-
  oac_zscores_with_pca %>% 
  select(OA11CD, PC1:PC60) %>% 
  rename_with(
    ~ paste0("uzs_", .x),
    PC1:PC60
  )

oac_pca60 %>% 
  skim() %>% 
  put()




## Save PCA60 data --------------------------------------------------------

oac_pca60 %>% 
  write_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--zscores-to-PCA60.csv")



# Clean-up and log --------------------------------------------------------

log_close()
writeLines(readLines(logr_object, encoding = "UTF-8"))
  