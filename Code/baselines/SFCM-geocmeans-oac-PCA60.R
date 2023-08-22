# install.packages("geocmeans")
# install.packages("ggpubr")

library(tidyverse)
library(sf)
library(geocmeans)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(viridis)
library(spdep)

rm(list = ls())

clustering_seed <- 456

process_PCA60_m_queen <- 1.20
process_PCA60_alpha_queen <- 1.80

process_PCA60_m_knn8 <- 1.20
process_PCA60_alpha_knn8 <- 1.80

process_PCA60_m_mdt <- 1.20
process_PCA60_alpha_mdt <- 1.80

process_PCA60_m_test <- FALSE
process_PCA60_alpha_test <- FALSE
process_PCA60_plot_map <- FALSE


# Load data ---------------------------------------------------------------

oac_pca60 <- 
  read_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--zscores-to-PCA60.csv")

oac_geo <- 
  st_read("Data/Input/London-Map-OAs/ons-oa-geo-london_valid.geojson") %>% 
  st_transform(27700) %>% 
  select(OA11CD) %>% 
  left_join(oac_pca60) %>% 
  as("Spatial")

oac_data_for_clustering <- 
  oac_geo@data %>% 
  select(uzs_PC1:uzs_PC60)



# Test for m --------------------------------------------------------------

if (process_PCA60_m_test) {
  
  oac60pca_fcm_dfindices <- 
    selectParameters(
      algo = "FCM", 
      data = oac_data_for_clustering,
      k = 8, 
      m = seq(1.05, 3, 0.05),
      spconsist = FALSE,
      standardize = FALSE,
      tol = 0.0001, 
      verbose = TRUE, 
      seed = clustering_seed
    )
  
  # https://cran.r-project.org/web/packages/geocmeans/vignettes/introduction.html#spatial-c-means-and-generalized-c-means
  # "Let us just stress that 
  # a larger silhouette index indicates a better classification, 
  # and a smaller Xie and Beni index indicates a better classification."
  
  p_oac60pca_fcm_dfindices_ein <-
    ggplot(oac60pca_fcm_dfindices) +
    geom_smooth(aes(x=m,y=Explained.inertia), color = "black") +
    geom_point(aes(x=m,y=Explained.inertia), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_fcm_dfindices_ein--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_fcm_dfindices_ein
  )
  
  p_oac60pca_fcm_dfindices_sil <-
    ggplot(oac60pca_fcm_dfindices) +
    geom_smooth(aes(x=m,y=Silhouette.index), color = "black") +
    geom_point(aes(x=m,y=Silhouette.index), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_fcm_dfindices_sil--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_fcm_dfindices_sil
  )
  
  p_oac60pca_fcm_dfindices_xbi <-
    ggplot(oac60pca_fcm_dfindices) +
    geom_smooth(aes(x=m,y=XieBeni.index), color = "black") +
    geom_point(aes(x=m,y=XieBeni.index), color = "red") +
    scale_y_log10()
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_fcm_dfindices_xbi--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_fcm_dfindices_xbi
  )
  
}

# Spatial c-means and generalized c-means ---------------------------------


## SFCM Queens ------------------------------------------------------------

oac_geo_queen_nb <- poly2nb(oac_geo, queen = TRUE)
oac_geo_queen_listw <- nb2listw(oac_geo_queen_nb,style="W",zero.policy = TRUE)


### Test alpha ------------------------------------------------------------

if (process_PCA60_alpha_test) {

  oac60pca_sfcm_queen_dfindices <- 
    selectParameters(
      algo = "SFCM", 
      data = oac_data_for_clustering,
      k = 8, 
      m = 1.5, 
      alpha = seq(0,3,0.05),
      nblistw = oac_geo_queen_listw, 
      standardize = FALSE,
      tol = 0.0001, 
      verbose = TRUE, 
      seed = clustering_seed
    )
  
  # https://cran.r-project.org/web/packages/geocmeans/vignettes/introduction.html#spatial-c-means-and-generalized-c-means
  # "Let us just stress that 
  # a larger silhouette index indicates a better classification, 
  # and a smaller Xie and Beni index indicates a better classification."
  
  p_oac60pca_sfcm_queen_dfindices_spc <-
    ggplot(oac60pca_sfcm_queen_dfindices) +
    geom_smooth(aes(x=alpha,y=spConsistency), color = "black") +
    geom_point(aes(x=alpha,y=spConsistency), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_queen_dfindices_spc--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_queen_dfindices_spc
  )
  
  p_oac60pca_sfcm_queen_dfindices_ein <-
    ggplot(oac60pca_sfcm_queen_dfindices) +
    geom_smooth(aes(x=alpha,y=Explained.inertia), color = "black") +
    geom_point(aes(x=alpha,y=Explained.inertia), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_queen_dfindices_ein--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_queen_dfindices_ein
  )
  
  p_oac60pca_sfcm_queen_dfindices_sil <-
    ggplot(oac60pca_sfcm_queen_dfindices) +
    geom_smooth(aes(x=alpha,y=Silhouette.index), color = "black") +
    geom_point(aes(x=alpha,y=Silhouette.index), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_queen_dfindices_sil--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_queen_dfindices_sil
  )
  
  p_oac60pca_sfcm_queen_dfindices_xbi <-
    ggplot(oac60pca_sfcm_queen_dfindices) +
    geom_smooth(aes(x=alpha,y=XieBeni.index), color = "black") +
    geom_point(aes(x=alpha,y=XieBeni.index), color = "red") +
    scale_y_log10()
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_queen_dfindices_xbi--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_queen_dfindices_xbi
  )

}


### Run -------------------------------------------------------------------

oac60pca_sfcm_queen <- 
  SFCMeans(
    oac_data_for_clustering, 
    oac_geo_queen_listw, 
    k = 8, 
    m = process_PCA60_m_queen, 
    alpha = process_PCA60_alpha_queen,
    standardize = FALSE,
    tol = 0.0001, 
    seed = clustering_seed
  )


if (process_PCA60_plot_map) {

  oac60pca_sfcm_maps <- 
    mapClusters(geodata = oac_geo, object = oac60pca_sfcm_queen$Belongings,undecided = 0)
  oac60pca_sfcm_maps$ClusterPlot

}


## SFCM KNN8 --------------------------------------------------------------

oac_geo_knn8_nb <- 
  oac_geo %>% 
  coordinates() %>% 
  knearneigh(k = 8) %>% 
  knn2nb()
oac_geo_knn8_listw <- nb2listw(oac_geo_knn8_nb,style="W",zero.policy = TRUE)


### Test alpha ------------------------------------------------------------

if (process_PCA60_alpha_test) {

  oac60pca_sfcm_knn8_dfindices <- 
    selectParameters(
      algo = "SFCM", 
      data = oac_data_for_clustering,
      k = 8, 
      m = 1.5, 
      alpha = seq(0,3,0.05),
      nblistw = oac_geo_knn8_listw, 
      standardize = FALSE,
      tol = 0.0001, 
      verbose = TRUE, 
      seed = clustering_seed
    )
  
  # https://cran.r-project.org/web/packages/geocmeans/vignettes/introduction.html#spatial-c-means-and-generalized-c-means
  # "Let us just stress that 
  # a larger silhouette index indicates a better classification, 
  # and a smaller Xie and Beni index indicates a better classification."
  
  p_oac60pca_sfcm_knn8_dfindices_spc <-
    ggplot(oac60pca_sfcm_knn8_dfindices) +
    geom_smooth(aes(x=alpha,y=spConsistency), color = "black") +
    geom_point(aes(x=alpha,y=spConsistency), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_knn8_dfindices_spc--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_knn8_dfindices_spc
  )
  
  p_oac60pca_sfcm_knn8_dfindices_ein <-
    ggplot(oac60pca_sfcm_knn8_dfindices) +
    geom_smooth(aes(x=alpha,y=Explained.inertia), color = "black") +
    geom_point(aes(x=alpha,y=Explained.inertia), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_knn8_dfindices_ein--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_knn8_dfindices_ein
  )
  
  p_oac60pca_sfcm_knn8_dfindices_sil <-
    ggplot(oac60pca_sfcm_knn8_dfindices) +
    geom_smooth(aes(x=alpha,y=Silhouette.index), color = "black") +
    geom_point(aes(x=alpha,y=Silhouette.index), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_knn8_dfindices_sil--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_knn8_dfindices_sil
  )
  
  p_oac60pca_sfcm_knn8_dfindices_xbi <-
    ggplot(oac60pca_sfcm_knn8_dfindices) +
    geom_smooth(aes(x=alpha,y=XieBeni.index), color = "black") +
    geom_point(aes(x=alpha,y=XieBeni.index), color = "red") +
    scale_y_log10()
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_knn8_dfindices_xbi--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_knn8_dfindices_xbi
  )

}


### Run -------------------------------------------------------------------

oac60pca_sfcm_knn8 <- 
  SFCMeans(
    oac_data_for_clustering, 
    oac_geo_knn8_listw, 
    k = 8, 
    m = process_PCA60_m_knn8, 
    alpha = process_PCA60_alpha_knn8,
    standardize = FALSE,
    tol = 0.0001, 
    seed = clustering_seed
  )


if (process_PCA60_plot_map) {
  
  oac60pca_sfcm_knn8_maps <- 
    mapClusters(geodata = oac_geo, object = oac60pca_sfcm_knn8$Belongings,undecided = 0)
  oac60pca_sfcm_knn8_maps$ClusterPlot

}


## SFCM distance ----------------------------------------------------------

oac_geo_coords <- 
  oac_geo %>% 
  coordinates() 

oac_geo_min_dist_thr <- 
  oac_geo_coords %>% 
  knearneigh() %>% 
  knn2nb() %>% 
  nbdists(oac_geo_coords) %>% 
  unlist() %>% 
  max()

oac_geo_mdt_nb <- 
  oac_geo %>% 
  coordinates() %>% 
  dnearneigh(0, oac_geo_min_dist_thr)

oac_geo_mdt_listw <- nb2listw(oac_geo_mdt_nb,style="W",zero.policy = TRUE)


### Test alpha ------------------------------------------------------------

if (process_PCA60_alpha_test) {
  
  oac60pca_sfcm_mdt_dfindices <- 
    selectParameters(
      algo = "SFCM", 
      data = oac_data_for_clustering,
      k = 8, 
      m = 1.5, 
      alpha = seq(0,3,0.05),
      nblistw = oac_geo_mdt_listw, 
      standardize = FALSE,
      tol = 0.0001, 
      verbose = TRUE, 
      seed = clustering_seed
    )
  
  
  # https://cran.r-project.org/web/packages/geocmeans/vignettes/introduction.html#spatial-c-means-and-generalized-c-means
  # "Let us just stress that 
  # a larger silhouette index indicates a better classification, 
  # and a smaller Xie and Beni index indicates a better classification."
  
  p_oac60pca_sfcm_mdt_dfindices_spc <-
    ggplot(oac60pca_sfcm_mdt_dfindices) +
    geom_smooth(aes(x=alpha,y=spConsistency), color = "black") +
    geom_point(aes(x=alpha,y=spConsistency), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_mdt_dfindices_spc--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_mdt_dfindices_spc
  )
  
  p_oac60pca_sfcm_mdt_dfindices_ein <-
    ggplot(oac60pca_sfcm_mdt_dfindices) +
    geom_smooth(aes(x=alpha,y=Explained.inertia), color = "black") +
    geom_point(aes(x=alpha,y=Explained.inertia), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_mdt_dfindices_ein--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_mdt_dfindices_ein
  )
  
  p_oac60pca_sfcm_mdt_dfindices_sil <-
    ggplot(oac60pca_sfcm_mdt_dfindices) +
    geom_smooth(aes(x=alpha,y=Silhouette.index), color = "black") +
    geom_point(aes(x=alpha,y=Silhouette.index), color = "red")
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_mdt_dfindices_sil--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_mdt_dfindices_sil
  )
  
  p_oac60pca_sfcm_mdt_dfindices_xbi <-
    ggplot(oac60pca_sfcm_mdt_dfindices) +
    geom_smooth(aes(x=alpha,y=XieBeni.index), color = "black") +
    geom_point(aes(x=alpha,y=XieBeni.index), color = "red") +
    scale_y_log10()
  ggsave(
    paste0(
      "Data/Output/GeoCMeans/p_oac60pca_sfcm_mdt_dfindices_xbi--",
      as.character(clustering_seed),
      ".png"
    ),
    p_oac60pca_sfcm_mdt_dfindices_xbi
  )

}


### Run -------------------------------------------------------------------

oac60pca_sfcm_mdt <- 
  SFCMeans(
    oac_data_for_clustering, 
    oac_geo_mdt_listw, 
    k = 8, 
    m = process_PCA60_m_mdt, 
    alpha = process_PCA60_alpha_mdt,
    standardize = FALSE,
    tol = 0.0001, 
    seed = clustering_seed
  )


if (process_PCA60_plot_map) {
  
  oac60pca_sfcm_mdt_maps <- 
    mapClusters(geodata = oac_geo, object = oac60pca_sfcm_mdt$Belongings,undecided = 0)
  oac60pca_sfcm_mdt_maps$ClusterPlot

}

## SFCM results -----------------------------------------------------------

oac60pca_sfcm_queen_results <-
  oac_geo@data %>% 
  bind_cols(
    oac60pca_sfcm_queen$Belongings %>% 
      as.data.frame()
  ) %>% 
  select(OA11CD,V1:V8) %>% 
  rename(
    belongs_queen_c0 = V1,
    belongs_queen_c1 = V2,
    belongs_queen_c2 = V3,
    belongs_queen_c3 = V4,
    belongs_queen_c4 = V5,
    belongs_queen_c5 = V6,
    belongs_queen_c6 = V7,
    belongs_queen_c7 = V8
  )

oac60pca_sfcm_queen_clusters <-
  oac60pca_sfcm_queen_results  %>% 
  pivot_longer(
    cols = -OA11CD,
    names_to = "cluster",
    values_to = "belonging"
  ) %>% 
  group_by(OA11CD) %>% 
  slice_max(order_by = belonging, n = 1) %>%
  mutate(Queens_SFCM_60pca = as.numeric(str_sub(cluster, start = 16))) %>% 
  ungroup()

oac60pca_sfcm_queen_output <-
  oac60pca_sfcm_queen_results %>%
  left_join(
    oac60pca_sfcm_queen_clusters %>% select(OA11CD, Queens_SFCM_60pca)
  )

oac60pca_sfcm_knn8_results <-
  oac_geo@data %>% 
  bind_cols(
    oac60pca_sfcm_knn8$Belongings %>% 
      as.data.frame()
  ) %>% 
  select(OA11CD,V1:V8) %>% 
  rename(
    belongs_knn8_c0 = V1,
    belongs_knn8_c1 = V2,
    belongs_knn8_c2 = V3,
    belongs_knn8_c3 = V4,
    belongs_knn8_c4 = V5,
    belongs_knn8_c5 = V6,
    belongs_knn8_c6 = V7,
    belongs_knn8_c7 = V8
  )

oac60pca_sfcm_knn8_clusters <-
  oac60pca_sfcm_knn8_results  %>% 
  pivot_longer(
    cols = -OA11CD,
    names_to = "cluster",
    values_to = "belonging"
  ) %>% 
  group_by(OA11CD) %>% 
  slice_max(order_by = belonging, n = 1) %>%
  mutate(KNN8_SFCM_60pca = as.numeric(str_sub(cluster, start = 15))) %>% 
  ungroup()

oac60pca_sfcm_knn8_output <-
  oac60pca_sfcm_knn8_results %>%
  left_join(
    oac60pca_sfcm_knn8_clusters %>% select(OA11CD, KNN8_SFCM_60pca)
  )


oac60pca_sfcm_mdt_results <-
  oac_geo@data %>% 
  bind_cols(
    oac60pca_sfcm_mdt$Belongings %>% 
      as.data.frame()
  ) %>% 
  select(OA11CD,V1:V8) %>% 
  rename(
    belongs_mdt_c0 = V1,
    belongs_mdt_c1 = V2,
    belongs_mdt_c2 = V3,
    belongs_mdt_c3 = V4,
    belongs_mdt_c4 = V5,
    belongs_mdt_c5 = V6,
    belongs_mdt_c6 = V7,
    belongs_mdt_c7 = V8
  )

oac60pca_sfcm_mdt_clusters <-
  oac60pca_sfcm_mdt_results  %>% 
  pivot_longer(
    cols = -OA11CD,
    names_to = "cluster",
    values_to = "belonging"
  ) %>% 
  group_by(OA11CD) %>% 
  slice_max(order_by = belonging, n = 1) %>%
  mutate(MDT_SFCM_60pca = as.numeric(str_sub(cluster, start = 14))) %>% 
  ungroup()

oac60pca_sfcm_mdt_output <-
  oac60pca_sfcm_mdt_results %>%
  left_join(
    oac60pca_sfcm_mdt_clusters %>% select(OA11CD, MDT_SFCM_60pca)
  )

oac60pca_sfcm_queen_output %>% 
  left_join(oac60pca_sfcm_knn8_output) %>% 
  left_join(oac60pca_sfcm_mdt_output) %>% 
  rename_with(
    ~ paste0(
      .x, 
      "_m",
      process_PCA60_m_queen %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_a",
      process_PCA60_alpha_queen %>% format(nsmall = 2) %>% str_remove_all("\\.")
    ),
    matches("Queens_SFCM_60pca")
  ) %>% 
  rename_with(
    ~ paste0(
      .x, 
      "_m",
      process_PCA60_m_knn8 %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_a",
      process_PCA60_alpha_knn8 %>% format(nsmall = 2) %>% str_remove_all("\\.")
    ),
    matches("KNN8_SFCM_60pca")
  ) %>% 
  rename_with(
    ~ paste0(
      .x, 
      "_m",
      process_PCA60_m_mdt %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_a",
      process_PCA60_alpha_mdt %>% format(nsmall = 2) %>% str_remove_all("\\.")
    ),
    matches("MDT_SFCM_60pca")
  ) %>% 
  write_csv(
    paste0(
      "Data/Output/GeoCMeans/SFCM_60pca_seed-",
      clustering_seed,
      "_m-alpha",
      "_queen-", 
      process_PCA60_m_queen %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "-",
      process_PCA60_alpha_queen %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_knn8-", 
      process_PCA60_m_knn8 %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "-",
      process_PCA60_alpha_knn8 %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_mdt-", 
      process_PCA60_m_mdt %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "-",
      process_PCA60_alpha_mdt %>% format(nsmall = 2) %>% str_remove_all("\\."),
      "_output.csv")
  )


