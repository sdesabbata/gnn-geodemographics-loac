# To recreate 
#   - ons-oa-geo-london_downloaded.geojson
#   - ons-oa-geo-london_valid.geojson
#   - ons-oa-geo-london_valid_BNG.geojson
# this script downloads 
#
# Output Areas (December 2011) Boundaries EW BGC (V2)
# https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2011-boundaries-ew-bgc-v2/about
#
# Contains both Ordnance Survey and ONS Intellectual Property Rights.
#
# Published Date: July 24, 2020
# Source: Office for National Statistics licensed under the Open Government Licence v.3.0
# https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
# Contains OS data © Crown copyright and database right 2020
#
#
# >>> REQUIRES MANUAL DOWNLOAD OF: <<<
#
#
# save in the same folder
#
#
# Output Area (2011) to LSOA to MSOA to LAD (Dec 2017) Lookup with Area Classification in GB
# https://geoportal.statistics.gov.uk/datasets/ons::output-area-2011-to-lsoa-to-msoa-to-lad-dec-2017-lookup-with-area-classification-in-gb/about
# 
# Contains both Ordnance Survey and ONS Intellectual Property Rights.
#
# Published Date: August 3, 2018
# Source: Office for National Statistics licensed under the Open Government Licence v.3.0
# https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
# Contains OS data © Crown copyright and database right 2018
#
#
# Author: Stef De Sabbata
# Date: 09 August 2024


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(magrittr)
library(sf)
library(jsonlite)



# Data download -----------------------------------------------------------

cat("Retrieving data\n")

lookup_London_OA_2011 <-
  read_csv("Output_Area_to_LSOA_to_MSOA_to_Local_Authority_District_(December_2017)_Lookup_with_Area_Classifications_in_Great_Britain.csv") %>% 
  filter(RGN11NM == "London") %>% 
  select(OA11CD, LSOA11CD, LSOA11NM, LAD17CD, LAD17NM, RGN11CD, RGN11NM)

lookup_queries <- 
  lookup_London_OA_2011 %>% 
    nrow() %>% 
    divide_by(50) %>% 
    floor()

for (i in 0:lookup_queries) {
  
  cat(paste("\nQuery", i+1, "of", lookup_queries+1,"\n"))
  
  tmp_OAs <-
    lookup_London_OA_2011 %>% 
    slice_tail(
      n = 
        lookup_London_OA_2011 %>% 
        nrow() %>% 
        subtract(i * 50)
    ) %>% 
    slice_head(n = 50) %>% 
    pull(OA11CD) %>% 
    paste0(
      "OA11CD%20%3D%20'",
      .,
      "'%20OR%20",
      collapse = ""
    ) %>% 
    str_sub(end = -9) %>% 
    paste0(
      "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Output_Areas_Dec_2011_Boundaries_EW_BGC_2022/FeatureServer/0/query?where=",
      .,
      "&outFields=*&outSR=4326&f=json"
    ) %>% 
    st_read()
  
  if (i == 0){
    geom_London_OAs <- tmp_OAs
  } else {
    geom_London_OAs %<>%
      bind_rows(tmp_OAs)
  }
  
}



# Check -------------------------------------------------------------------

cat("Checking: resulting table should be empty...\n")

lookup_London_OA_2011 %>% 
  select(OA11CD) %>% 
  anti_join(
    geom_London_OAs %>% 
      st_drop_geometry() %>% 
      select(OA11CD)
  )



# Write -------------------------------------------------------------------

cat("Write downloaded file\n")

geom_London_OAs %>% 
  st_write("ons-oa-geo-london_downloaded.geojson")



# Validate ----------------------------------------------------------------

if(all(st_is_valid(geom_London_OAs))){
  
  cat("Geometries are valid, write to file\n")
  
  geom_London_OAs %>% 
    st_write("ons-oa-geo-london_valid.geojson")
  
}else{
  
  cat("Geometries are not valid, fix and write to file\n")
  
  geom_London_OAs %>% 
    st_make_valid() %>% 
    st_write("ons-oa-geo-london_valid.geojson")
  
}



# To BNG ------------------------------------------------------------------

st_read("ons-oa-geo-london_valid.geojson") %>% 
  st_transform(27700) %>% 
  st_write("ons-oa-geo-london_valid_BNG.geojson")
