library(tidyverse)
library(rmarkdown)

rm(list = ls())


# Environment -------------------------------------------------------------

classifications_file_folder <- "Data/Output/Geodemographic-Clusters/"
classifications_file_name <- "GreaterLondon_2011OA_classifications_LOAC_SFCM_vGNN_NAGAE.csv"
#output_file_prefix <- "GeodemoComparisonReports/compare-geodemo-LOAC__"
output_file_prefix <- "GeodemoComparisonReports/compare-classification-with-LOAC__"

geodemo_approaches <-
  read_csv(
    paste0(
      classifications_file_folder,
      classifications_file_name
    )
  ) %>%
  select(-oa_code, -LOAC_supgrp_cd) %>%
  colnames()


# Generate report ---------------------------------------------------------

# Function generating the reports
generate_report_for <- 
  function(a_output_filename, a_dlclass_name){
    rmarkdown::render(
      "Analysis/compare-geodemo-classifications-report.Rmd", 
      params = list(
        output_filename = a_output_filename,
        dlclass_name = a_dlclass_name
        #,
        #dlclass_mapping = a_dlclass_mapping
      ),
      output_file = paste0(output_file_prefix, a_dlclass_name, ".pdf")
    )  
  }

for (an_approach in geodemo_approaches) {
  rm(list = ls()[!"generate_report_for" %in% ls()])
  generate_report_for(
    #"../output/output-for-comparison-2021-09-10-final.csv",
    paste0(
      "../",
      classifications_file_folder,
      classifications_file_name
    ),
    an_approach
  )
  rm(list = ls()[!"generate_report_for" %in% ls()])
}