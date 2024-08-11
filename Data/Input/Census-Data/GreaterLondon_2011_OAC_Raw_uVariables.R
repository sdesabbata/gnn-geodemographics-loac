# To recreate 
#   - GreaterLondon_2011_OAC_Raw_uVariables.csv
#
#
# >>> REQUIRES PRELIMINARY EXECUTION OF: <<<
#
# Data/Input/London-Map-OAs/ons-oa-geo-london_valid.R
#
#
# >>> REQUIRES MANUAL DOWNLOAD OF: <<<
#
#
# To reproduce the results, copy in this folder the 167 variables from the 
# 2011 UK Census data originally taken into account for the creation of the 
# 2011 Output Area Classification developed by Chris Gale and availalbe 
# from Gale's repository. That dataset is released under the Open 
# Government Licence v3.
#
# http://geogale.github.io/2011OAC
#
# Copy the contents of the unzipped `2011 OAC 167 Variables/` folder
# in the folder `Data/Input/Census-Data/`
#
#
# Author: Stef De Sabbata
# Date: 09 August 2024


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(sf)

read_csv("Data/Input/Census-Data/2011_OAC_Raw_uVariables.csv") %>% 
  semi_join(
    st_read("Data/Input/London-Map-OAs/ons-oa-geo-london_valid.geojson") %>% 
      st_drop_geometry(),
    by = c("OA" = "OA11CD")
  ) %>% 
  rename(OA11CD = OA) %>% 
  write_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables.csv")

