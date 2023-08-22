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
      "GreaterLondon_2011_OAC_Raw_uVariables--zscores-with-colnames--",
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



# Information for testing -------------------------------------------------

# The u-variables with no normalising total
# should be:
#
# u005	Area	  Area (Hectares)	  Area hectares
# u006	Ratio	  Density	          Density (number of persons per hectare)
# u020	Years	  Person	          Mean age
# u021	Years	  Person	          Median age
# u104	Ratio	  SIR	              Day-to-day activities limited a lot or a little Standardised Illness Ratio
test__uvars_not_to_normalise <-
  c("u005", "u006", "u020", "u021", "u104")

# Epsilon
test__num_eps <- 1.0e-12




# Load variable information -----------------------------------------------

# Load lookup table
lookup <- 
  read_csv("Data/Input/Census-Data/2011_OAC_Raw_uVariables_Lookup.csv")



## Load totals info -------------------------------------------------------

# Select totals with statistical unit
norm_lookup_tot <-
  lookup %>% 
  select(VariableCode, VariableMeasurementUnit, VariableStatisticalUnit) %>% 
  filter(!(str_starts(VariableCode, "u"))) %>% 
  rename(
    tot_code = VariableCode,
    var_measurement_unit = VariableMeasurementUnit,
    var_statistical_unit = VariableStatisticalUnit
  )

### Unit test --------------------------------------------------------------

# Number of totals columns should be 10
expect(
  nrow(norm_lookup_tot) == 10,
  sprintf(
    "%s has length %i, not length %i.", 
    "norm_lookup_tot", 
    nrow(norm_lookup_tot), 
    10
  )
)
log_print("Unit test 001: correct number of totals columns")



## Load u-variables info --------------------------------------------------

# Select u-variables with statistical unit
norm_lookup_u <-
  lookup %>% 
  select(VariableCode, VariableDescription, VariableMeasurementUnit, VariableStatisticalUnit) %>% 
  filter(str_starts(VariableCode, "u")) %>% 
  rename(
    var_code = VariableCode,
    var_desc = VariableDescription,
    var_measurement_unit = VariableMeasurementUnit,
    var_statistical_unit = VariableStatisticalUnit
  )

### Unit test --------------------------------------------------------------

# Number of u-variables should be 167
expect(
  nrow(norm_lookup_u) == 167,
  sprintf(
    "%s has length %i, not length %i.", 
    "norm_lookup_u", 
    nrow(norm_lookup_u), 
    167
  )
)
log_print("Unit test 002: correct number of u-variables columns")



## Join u-vars with totals ------------------------------------------------

# Join by statistical unit
norm_table <-
  norm_lookup_u %>% 
  left_join(norm_lookup_tot) 

### Unit test --------------------------------------------------------------

# Number of rows in norm table should be 167
# as the number of variables
expect(
  nrow(norm_table) == 167,
  sprintf(
    "%s has length %i, not length %i.", 
    "norm_table", 
    nrow(norm_table), 
    167
  )
)
log_print("Unit test 003: correct number of u-variables rows")

# Select u-variables which have NA as total
test__na_norm_table <-
  norm_table %>% 
  filter(is.na(tot_code)) %>% 
  arrange(var_code) %>% 
  pull(var_code)

# Check u-variables not to be normalise have NA total
expect(
  all(test__na_norm_table == test__uvars_not_to_normalise),
  sprintf(
    "%s has an incorrect list of variables not to be normalised:\n%s", 
    "norm_table", 
    test__na_norm_table
  )
)
log_print("Unit test 004: correct list of variables not to be normalised")



## Save normalisation table -----------------------------------------------

norm_table %>% 
  write_csv("Data/Input/Census-Data/2011_OAC_Raw_uVariables_Lookup--normalisation.csv")

# Log normalisation table
norm_table %>% 
  kable() %>% 
  put()

# Remove variable description for the rest of the script
norm_table <-
  norm_table %>% 
  select(-var_desc)



# Normalisation -----------------------------------------------------------



## Load data --------------------------------------------------------------

# Load census data
census_data <- 
  read_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables.csv")

# Log skim census_data
census_data %>% 
  skim() %>% 
  put()

# Select totals and transform to long
census_data_tot_long <-
  census_data %>% 
  select(OA11CD, Total_Population:Total_Population_3_and_over) %>% 
  pivot_longer(
    cols = Total_Population:Total_Population_3_and_over,
    names_to = "tot_code",
    values_to = "tot_value"
  )

# Select u-variables and transform to long
census_data_var_long <-
  census_data %>% 
  select(OA11CD, u001:u167) %>% 
  pivot_longer(
    cols = u001:u167,
    names_to = "var_code",
    values_to = "var_value"
  ) 

### Unit test --------------------------------------------------------------

num_of_oas <- 
  census_data %>% 
  nrow()

# Number of rows in census_data_tot_long should be 10 per OA
expect(
  nrow(census_data_tot_long) == (10 * num_of_oas),
  sprintf(
    "%s has length %i, not length %i.", 
    "census_data_tot_long", 
    nrow(census_data_tot_long), 
    (10 * num_of_oas)
  )
)
log_print("Unit test 005: correct number of totals columns")


# Number of rows in census_data_var_long should be 167 per OA
expect(
  nrow(census_data_var_long) == (167 * num_of_oas),
  sprintf(
    "%s has length %i, not length %i.", 
    "census_data_var_long", 
    nrow(census_data_var_long), 
    (167 * num_of_oas)
  )
)
log_print("Unit test 006: correct number of variable columns")



## Normalise data ---------------------------------------------------------

# Join through normalisation table
# and calculate normalised value
census_data_norm_long <-
  census_data_var_long %>% 
  left_join(
    norm_table
  ) %>% 
  left_join(
    census_data_tot_long
  ) %>% 
  mutate(
    norm_value = 
      if_else(
        is.na(tot_code),
        var_value,
        (var_value / tot_value) * 100
      )
  )

# Log a sample
census_data_norm_long %>% 
  slice_sample(n = 167) %>% 
  kable() %>% 
  put()



### Unit test --------------------------------------------------------------

# Check that no normalised value is above 100

# Filter data were normalised value is above 100
test__normalised_values_above100 <-
  census_data_norm_long %>% 
  filter(!(var_code %in% test__uvars_not_to_normalise)) %>% 
  filter(norm_value > 100) 

test__normalised_values_above100 %>% 
  write_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--incongruences.csv")

# There seem to be some issues in the original data
# where values are higher than totals for:
#
# u084 Household spaces with at least one usual resident: 2314 cases
# u085 Household spaces with no usual residents: 1 case
#
# As those are part of the original data as available 
# from Chris Gale's repo (https://github.com/geogale/2011OAC)
# they won't be considered as errors here

test__normalised_values_above100 <-
  test__normalised_values_above100 %>% 
  filter(!(var_code %in% c("u084", "u085")))

expect(
  nrow(test__normalised_values_above100) == 0,
  sprintf(
    "There are %s normalised values above 100.",
    nrow(test__normalised_values_above100)
  ),
  info = kable(test__normalised_values_above100)
)
log_print("Unit test 007: correct normalised values (check values not above 100, except where expected)")



## Transform to wide table ------------------------------------------------

census_data_norm <-
  census_data_norm_long %>% 
  pivot_wider(
    id_cols = OA11CD,
    names_from = var_code,
    values_from = norm_value
  )

### Unit test -------------------------------------------------------------

# Number of rows in normalised data table should be 
# the same as the original table
expect(
  nrow(census_data_norm) == nrow(census_data),
  sprintf(
    "%s has %i rows, not %i rows.", 
    "census_data_norm", 
    nrow(census_data_norm), 
    nrow(census_data)
  )
)
log_print("Unit test 008: correct number of rows in wide normalised data table")

# Number of columns in normalised data table should be 
# 167 + 1 (the OA code column)
expect(
  ncol(census_data_norm) == (167 + 1),
  sprintf(
    "%s has %i columns, not %i columns.", 
    "census_data_norm", 
    ncol(census_data_norm), 
    (167 + 1)
  )
)
log_print("Unit test 009: correct number of columns. in wide normalised data table")

# Log skim census_data_norm
census_data_norm %>% 
  skim() %>% 
  put()


# Checks for infinite, NA and NaN

test__norm_infinite <-
  census_data_norm %>% 
  filter_all(any_vars(is.infinite(.)))

expect(
  nrow(test__norm_infinite) == 0,
  sprintf(
    "%s has %i infinite values, should be zero.", 
    "census_data_norm", 
    nrow(test__norm_infinite)
  )
)
log_print("Unit test 010: no infinite values")

test__norm_na <-
  census_data_norm %>% 
  filter_all(any_vars(is.na(.)))

expect(
  nrow(test__norm_na) == 0,
  sprintf(
    "%s has %i NA values, should be zero.", 
    "census_data_norm", 
    nrow(test__norm_na)
  )
)
log_print("Unit test 011: no NA values")

test__norm_nan <-
  census_data_norm %>% 
  filter_all(any_vars(is.nan(.)))

expect(
  nrow(test__norm_nan) == 0,
  sprintf(
    "%s has %i NaN values, should be zero.", 
    "census_data_norm", 
    nrow(test__norm_nan)
  )
)
log_print("Unit test 012: no NaN values")


# Z-scores ----------------------------------------------------------------



## Calculate z-scores -----------------------------------------------------

census_data_zscores <-
  census_data_norm %>% 
  mutate(
    across(
      u001:u167,
      # Create zscores using scale
      ~ as.numeric(scale(.))
    )
  )

### Unit test -------------------------------------------------------------

# Calculate mean
test__zscores_means <-
  census_data_zscores %>% 
  select(u001:u167) %>% 
  summarise(
    across(
      u001:u167,
      ~ mean(.)
    )
  ) %>% 
  pivot_longer(
    cols = everything(),
    names_to = "var_code",
    values_to = "var_mean"
  ) %>% 
  pull(var_mean)

# Check means are 0
expect(
  all(abs(test__zscores_means) < test__num_eps),
  sprintf(
    "%i variables have non-zero mean.", 
    (abs(test__zscores_means) >= test__num_eps) %>% 
      which() %>% 
      length()
  )
)
log_print("Unit test 013: z-scores means are zero")

# Calculate standard deviations
test__zscores_sds <-
  census_data_zscores %>% 
  select(u001:u167) %>% 
  summarise(
    across(
      u001:u167,
      ~ sd(.)
    )
  ) %>% 
  pivot_longer(
    cols = everything(),
    names_to = "var_code",
    values_to = "var_sd"
  ) %>% 
  pull(var_sd)

# Check standard deviations are 1
expect(
  all(abs(test__zscores_sds) < (1 + test__num_eps)),
  sprintf(
    "%i variables have non-zero mean.", 
    (abs(test__zscores_sds) >= (1 + test__num_eps)) %>% 
      which() %>% 
      length()
  )
)
log_print("Unit test 014: z-scores standard deviations are one")

# Log skim census_data_zscores
census_data_zscores %>% 
  skim() %>% 
  put()


# Checks for infinite, NA and NaN

test__zscores_infinite <-
  census_data_zscores %>% 
  filter_all(any_vars(is.infinite(.)))

expect(
  nrow(test__zscores_infinite) == 0,
  sprintf(
    "%s has %i infinite values, should be zero.", 
    "census_data_zscores", 
    nrow(test__zscores_infinite)
  )
)
log_print("Unit test 015: no infinite values")

test__zscores_na <-
  census_data_zscores %>% 
  filter_all(any_vars(is.na(.)))

expect(
  nrow(test__zscores_na) == 0,
  sprintf(
    "%s has %i NA values, should be zero.", 
    "census_data_zscores", 
    nrow(test__zscores_na)
  )
)
log_print("Unit test 016: no NA values")

test__zscores_nan <-
  census_data_zscores %>% 
  filter_all(any_vars(is.nan(.)))

expect(
  nrow(test__zscores_nan) == 0,
  sprintf(
    "%s has %i NaN values, should be zero.", 
    "census_data_zscores", 
    nrow(test__zscores_nan)
  )
)
log_print("Unit test 017: no NaN values")



## Save z-scores ----------------------------------------------------------

census_data_zscores %>% 
  write_csv("Data/Input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--zscores-with-colnames.csv")



# Clean-up and log --------------------------------------------------------

log_close()
writeLines(readLines(logr_object, encoding = "UTF-8"))
