# Census data

To reproduce the results, copy in this folder the 167 variables from the 2011 UK Census data originally taken into account for the creation of the 2011 Output Area Classification developed by Chris Gale and [availalbe from Gale's repository](http://geogale.github.io/2011OAC). That dataset is released under the [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).


## Data preparation 

Run the scripts in the following order, following the instructions in the header of the script

1. GreaterLondon_2011_OAC_Raw_uVariables.R
2. 2011OAC-data-norm-and-zscores.R
3. 2011OAC-zscores-to-PCA60.R
