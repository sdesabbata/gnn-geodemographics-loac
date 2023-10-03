# A graph neural network framework for spatial geodemographic classification

This repository includes the code and data that support the findings of a forthcoming paper ([De Sabbata and Liu, 2023](https://doi.org/10.1080/13658816.2023.2254382)). The documentation is currently a draft and it will be updated soon.


## Abstract

Geodemographic classifications are exceptional tools for geographic analysis, business and policy-making, providing an overview of the socio-demographic structure of a region by creating an unsupervised, bottom-up classification of its areas based on a large set of variables. Classic approaches can require time-consuming preprocessing of input variables and are frequently a-spatial processes. In this study, we present a groundbreaking, systematic investigation of the use of graph neural networks for spatial geodemographic classification. Using Greater London as a case study, we compare a range of graph autoencoder designs with the official London Output Area Classification and baseline classifications developed using spatial fuzzy c-means. The results show that our framework based on a Node Attributes-focused Graph AutoEncoder (NAGAE) can perform similarly to classic approaches on class homogeneity metrics while providing higher spatial clustering. We conclude by discussing the current limitations of the proposed framework and its potential to develop into a new paradigm for creating a range of geodemographic classifications, from simple, local ones to complex classifications able to incorporate a range of spatial relationships into the process.


## License

The dataset containing the results is released under [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) and it includes the baselines and the London Output Area Classification by Singleton and Longley (2015).

The code is relased under MIT License.


## Data and codes availability statement

The data that support the findings of this study are available under Open Government Licence v3 at [doi.org/10.25392/leicester.data.20503230](https://doi.org/10.25392/leicester.data.20503230). The code is available under MIT License at [doi.org/10.25392/leicester.data.20503311](https://doi.org/10.25392/leicester.data.20503311). Additional detailed reports on the evaluation are available under Creative Commons Attribution 4.0 International (CC BY 4.0) licence at [doi.org/10.25392/leicester.data.20503374](https://doi.org/10.25392/leicester.data.20503374). 


## Acknowledgments

The authors would like to thank Prof May Yuan, Prof Harvey Miller and the anonymous reviewers for their valuable comments. This research used the ALICE High Performance Computing Facility at the University of Leicester.


## References

De Sabbata S, Liu P (2023). A graph neural network framework for spatial geodemographic classification. International Journal of Geographical Information Science, DOI: 10.1080/13658816.2023.2254382

Singleton A D, Longley P A (2015). The Internal Structure of Greater London: A Comparison of National and Regional Geodemographic Models. Geo: Geography and Environment. Available from: dx.doi.org/10.1002/geo2.7

