# Data

This directory contains the patient mutation and metadata on which the analyses were performed.

## Content

- [_G13_LUAD_12.csv_](G13_LUAD_12.csv) is the binary mutation data for 12 genes of 3662 lung adenocarcinoma patients.
- [_G13_COAD_12.csv_](G13_COAD_12.csv) is the binary mutation data for 12 genes of 2269 colon adenocarcinoma patients.
- [_msk_chord_2024_clinical_data.tsv_](msk_chord_2024_clinical_data.tsv) contains clinical patient data, including survival status.
- [_data_timeline_treatment.txt_](data_timeline_treatment.txt)

## Preparation

The results we show in this article are based on MHNs that were trained on the same datasets as in [Schill et al. (2024)](https://doi.org/10.1089/cmb.2024.0666):
These datasets were collected by the Memorial Sloan Kettering Cancer Center ([Nguyen et al., 2022](https://doi.org/10.1016/j.cell.2022.01.003)) and retrieved through AACR GENIE ([The AACR Project GENIE Consortium et al., 2017](https://doi.org/10.1158/2159-8290.CD-17-0151)).
We selected the 12 most commonly affected genes and followed Schill et al. (2024)'s preprocessing steps.