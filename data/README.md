# Data Provenance & Dictionary Log

## Dataset Identification
- **Dataset Name**: Online Shoppers Purchasing Intention Dataset
- **Author(s)**: C. O. Sakar, S. O. Polat, K. Katircioglu, and Y. Kasa
- **Publication Reference**: Sakar, C. O., Polat, S. O., Katircioglu, K., & Kasa, Y. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks*. Knowledge-Based Systems, 166, 156-168.
- **DOI**: [10.1016/j.knosys.2018.10.036](https://doi.org/10.1016/j.knosys.2018.10.036)
- **Authoritative Source**: UCI Machine Learning Repository (Dataset ID: 468)
- **Source URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv`
- **UCI Dataset Landing Page**: `https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Raw File Provenance & Integrity
- **Original Filename**: `online_shoppers_intention.csv`
- **Storage Location**: `c:\Users\rishi\OneDrive\Desktop\AI-Based Sales Conversion Probability Prediction\data\raw\online_shoppers_intention.csv`
- **File Size**: 1,072,063 bytes (~1.02 MB)
- **SHA-256 Checksum**: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`
- **Download Date**: September 11, 2026

## Dataset Overview
- **Total Rows (Sessions)**: 12,330
- **Total Columns**: 18 (17 predictive features + 1 target variable)
- **Target Variable**: `Revenue` (Boolean)
- **Target Distribution**:
  - `False` (Did not convert): 10,422 (84.53%)
  - `True` (Converted sale): 1,908 (15.47%)

## Column Schema Dictionary

| Column Name | Data Type | Feature Type | Description |
| :--- | :--- | :--- | :--- |
| `Administrative` | Integer | Numerical | Number of administrative pages visited by user |
| `Administrative_Duration` | Float | Numerical | Total time spent (seconds) on administrative pages |
| `Informational` | Integer | Numerical | Number of informational pages visited by user |
| `Informational_Duration` | Float | Numerical | Total time spent (seconds) on informational pages |
| `ProductRelated` | Integer | Numerical | Number of product-related pages visited by user |
| `ProductRelated_Duration` | Float | Numerical | Total time spent (seconds) on product-related pages |
| `BounceRates` | Float | Numerical | Percentage of visitors entering site who exit without triggering further requests |
| `ExitRates` | Float | Numerical | Percentage of pageviews for page that were the last in the session |
| `PageValues` | Float | Numerical | Average value of page visited before completing a transaction |
| `SpecialDay` | Float | Numerical | Closeness of site visiting time to a specific holiday/special day (0 to 1) |
| `Month` | Object | Categorical | Month of the visit (Jan - Dec) |
| `OperatingSystems` | Integer | Categorical | Operating system ID used by visitor |
| `Browser` | Integer | Categorical | Browser ID used by visitor |
| `Region` | Integer | Categorical | Geographic region ID of visitor |
| `TrafficType` | Integer | Categorical | Traffic source type ID |
| `VisitorType` | Object | Categorical | Visitor category (`Returning_Visitor`, `New_Visitor`, `Other`) |
| `Weekend` | Boolean | Categorical | Whether session occurred on a weekend |
| `Revenue` | Boolean | Target | **Target Variable**: `True` if session ended in a sale, `False` otherwise |

## Distinctness & Teammate Risk Verification
- **Peer Verification Note**: Due to absolute lack of communication channels with course teammates, direct cross-verification of datasets could not be executed.
- **Risk Mitigation**: Selected a peer-reviewed UCI repository dataset with documented academic provenance (DOI: 10.1016/j.knosys.2018.10.036) that is distinct from standard tutorial datasets (e.g., X Education Lead Scoring).
