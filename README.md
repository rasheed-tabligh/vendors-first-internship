# Late Government Payment Prediction

Predicting which government payments to vendors will be late, built for Vendors First during the iXperience Data Science internship.

## About the Project

Vendors First finances vendors who are waiting on government payments. Knowing in advance which payments are likely to run late helps the company decide which vendors to target and how to price risk. This project builds a model that flags high risk transactions and ranks vendors, counties, agencies, and programs by late payment risk.

## About the iXperience Program

iXperience is a program that pairs university students with real companies for a teaching phase followed by a client internship phase. This project was the client internship phase, working in a team for Vendors First.

## Dataset

- Source: USASpending.gov federal assistance prime transactions (public data), downloaded June 2026
- Coverage: 4 Southern California counties (San Diego, Orange, Riverside, Imperial)
- Modeling table: 500 ranked transactions from a pull of 1,488, stored in `data/Invoice_Late_Payment_Analysis_REAL.xlsx`
- A payment is labeled late when its processing lag is above the median of 23 days, giving 249 late and 251 on time records

## Approach

- Cleaning: trimmed and renamed columns, converted numeric fields, filled missing values with medians
- Features: county, awarding agency, and CFDA program (one hot encoded), plus obligation, outlay, and percent outlayed
- Data leakage fix: an early version included the lag based columns (risk index, risk percentile, risk tier) as features and scored a perfect 1.0 AUC, which gave the model the answer. The final version drops every column derived from processing lag
- Models: Random Forest (500 trees, balanced class weights) and Logistic Regression on the same 80/20 stratified split

## Results

| Model | Accuracy | ROC-AUC | Late recall | Late precision |
|---|---|---|---|---|
| Random Forest | 0.69 | 0.758 | 0.68 | 0.69 |
| Logistic Regression | 0.67 | 0.761 | 0.84 | 0.63 |

The two models perform about the same overall. Logistic regression catches more of the true late payments (recall 0.84) at the cost of more false alarms. Obligation and outlay amounts are the strongest predictors, and percent outlayed is the biggest positive driver in the logistic model.

Risk rankings from the model: San Diego is the highest risk county, the Department of Justice the highest risk agency, the Body Worn Camera Policy and Implementation program the highest risk program, and Orange County Housing Authority the top priority vendor.

Limitations: the sample is small (500 records), the late label is relative to the median rather than a contract deadline, and an AUC around 0.76 is useful for ranking but not for firm predictions.

## Repository Structure

```
vendors-first-internship/
├── README.md
├── notebooks/
│   ├── vendors_first.ipynb   # final notebook with narrative and outputs
│   └── vendors_first.html    # rendered version
├── src/
│   └── vendors_first.py      # raw Python source (Databricks export)
├── data/
│   └── Invoice_Late_Payment_Analysis_REAL.xlsx   # modeling input
├── reports/
│   └── Invoice_Late_Payment_Analysis.xlsx        # client deliverable
└── docs/                     # screenshots (coming)
```

## How to Run

Open `notebooks/vendors_first.ipynb` in Jupyter or import it into Databricks. It needs pandas, numpy, scikit-learn, matplotlib, and openpyxl. The notebook reads the Excel file from its working directory, so either run it from `data/` or update the path in the load cell to `../data/Invoice_Late_Payment_Analysis_REAL.xlsx`.

## My Role

This was a team project. I worked on data preparation, modeling, and evaluation, caught the data leakage issue that produced the perfect AUC, and built the vendor, county, agency, and program risk rankings.

## Team

- Rashid Mikidadi
- Tayveon Moore
- Rea Lila

## Acknowledgements

Thanks to Solomon at Vendors First for the project and feedback, to Michael Keutmann and the iXperience teaching team, and to my teammates for the collaboration.
