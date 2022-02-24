# CostSensitiveLearning

This is the code for the paper on ["Predict-then-optimize or predict-and-optimize? An empirical evaluation of cost-sensitive learning strategies"](https://www.sciencedirect.com/science/article/pii/S0020025522001542).

An experiment is conducted by running the `experiments/overview.py` file where settings, data set, methodologies, and evaluators can be chosen.

## Data

Due to size limitations, not all data is provided in this repository. Data sets can be found online at the following links: 
- [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Kaggle IEEE Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- [UCI KDD98 Direct Mailing](http://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html)
- [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- [TV Subscription Churn](https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/datasets/data/churn_tv_subscriptions.csv.gz)
- [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#)
- [VUB Credit Scoring](https://github.com/vub-dl/data-csl-pdcs)

## Acknowledgments
The code for cslogit and csboost are Python versions of the [original cslogit by Sebastiaan Höppner et al.](https://github.com/SebastiaanHoppner/CostSensitiveLearning).

## Reference
If you use this software, please cite it as follows:
`
@article{vanderschueren2022predict, 
  title={Predict-then-optimize or predict-and-optimize? An empirical evaluation of cost-sensitive learning strategies},
  author={Vanderschueren, Toon, Verdonck, Tim, Baesens, Bart and Verbeke, Wouter},
  journal={Information Sciences},
  year={2022},
  publisher={Elsevier}
}
`

Contact the author at [toon.vanderschueren@gmail.com](mailto:toon.vanderschueren@gmail.com).
