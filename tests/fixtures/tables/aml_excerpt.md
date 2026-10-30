and PNA models on the _HI-Medium_ and _-Large_ datasets enables higher F1 scores than those of
XGBoost models trained on the respective LI datasets (see Table 2). Furthermore, the classification

8

Table 2: Minority class F1 scores (%). HI indicates a higher illicit ratio. LI indicates a lower ratio.

Model HI - _Small_ LI - _Small_ HI - _Medium_ LI - _Medium_ HI - _Large_ LI - _Large_

Table 3: Minority class F1 scores (%) of models trained on HI datasets and evaluated on LI datasets.

<u>Model</u> <u>LI</u> <u>-</u> _<u>Small</u>_ <u>LI</u> <u>-</u> _<u>Medium</u>_ <u>LI</u> <u>-</u> _<u>Large</u>_

100%

80%


size of 64. The total GPU time, including initial experiments and hyperparameter optimization, is
estimated to have been around 1000 GPU hours.

Table 12: Total training times (TTT) and inference performance in Transaction per Second (TPS) for
all GNN baselines on the AML Small and Medium datasets using an Nvidia Tesla V100 GPU.

HI        - _Small_ LI        - _Small_ HI        - _Medium_ LI        - _Medium_
Model
<u>TTT (s)</u> <u>TPS</u> <u>TTT (s)</u> <u>TPS</u> <u>TTT (s)</u> <u>TPS</u> <u>TTT (s)</u> <u>TPS</u>

GIN 22703 17210 29655 15784 85652 7101 85994 7316
GIN+EU 26046 11844 36625 11640 85753 4981 85887 5097
<u>PNA</u> <u>27745</u> <u>16557</u> <u>39648</u> <u>14994</u> <u>85654</u> <u>6725</u> <u>85827</u> <u>6819</u>

In addition to the minority class F1 scores in Table 2, we include more fine-grained results detailing the
precision and recall rates of the GNN-based models. Precision evaluates the accuracy of laundering
predictions, recall measures the model’s ability to identify all laundering instances, while the F1 score
is the harmonic mean of precision and recall. Table 13 shows the laundering recall rates and Table
14 shows the corresponding precision scores. In Figure 9, we give example precision-recall curves,
which visually capture the trade-off between precision and recall across different decision thresholds.
The examples are taken from the best-performing seed from the best-performing model (PNA) on all
small and medium AML datasets.

Table 13: Minority class recall rate (%) for the GNN-based models. HI indicates a higher illicit ratio.
LI indicates a lower illicit ratio.

Model HI      - _Small_ LI      - _Small_ HI      - _Medium_ LI      - _Medium_

Table 14: Minority class precision (%) for the GNN-based models. HI indicates a higher illicit ratio.
LI indicates a lower illicit ratio.

Model HI      - _Small_ LI      - _Small_ HI      - _Medium_ LI      - _Medium_

F Additional GBT Experiments
