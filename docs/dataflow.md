```mermaid

flowchart TD

A[MIMIC-IV<br>10,000 patients]

B[Data Split<br>60/20/20]

C[Time Windows<br>3h 6h 12h 24h 48h]

D[Train Val Test<br>Stratified split]

E[Feature Engineering<br>Aggregations]

F[Normalization<br>StandardScaler]

G[Deep Q Network<br>State features AUC diversity progress<br>Actions explore exploit]

H[Grey Wolf Optimizer<br>20 wolves searching<br>Alpha Beta Delta lead]

I[Selected Features<br>25 optimal features<br>83% reduction]

J[LightGBM Training]

K[Prediction + SHAP]

A --> D
B --> E
C --> F

D --> G
E --> G
F --> G

G -->|Q values| H
H -->|Rewards| G

H --> I

I --> J
I --> K
