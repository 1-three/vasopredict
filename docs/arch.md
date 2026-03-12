```mermaid
flowchart TD

A[Input Layer<br>MIMIC-IV Data<br>Labs + Vitals + Treatments]

B[Temporal Features<br>5 Time Windows<br>3h,6h,12h,24h,48h<br>Mean Max Min Std Slope]

C[Demographics + Static<br>Age Gender<br>Comorbidities<br>Initial Values]

D[Q-Learning Grey Wolf Optimizer<br>Deep Q Network guides selection<br>Wolf pack explores combinations<br>Adaptive convergence]

E[Optimized Feature Set<br>25 features from 150]

F[Classification Layer<br>LightGBM<br>Ensemble Learning]

G[Sepsis Prediction<br>Probability 0-100%<br>AUC 0.87<br>Sensitivity 91%]

H[Risk Stratification<br>Low Moderate High<br>6h window<br>Optimal threshold]

I[Explainability<br>SHAP values<br>Feature importance<br>Clinical reasoning]

A --> B
A --> C

B --> D
C --> D

D --> E

E --> F

F --> G
F --> H
F --> I
