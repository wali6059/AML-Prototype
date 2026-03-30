# Background & Motivation

Tipping is a complex, noisy real-world behavior shaped by a confluence of geography, temporal dynamics, trip context, and underlying socioeconomic patterns. In the context of New York City's highly structured transportation ecosystem, tipping presents an ideal yet challenging Applied Machine Learning problem. The NYC TLC dataset provides immense volume and structural richness, yet the ultimate target, whether a rider tips and by how much, is highly heterogeneous across different boroughs and times of day.

Understanding these tipping dynamics matters significantly for two reasons. Operationally, driver compensation relies heavily on tips; accurately mapping regions and temporal windows with high expected tip values can optimize fleet distribution and improve driver livelihoods. Methodologically, this dataset serves as a rigorous benchmark for tabular machine learning, requiring the synthesis of high-cardinality spatial features, cyclical temporal features, and continuous monetary variables to predict an imperfectly distributed label.

# Prior Work & Methodology Alignment

Previous transportation analytics studies using TLC records have predominantly focused on macroscopic trends: demand forecasting, travel time estimation, and broader spatial mobility patterns. When examining tipping specifically, prior behavioral economic studies generally treat it as an outcome influenced by service quality and payment friction. On the machine learning front, predicting such targets is often tackled using generalized linear models or tree-based ensembles.

However, attempting to model tipping as a single continuous target often fails because the underlying data generation process is dual-natured: the decision to tip at all is fundamentally different from the decision of how much to tip.

To address this, our project adopts a **Two-Stage Hurdle Model architecture**:

1. **Stage 1 (Classification):** A probabilistic model predicting the likelihood of a trip receiving any recorded electronic tip.
2. **Stage 2 (Regression/Generative):** A conditional model estimating the exact tip amount, strictly trained on the subset of data where a tip occurred.

This decoupled approach prevents the large influx of zero-tip rides from skewing the regression mechanics, aligning our technical pipeline more closely with the true behavioral structure of the data.

# Current Prototype Scope

This prototype serves as a functional proof of concept. It uses the 2025 NYC TLC yellow and green taxi trip records, joins them with geographic zone metadata, performs feature engineering, and deploys an initial two-stage ML pipeline using histogram-based gradient boosting baselines.

This milestone validates the end-to-end data pipeline, establishes evaluation metrics such as ROC-AUC and RMSE, and provides a functioning interactive deployment.

# Pathway To The Proposed Architecture

While the current prototype captures basic non-linearities using tree-based baselines, the final proposal targets a more expressive deep learning approach for higher-order spatiotemporal interactions.

- **Backbone Upgrade (Tabular Transformer):** Transition from tree-based models to a PyTorch-based tabular transformer to learn richer interactions across mixed feature types.
- **Head Upgrade (Mixture Density Network):** Replace the second-stage point estimate with a probabilistic conditional tip distribution so the model can represent uncertainty and multimodal tipping behavior.

# Data Limitations

As noted in the TLC data dictionary, the `tip_amount` field is only automatically populated for credit-card transactions; cash tips are omitted. Consequently, the dataset is explicitly filtered to include only credit-card trips. The task here is predicting recorded electronic tipping behavior, not total real-world tipping behavior.
