# Background & Motivation

Tipping is a complex, noisy real-world behavior shaped by geography, temporal dynamics, trip context, and underlying socioeconomic patterns. In the context of New York City's highly structured transportation ecosystem, tipping presents an ideal learning problem. The NYC TLC dataset provides immense volume and feature richness, yet whether a rider tips and by how much is highly varied across different boroughs and times of day.

Understanding these tipping dynamics matters significantly for two reasons. Driver compensation relies heavily on tips sp accurately mapping regions and temporal windows with high expected tip values can optimize fleet distribution and improve driver livelihoods. This dataset serves as a rigorous benchmark for tabular machine learning, requiring the synthesis of high-cardinality spatial features, cyclical temporal features, and continuous monetary variables to predict an imperfectly distributed label.

# Prior Work & Methodology Alignment

Previous transportation analytics studies using TLC records have predominantly focused on macroscopic trends like demand forecasting, travel time estimation, and broader spatial mobility patterns. When examining tipping specifically, prior behavioral economic studies generally treat it as an outcome influenced by service quality and payment friction. On the machine learning front, predicting such targets is often tackled using generalized linear models or tree-based ensembles.

However, attempting to model tipping as a single continuous target often fails because the underlying data generation process is dual-natured as the decision to tip at all is fundamentally different from the decision of how much to tip.

To address this, our project adopts a **Two-Stage Hurdle Model architecture**:

1. **Stage 1 (Classification):** A probabilistic model predicting the likelihood of a trip receiving any recorded electronic tip.
2. **Stage 2 (Regression):** A conditional model estimating the exact tip amount, strictly trained on the subset of data where a tip occurred.

This decoupled approach prevents the large influx of zero-tip rides from skewing the regression mechanics, aligning the technical pipeline more closely with the behavioral structure of the data.

# Current Prototype Scope

This prototype serves as a functional proof of concept. It uses the 2025 NYC TLC yellow and green taxi trip records, joins them with geographic zone metadata, performs feature engineering, and deploys an initial two-stage ML pipeline using histogram-based gradient boosting baselines. This validates the end-to-end data pipeline, establishes evaluation metrics such as ROC-AUC and RMSE, and provides a functioning interactive deployment.

# What Machine Learning Has Been Implemented

The current prototype implements a real supervised learning baseline rather than a mock interface. We train separate model bundles for yellow taxis and green taxis, since the two services have different trip distributions and operating patterns. Each bundle contains two connected models.

The first model is a **HistGradientBoostingClassifier** from scikit-learn. It predicts the probability that a trip receives any recorded electronic tip. The second model is a **HistGradientBoostingRegressor** trained only on rides with a positive recorded tip. It predicts the tip amount conditional on tipping.

The training features come from the raw TLC trip records after cleaning and feature engineering. They include:

- temporal features such as pickup hour, pickup weekday, and pickup month,
- trip-level numeric features such as trip distance, fare amount, and trip duration,
- operational variables such as vendor ID, passenger-count bucket, rate code, and store-and-forward flag,
- spatial context from the taxi zone lookup, including pickup borough, pickup zone, dropoff borough, and dropoff zone.

Categorical variables are one-hot encoded and numeric variables are passed through directly. The data split is time-based: January through September for training, October for validation-style development, and November through December for testing.

# How The Prediction Works In The App

When a user enters a hypothetical trip in the app, the interface constructs one feature row using the same schema as the training pipeline. That row is sent to the classifier first to estimate the probability of a recorded electronic tip. The same row is then sent to the regressor to estimate the conditional tip amount.

The app returns three outputs:

1. the probability of a recorded electronic tip,
2. the predicted tip amount if a tip occurs,
3. the expected tip value, computed as `tip probability x conditional tip amount`.

This two-stage structure matters because the decision to tip and the amount tipped are different behaviors. Modeling them separately gives a more realistic baseline than treating the entire problem as one continuous regression target.

# Pathway To The Proposed Architecture

While the current prototype captures basic non-linearities using tree-based baselines, the final proposal targets a more expressive deep learning approach for higher-order spatiotemporal interactions.

- **Backbone Upgrade (Tabular Transformer):** Transition from tree-based models to a PyTorch-based tabular transformer to learn richer interactions across mixed feature types.
- **Head Upgrade (Mixture Density Network):** Replace the second-stage point estimate with a probabilistic conditional tip distribution so the model can represent uncertainty and multimodal tipping behavior.

# Data Limitations

As noted in the TLC data dictionary, the `tip_amount` field is only automatically populated for credit-card transactions; cash tips are omitted. Consequently, the dataset is explicitly filtered to include only credit-card trips. The task here is predicting recorded electronic tipping behavior, not total real-world tipping behavior.
