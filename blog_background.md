# Background & Motivation

Tipping is a complex, noisy real-world behavior shaped by a confluence of geography, temporal dynamics, trip context, and underlying socioeconomic patterns. In the context of New York City's highly structured transportation ecosystem, tipping presents an ideal yet challenging Applied Machine Learning problem. The NYC TLC dataset provides immense volume and structural richness, yet the ultimate target—whether a rider tips and by how much—is highly heterogeneous across different boroughs and times of day.

Understanding these tipping dynamics matters significantly for two reasons. Operationally, driver compensation relies heavily on tips; accurately mapping regions and temporal windows with high expected tip values can optimize fleet distribution and improve driver livelihoods. Methodologically, this dataset serves as a rigorous benchmark for tabular machine learning, requiring the synthesis of high-cardinality spatial features (over 260 distinct taxi zones), cyclical temporal features, and continuous monetary variables to predict an imperfectly distributed label.

# Prior Work & Methodology Alignment

Previous transportation analytics studies using TLC records have predominantly focused on macroscopic trends: demand forecasting, travel time estimation, and broader spatial mobility patterns. When examining tipping specifically, prior behavioral economic studies generally treat it as an outcome influenced by service quality and payment friction. On the machine learning front, predicting such targets is often tackled using generalized linear models or tree-based ensembles.

However, attempting to model tipping as a single continuous target often fails because the underlying data generation process is dual-natured: the decision to tip at all (often binary, influenced by payment method or personal habit) is fundamentally different from the decision of _how much_ to tip (often a percentage of the fare, influenced by trip distance and duration).

To address this, our project adopts a **Two-Stage Hurdle Model architecture**:

1. **Stage 1 (Classification):** A probabilistic model predicting the likelihood of a trip receiving any recorded electronic tip ($P(tip\_amount > 0)$).
2. **Stage 2 (Regression/Generative):** A conditional model estimating the exact tip amount, strictly trained on the subset of data where a tip occurred.

This decoupled approach prevents the large influx of zero-tip rides from skewing the regression mechanics, aligning our technical pipeline deeply with the true behavioral structure of the data.

# Current Prototype Scope

This prototype serves as our functional proof-of-concept to demonstrate meaningful progress. It utilizes the 2025 NYC TLC yellow and green taxi trip records, joins them with geographic zone metadata, executes necessary feature engineering, and deploys an initial two-stage ML pipeline using robust baseline models (Histogram-based Gradient Boosting).

This milestone validates our end-to-end data processing capabilities, establishes our evaluation metrics (ROC-AUC for Stage 1, RMSE for Stage 2), and provides a functioning interactive deployment.

# Pathway to the Proposed Architecture

While our current prototype successfully captures basic non-linearities using tree-based baselines, our final proposal targets a much more sophisticated deep learning approach to capture high-order spatiotemporal interactions. In the next phase of the project, we will implement the following upgrades:

- **Backbone Upgrade (Tabular Transformer):** We will transition from tree-based models to a PyTorch-based Tabular Transformer. By encoding mixed tabular features as tokens and applying self-attention, the model will intrinsically learn complex, high-order cross-features (e.g., $pickup\_zone \times dropoff\_zone \times time \times fare$).
- **Head Upgrade (Mixture Density Network - MDN):** Instead of outputting a static point estimate for the conditional tip amount, Stage 2 will be upgraded with a Generative AI head (MDN). This will output a conditional probability density function ($\{\pi_k(x), \mu_k(x), \sigma_k(x)\}_{k=1}^K$), allowing us to model the rich variance and uncertainty inherent in human tipping behavior.

# Data Limitations

As per the TLC data dictionary, the `tip_amount` field is only automatically populated for credit-card transactions; cash tips are omitted. Consequently, our dataset is explicitly filtered to include only credit-card trips. Our task is strictly defined as predicting _recorded electronic tipping behavior_, not global tipping compliance.that, this prototype focuses on credit-card trips and describes the task as predicting recorded electronic tips rather than all real-world tipping behavior.
