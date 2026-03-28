# Background

Tipping is a noisy real-world behavior shaped by geography, time, trip context, and rider payment patterns. In New York City taxi data, that makes tipping an interesting applied machine learning problem: the data is large, structured, and strongly tied to real transportation behavior, but the outcome is still uncertain and heterogeneous across neighborhoods and times of day.

This problem matters for two reasons. First, tipping affects driver income, so understanding where and when recorded electronic tips are more likely can support planning decisions. Second, taxi trip records provide a realistic benchmark for tabular machine learning because they combine spatial features, temporal features, monetary variables, and imperfect labels.

# Prior Work

Prior work on tipping behavior generally treats tipping as a behavioral outcome influenced by service context, socioeconomic factors, and payment method. More broadly, transportation analytics has used trip-record data to study demand, travel times, zone-level mobility patterns, and geographic disparities across cities. On the machine learning side, structured tabular prediction problems like this are often approached with linear models, tree-based methods, gradient boosting, and increasingly transformer-style architectures for mixed categorical and numeric features.

For this project, the proposal points toward a two-stage hurdle-style model. The first stage asks whether a trip receives a tip at all. The second stage models how large the tip is, conditional on a tip occurring. That structure is appropriate because zero-tip and positive-tip behavior are generated differently, and a single regression target can blur those regimes together.

# Prototype Scope

The prototype is intentionally smaller than the final proposal. It uses the 2025 NYC TLC yellow and green taxi trip records already collected by the team, joins them with taxi-zone metadata, applies basic cleaning, and trains simple baseline models. This demonstrates progress on the full project while leaving room for later upgrades such as stronger tabular models, uncertainty-aware prediction, and zone-level recommendation maps.

# Data Limitation

The TLC data dictionaries note an important limitation: `tip_amount` is automatically populated for credit-card tips, and cash tips are not included. Because of that, this prototype focuses on credit-card trips and describes the task as predicting recorded electronic tips rather than all real-world tipping behavior.
