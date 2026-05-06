# Background and Motivation

Tipping in taxis is noisy. It depends on the trip, the area, the time, and the rider. That makes it a good applied machine learning problem. The NYC TLC data is large and detailed, but the target is not simple.

The main goal of this project is to model recorded electronic tips. This matters because tips are part of driver income. It also matters because the data has many common ML issues from class. The target has many zeros. The positive tips are skewed. The inputs mix numbers, categories, time, and location.

# Main Idea

Tip amount is not modeled as one plain regression target. That would mix two different behaviors. First, a rider either leaves a recorded electronic tip or does not. Then, if there is a tip, the amount has to be modeled.

So the project uses a two stage model.

1. Stage 1 predicts if the trip gets a recorded electronic tip.
2. Stage 2 predicts the positive tip amount for trips that did get a tip.

This is the hurdle model idea from class. It fits the data better than treating all zero and positive values as one continuous target.

# Dataset

The project uses NYC TLC Yellow and Green taxi records from 2024 and 2025. The trips are joined with the TLC taxi zone lookup table. This gives pickup and dropoff boroughs and zone names.

The project only keeps credit card trips. This is important. The TLC `tip_amount` field records electronic tips. It does not include cash tips. So the target is recorded electronic tipping, not all tipping.

The data is cleaned before training. Trips with invalid fare, distance, duration, or dates are removed. The split is based on time. Most of 2024 is used for training. Late 2024 is used for validation. All of 2025 is used for testing.

# Models

The first baseline is a simple logistic and ridge hurdle model. It gives a clean starting point.

The second baseline is a boosted tree hurdle model. This is a strong tabular model. It works well with fare fields, distance, time, and zone features.

The deep model is a Tabular Transformer with a Mixture Density Network head. The transformer learns from categorical and numeric trip features. The MDN head predicts a distribution over positive tips instead of one number.

The tree model gives the best point predictions. The Transformer MDN is still useful because it gives uncertainty. That lets the demo show lower risk zones, intervals, and safer ride choices.

# App and Demo

The Hugging Face demo lets a user inspect the model. A user can enter a trip and get the tip probability, the expected positive tip, and the final expected tip.

The Model Lab also compares the tree hurdle model and the Transformer MDN on the same ride. This shows the average prediction beside a lower and upper tip range.

The app also has a what if panel. It shows how predictions change when hour, fare, distance, or duration changes. There are maps and tables for borough and zone patterns.

The Driver Copilot is the language layer. A driver can ask about a pickup area or compare two ride options. The copilot looks up model results and gives a grounded answer. It does not just make up advice. It uses expected tip, tip probability, downside risk, and trip counts from the project artifacts.

# Limitations

The biggest limitation is cash tips. They are not observed in the TLC tip field. A zero tip in this dataset means no electronic tip was recorded. It does not always mean the rider left no tip.

The model is also not causal. It can show patterns in the data. It cannot prove that choosing one area will cause a higher tip. The demo should be read as a planning and exploration tool, not a guarantee of income.
