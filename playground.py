import numpy as np
from pymcdm.methods import VIKOR

# Step 1: Define the decision matrix (rows: alternatives, columns: criteria)
decision_matrix = np.array([
    [7, 9, 8],
    [8, 7, 9],
    [6, 8, 7],
    [9, 6, 8]
])

# Step 2: Define the weights for each criterion
weights = np.array([0.4, 0.3, 0.3])

# Step 3: Define the type of each criterion: 1 for benefit, -1 for cost
criteria_types = np.array([1, 1, 1])

# Step 4: Create the VIKOR object
vikor = VIKOR()

# Step 5: Apply the VIKOR method to get rankings
rankings = vikor(decision_matrix, weights, criteria_types)

# Step 6: Find the best alternative (lowest Q value)
best_alternative_index = np.argmin(rankings)
best_alternative = best_alternative_index + 1  # Adjusting index to match alternative number

# Display the rankings and the best alternative
print("VIKOR rankings (lower is better):")
for i, ranking in enumerate(rankings):
    print(f"Alternative A{i+1}: {ranking}")

print(f"\nThe best alternative is: A{best_alternative}")
