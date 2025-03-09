from modules.rl_qaoa import RL_QAOA
from modules.data_process import add_constraint
import numpy as np


from modules.data_process import add_constraint
from modules.data_process import data_to_QUBO
import json
import numpy as np

"""
### **Applying Soft Constraints to the QUBO Model**

This section of the code **defines a constrained QUBO model** for optimizing energy distribution while incorporating additional constraints.

**Process:**
1. **Load variance data**: Load the dataset representing energy source fluctuations.
2. **Define QUBO parameters**:
   - `λ` (l): The variance sensitivity factor.
   - `hamming_weights`: Number of power plants to be selected.
3. **Generate the QUBO matrix**: Convert variance data into a **QUBO formulation**.
4. **Incorporate Soft Constraints**:
   - Apply **a penalty term** to enforce constraints.
   - Ensure the solution adheres to a predefined **hamming weight constraint (e.g., selecting exactly 4 power plants)**.

**Key Insight:**
- The constraint **limits the number of selected power plants** while optimizing energy stability.
- **Soft constraints** allow flexibility while penalizing solutions that violate the selection condition.

"""

# Load the dataset containing energy fluctuation matrices
with open(f"./data/matrices9by9.json", "r") as f:
    matrices_data = json.load(f)

# Define lambda (variance sensitivity factor) and hamming weight (number of selected power plants)
l = 5
hamming_weights = 4

# Convert variance data into QUBO format
QUBO_matrix = data_to_QUBO(np.array(matrices_data[1]), hamming_weights, l)

# Apply Soft Constraints to the QUBO matrix
n = len(QUBO_matrix)  # Number of energy sources
penalty = 1.5  # Penalty factor for constraint violations

# Generate a constraint matrix ensuring exactly 4 power plants are selected
qubo_const = (
    QUBO_matrix
    + add_constraint(node_hamming_weights=[1] * n, hamming_weights=hamming_weights)
    * penalty
)

# Define the stopping condition for RL-QAOA (number of remaining nodes)
n_c = 2  # RL-QAOA halts when this number of nodes remains

# Initialize QAOA parameters
init_params = np.reshape(np.array([0.11805387, -0.34618458] * (n - n_c)), -1)

# Define the initial edge temperature matrix for RL-QAOA training
b_vector = np.array([[50.0] * int(n**2) for i in range(n - n_c)])

# Initialize RL-QAOA with the constraint-enhanced QUBO
rl_qaoa = RL_QAOA(
    qubo_const,
    n_c=n_c,
    init_paramter=init_params,
    b_vector=b_vector,
    QAOA_depth=1,
    learning_rate_init=[0.2, 0.50],
)

# Perform brute-force search to obtain the correct solution
rl_qaoa.n_c = n
brute_force_result = rl_qaoa.rqaoa_execute()
correct_ans = brute_force_result[2]
rl_qaoa.n_c = n_c  # Reset n_c to original value

# Train RL-QAOA model with the soft constraint QUBO
rl_qaoa.RL_QAOA(episodes=1, epochs=1, log_interval=5, correct_ans=correct_ans)
