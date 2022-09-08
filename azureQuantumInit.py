from azure.quantum.qiskit import AzureQuantumProvider
provider = AzureQuantumProvider (
    resource_id = "/subscriptions/5cbb9ed4-52fa-401a-8cd7-486cad53d05a/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/NTUQuantum",
    location = "westus"
)

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

print("This workspace's targets:")
for backend in provider.backends():
    print("- " + backend.name())

# Create a quantum circuit acting on a single qubit
circuit = QuantumCircuit(1,1)
circuit.name = "Single qubit random"
circuit.h(0)
circuit.measure(0, 0)

# Print out the circuit
circuit.draw()

# Create an object that represents Quantinuum's Syntax Checker target, "quantinuum.hqs-lt-s2-apival".
#   Note that any target you have enabled in this workspace can
#   be used here. Azure Quantum makes it extremely easy to submit
#   the same quantum program to different providers. 
quantinuum_api_val_backend = provider.get_backend("quantinuum.hqs-lt-s2-apival")

# Using the Quantinuum target, call "run" to submit the job. We'll
# use a count of 100 (simulated runs).
job = quantinuum_api_val_backend.run(circuit, count=100)
print("Job id:", job.id())

job_monitor(job)

result = job.result()

# The result object is native to the Qiskit package, so we can use Qiskit's tools to print the result as a histogram.
# For the syntax check, we expect to see all zeroes.
plot_histogram(result.get_counts(circuit), title="Result", number_to_keep=2)

backend = provider.get_backend("quantinuum.hqs-lt-s2")
cost = backend.estimate_cost(circuit, shots=100)
print(f"Estimated cost: {cost.estimated_total} {cost.currency_code}")