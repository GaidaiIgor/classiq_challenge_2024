from classiq import *

from classiq.interface.backend.backend_preferences import ClassiqBackendPreferences
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.model.model import Model as Model_Designer

import numpy as np
import json


def evaluate_score_stage_1(results, user_input_precision):
    # The array of values the code is compared against
    calculated_precision = 10
    domain = np.arange(0, 1, 1 / 2 ** calculated_precision)
    expected_y = np.tanh(domain)
    parsed_counts = results[0].value.parsed_counts
    results_dict = {s.state['x']: s.state['y'] for s in parsed_counts}

    # Verify all strings were sampled, also no superpositions
    assert len(results_dict) == 2 ** user_input_precision

    # Comparing to the users results
    measured_y = []
    for x_val in domain:
        x_val_floored = int(x_val * (2 ** user_input_precision)) / (2 ** user_input_precision)
        measured_y.append(results_dict[x_val_floored])

    # The metric assesses what's the largest distance between the expected value and the measured value, the smallest the better
    max_distance = np.max(np.abs(expected_y - np.array(measured_y)))
    return max_distance


def print_depth_width(quantum_program):
    generated_circuit = GeneratedCircuit.parse_obj(json.loads(quantum_program))
    print(f"Synthesized circuit width: {generated_circuit.data.width}, depth: {generated_circuit.transpiled_circuit.depth}")


@QFunc
def compute_tanh(precision: QParam[int], x: QNum, tanh_x: Output[QNum]):
    allocate_num(precision, False, precision, tanh_x)


@QFunc
def main(x: Output[QNum], y: Output[QNum]):
    allocate_num(num_qubits=precision, is_signed=False, fraction_digits=precision, out=x)
    hadamard_transform(x)
    compute_tanh(precision, x, y)


precision = 4
qmod = create_model(main)
qprog = synthesize(qmod)
print_depth_width(qprog)
job_result = execute(qprog).result()
parsed_counts = job_result[0].value.parsed_counts
sorted_counts = sorted(parsed_counts, key=lambda x: x.state["x"])
for state_counts in sorted_counts:
    print('x: ', state_counts.state['x'], '| y: ', state_counts.state['y'], '| Diff: ', abs(state_counts.state['y'] - np.tanh(state_counts.state['x'])))

diffs = np.array([state.state['y'] - np.tanh(state.state['x']) for state in sorted_counts])
print(f'Max dist at points: {np.max(abs(diffs))}')

print(f'Max dist overall: {evaluate_score_stage_1(job_result, precision)}')
