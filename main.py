from classiq import *

from classiq.interface.backend.backend_preferences import ClassiqBackendPreferences
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.model.model import Model as Model_Designer

import numpy as np
import json
from numpy import ndarray


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


def real_to_bits(num: float, num_bits: int) -> ndarray:
    """
    Returns the closest fixed point binary representation of a given real number in big endian notation.
    :param num: Number to convert.
    :param num_bits: Desired number of bits.
    :return: 1D array of bits that represent the given number.
    """
    bits = np.zeros(num_bits, dtype=int)
    represented = 0
    min_step = 2 ** -num_bits
    for i in range(len(bits)):
        bit_value = 2 ** (-i - 1)
        if represented + bit_value - min_step / 2 <= num:
            bits[i] = 1
            represented += bit_value
    return bits


def bits_to_real(bits: ndarray) -> float:
    """
    Returns a real number represented by given binary array representation in fixed point big endian notation.
    :param bits: 1D array of bits.
    :return: Represented real number.
    """
    return sum([2 ** (-ind - 1) for ind, val in enumerate(bits) if val == 1])


def bits_to_int(bits: ndarray) -> int:
    """
    Returns an integer represented by given binary array in big endian notation.
    :param bits: 1D array of bits.
    :return: Represented real number.
    """
    return sum([2 ** (len(bits) - 1 - ind) for ind, val in enumerate(bits) if val == 1])


@QFunc
def prepare_qnum(out: Output[QNum]):
    """
    Prepares QNum in a value given by val with given precision_x.
    Uses global variables instead of QParams since QParams do not work.
    :param out: Prepared QNum.
    """
    bits = real_to_bits(const_val, precision_y)
    basis_ind = bits_to_int(bits)
    probabilities = [0] * 2 ** len(bits)
    probabilities[basis_ind] = 1
    prepare_state(probabilities, 0, out)
    reinterpret_num(False, precision_y, out)


@QFunc
def compute_tanh(precision: QParam[int], x: QNum, tanh_x: Output[QNum]):
    prepare_qnum(tanh_x)
    # allocate_num(precision_x, False, precision_x, tanh_x)


@QFunc
def main(x: Output[QNum], y: Output[QNum]):
    allocate_num(precision_x, False, precision_x, x)
    hadamard_transform(x)
    compute_tanh(precision_x, x, y)


if __name__ == '__main__':
    precision_x = 4
    precision_y = 4
    const_val = 0.462117
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

    print(f'Max dist overall: {evaluate_score_stage_1(job_result, precision_x)}')
