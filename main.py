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
    measured_y = np.array(measured_y)

    diff = np.abs(expected_y - measured_y)
    max_distance = np.max(diff)
    max_ind = np.argmax(diff)
    return max_distance, domain[max_ind]


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


def transform_val(val: float, num_bits_val: int, num_bits_integer: int, num_bits_decimal: int) -> tuple[float, int, int]:
    """
    Rounds given value between 0 and 1 to specified precision and returns representations of the result.
    :param val: Value to convert, between 0 and 1.
    :param num_bits_val: Number of bits in val conversion.
    :param num_bits_integer: Number of bits in the result.
    :param num_bits_decimal: Number of bits to use for value conversion.
    :return: 1) Float representation; 2) Int representation; 3) Number of zeros on the left from the highest 1.
    """
    assert val >= 0
    assert val < 1
    assert num_bits_decimal >= num_bits_val

    bits = real_to_bits(val, num_bits_val)
    bits = np.concatenate((bits, [0] * (num_bits_decimal - num_bits_val)))
    first_one_ind = np.where(bits == 1)[0][0]
    zeros_left = num_bits_integer + first_one_ind
    return bits_to_real(bits), bits_to_int(bits), zeros_left


@QFunc
def prepare_qnum_plain(int_repr: QParam[int], num_digits: QParam[int], out1: Output[QNum]):
    """
    Prepares QNum in a given int state without prepending with zeros.
    :param int_repr: Integer value to prepare.
    :param num_digits: Number of binary digits in int_repr.
    :param out1: Prepared QNum.
    """
    prepare_int(int_repr, out1)
    reinterpret_num(False, num_digits, out1)


@QFunc
def prepare_qnum_append(int_repr: QParam[int], num_zeros: QParam[int], num_decimal_bits: QParam[int], out2: Output[QNum]):
    """
    Prepares QNum in a given int state and prepends with specified number of zeros.
    :param int_repr: Integer value to prepare.
    :param num_zeros: Number of zeros to prepend on the left.
    :param num_decimal_bits: Number of decimal bits for reinterpretation.
    :param out2: Prepared QNum.
    """
    int_arr = QArray('int_arr')
    prepare_int(int_repr, int_arr)
    zeros = QArray('zeros')
    allocate(num_zeros, zeros)
    join(int_arr, zeros, out2)
    reinterpret_num(False, num_decimal_bits, out2)


# @QFunc
# def prepare_qnum(cond: QParam[bool], int_repr: QParam[int], num_zeros: QParam[int], num_digits: QParam[int], out: Output[QNum]):
#     pass
    # prepare_qnum_append(int_repr, num_zeros, out1)

    # if cond:
    #     prepare_int(int_repr, out1)
    #     reinterpret_num(False, num_digits, out1)
    # else:
    #     int_arr = QArray('int_arr')
    #     prepare_int(int_repr, int_arr)
    #     zeros = QArray('zeros')
    #     allocate(num_zeros, zeros)
    #     join(int_arr, zeros, out1)
    #     reinterpret_num(False, int_arr.len() + zeros.len(), out1)

    # if_(cond, lambda: prepare_qnum_plain(int_repr, num_digits, out), lambda: prepare_qnum_append(int_repr, num_zeros, out))


@QFunc
def multiplication(a: QParam[float], x: QNum, y: Output[QNum]):
    y |= a * x


@QFunc
def mult(a: QParam[float], x: QNum, y: QNum):
    tmp = QNum('tmp')
    within_apply(lambda: multiplication(a, x, tmp), lambda: inplace_add(tmp, y))
    # multiplication(a, x, tmp)
    # inplace_add(tmp, y)


@QFunc
def compute_tanh(precision: QParam[int], x: QNum, tanh_x: Output[QNum]):
    # if taylor_coeffs[0][2] == 0:
    #     prepare_qnum_plain(taylor_coeffs[0][1], taylor_coeffs[0][3], tanh_x)
    # else:
    #     prepare_qnum_append(taylor_coeffs[0][1], taylor_coeffs[0][2], tanh_x)

    prepare_qnum_append(taylor_coeffs[0][1], taylor_coeffs[0][2], precision_y_decimal, tanh_x)

    if len(taylor_coeffs) > 1:
        mult(taylor_coeffs[1][0], x, tanh_x)
    # first_order = QNum('first_order')
    # multiplication(0.786448, x, first_order)
    # inplace_add(first_order, tanh_x)


@QFunc
def main(x: Output[QNum], y: Output[QNum]):
    allocate_num(precision_x, False, precision_x, x)
    hadamard_transform(x)
    compute_tanh(precision_x, x, y)


if __name__ == '__main__':
    precision_x = 6
    precision_y_integer = 1
    precision_y_decimal = 6
    precision_consts = 6
    # taylor_coeffs = [0.462117]
    taylor_coeffs = [0.068893, 0.786448]

    taylor_coeffs = [transform_val(val, precision_consts, precision_y_integer, precision_y_decimal) for val in taylor_coeffs]
    qmod = create_model(main)
    qmod = set_constraints(qmod, Constraints(optimization_parameter='width'))
    qprog = synthesize(qmod)
    print_depth_width(qprog)
    job_result = execute(qprog).result()
    parsed_counts = job_result[0].value.parsed_counts
    sorted_counts = sorted(parsed_counts, key=lambda x: x.state["x"])
    for state_counts in sorted_counts:
        print('x: ', state_counts.state['x'], '| y: ', state_counts.state['y'], '| Diff: ', abs(state_counts.state['y'] - np.tanh(state_counts.state['x'])))

    diffs = np.array([state.state['y'] - np.tanh(state.state['x']) for state in sorted_counts])
    print(f'Max dist at points: {np.max(abs(diffs))}')

    eval_result = evaluate_score_stage_1(job_result, precision_x)
    print(f'Max dist overall: {eval_result[0]} at x = {eval_result[1]}')
