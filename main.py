from classiq import *
from typing import List

from classiq.interface.backend.backend_preferences import ClassiqBackendPreferences
from classiq.interface.executor.execution_preferences import ExecutionPreferences
from classiq.interface.model.model import Model as Model_Designer

import numpy as np
import json
from numpy import ndarray


@QStruct
class FloatInfo:
    repr: float
    int_repr: int
    num_zeros: int


@QStruct
class Segment:
    coeffs: List[FloatInfo]


@QStruct
class Domain:
    segments: List[Segment]


@QStruct
class OldStruct:
    reprs: List[float]
    int_reprs: List[int]
    num_zeros: List[int]


@QStruct
class ListWrapper:
    lst: List[int]


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


def get_float_info(val: float, num_bits_val: int, num_bits_integer: int, num_bits_decimal: int) -> FloatInfo:
    """
    Converts given float to FloatInfo struct.
    :param val: Value to convert, between 0 and 1.
    :param num_bits_val: Number of bits in val conversion.
    :param num_bits_integer: Number of bits in the result.
    :param num_bits_decimal: Number of bits to use for value conversion.
    :return: FloatInfo.
    """
    assert val > 0
    assert val < 1
    assert num_bits_decimal >= num_bits_val

    bits = real_to_bits(val, num_bits_val)
    bits = np.concatenate((bits, [0] * (num_bits_decimal - num_bits_val)))
    first_one_ind = np.where(bits == 1)[0][0]
    zeros_left = num_bits_integer + first_one_ind
    return FloatInfo(bits_to_real(bits), bits_to_int(bits), zeros_left)


@QFunc
def prepare_fraction_plain(int_repr: QParam[int], num_digits: QParam[int], out: Output[QNum]):
    """
    Prepares QNum in a given int state without prepending with zeros.
    :param int_repr: Integer value to prepare.
    :param num_digits: Number of binary digits in int_repr.
    :param out: Prepared QNum.
    """
    prepare_int(int_repr, out)
    reinterpret_num(False, num_digits, out)


@QFunc
def prepare_fraction(int_repr: QParam[int], num_zeros: QParam[int], num_decimal_bits: QParam[int], out: Output[QNum]):
    """
    Prepares QNum in a given fraction state described by the input parameters.
    :param int_repr: Integer representation of the decimal value starting from the highest bit = 1.
    :param num_zeros: Number of zeros to prepend on the left.
    :param num_decimal_bits: Number of decimal bits in the final reinterpretation.
    :param out: Prepared QNum.
    """
    int_arr = QArray('int_arr')
    prepare_int(int_repr, int_arr)
    zeros = QArray('zeros')
    allocate(num_zeros, zeros)
    join(int_arr, zeros, out)
    reinterpret_num(False, num_decimal_bits, out)


@QFunc
def multiplication(a: QParam[float], x: QNum, y: Output[QNum]):
    y |= a * x


@QFunc
def mult_add(a: QParam[float], x: QNum, y: QNum):
    tmp = QNum('tmp')
    within_apply(lambda: multiplication(a, x, tmp), lambda: inplace_add(tmp, y))


@QFunc
def mult_add_extra(a: QParam[float], i: QParam[int], x: QArray, y: QNum):
    # if_(i == 0, lambda: IDENTITY(y), lambda: mult_add_extra_2(a, i, y))
    if_(i == 0, lambda: mult_add_extra_0(a, x, y), lambda: IDENTITY(y))
    # tmp = QNum('tmp')
    # arr = ListWrapper([0, 1])
    # prepare_int(arr.lst[i], tmp)

    # mult_add(a, x, y)

    # tmp2 = QArray('tmp2')
    # join(x, tmp, tmp2)
    #
    # mult_add_extra_2(a, tmp2, y)


@QFunc
def mult_add_extra_0(a: QParam[float], x: QArray, y: QNum):
    tmp = QArray('tmp')
    allocate(1, tmp)
    mult_add(a, x, y)


@QFunc
def mult_add_extra_2(a: QParam[float], x: QNum, y: QNum):
    reinterpret_num(False, 2, x)
    mult_add(a, x, y)


@QFunc
def frac_add(int_repr: QParam[int], num_zeros: QParam[int], num_decimals: QParam[int], y: QNum):
    tmp = QNum('tmp')
    within_apply(lambda: prepare_fraction(int_repr, num_zeros, num_decimals, tmp), lambda: inplace_add(tmp, y))


# @QFunc
# def segment_selector(domain: QParam[Domain], selector: QNum, y: Output[QNum]):
#     allocate_num(precision_y_integer + precision_y_decimal, False, precision_y_decimal, y)
#     repeat(2, lambda i: quantum_if(selector == i, lambda: frac_add(domain.segments[i].coeffs[0].int_repr, domain.segments[i].coeffs[0].num_zeros, precision_y_decimal, y)))
#     # repeat(2, lambda i: quantum_if(selector == i, lambda: mult_add(domain.segments[i].coeffs[1].repr, selector, y)))


@QFunc
def segment_selector(domain: QParam[OldStruct], selector: QNum, x: QArray, y: Output[QNum]):
    allocate_num(precision_y_integer + precision_y_decimal, False, precision_y_decimal, y)
    # repeat(2, lambda i: quantum_if(selector == i, lambda: frac_add(domain.int_reprs[i], domain.num_zeros[i], precision_y_decimal, y)))

    # tmp = QNum('tmp')
    # join(x, selector, tmp)

    # reinterpret_num(False, precision_x - 1, x)
    repeat(2, lambda i: quantum_if(selector == i, lambda: mult_add_extra(domain.reprs[i + 2], i, x, y)))


@QFunc
def segment_selector_1(domain: QParam[OldStruct], x: QNum, y: Output[QNum]):
    allocate_num(precision_y_integer + precision_y_decimal, False, precision_y_decimal, y)
    frac_add(domain.int_reprs[0], domain.num_zeros[0], precision_y_decimal, y)
    mult_add(domain.reprs[1], x, y)


# @QFunc
# def segment_selector(domain: QParam[OldStruct], selector: QNum, x: QNum, y: Output[QNum]):
#     # allocate_num(precision_y_integer + precision_y_decimal, False, precision_y_decimal, y)
#     # repeat(2, lambda i: quantum_if(selector == i, lambda: frac_add(domain.int_reprs[i], domain.num_zeros[i], precision_y_decimal, y)))
#
#     allocate_num(6, False, 5, y)
#     reinterpret_num(False, precision_x - 1, x)
#     mult_add(0.5, x, y)
#
#     # repeat(2, lambda i: quantum_if(selector == i, lambda: mult_add_extra(domain.reprs[i + 2], i, x, y)))


@QFunc
def compute_tanh(precision: QParam[int], x: QNum, tanh_x: Output[QNum]):
    # # allocate_num(precision_y_integer + precision_y_decimal, False, precision_y_decimal, tanh_x)
    # # frac_add(domain_data.int_reprs[0], domain_data.num_zeros[0], precision_y_decimal, tanh_x)
    # prepare_fraction(1, 4, 5, tanh_x)
    # # tmp = QNum('tmp')
    # # prepare_fraction(1, 1, 2, tmp)
    # # reinterpret_num(False, precision_x, x)
    # mult_add(0.125, x, tanh_x)

    segment_selector(domain_data, x[precision_x - 1], x[0:precision_x - 1], tanh_x)

    # segment_selector_1(domain_data, x, tanh_x)


@QFunc
def main(x: Output[QNum], y: Output[QNum]):
    allocate_num(precision_x, False, precision_x, x)
    hadamard_transform(x)
    compute_tanh(precision_x, x, y)

    # prepare_fraction(1, 2, 3, x)
    # prepare_fraction(2, 2, 4, y)
    # mult_add(0.9375, x, y)


def save_new_file(program_to_save, file_name):
    file = open(file_name, "w")
    file.write(program_to_save)
    file.close()


if __name__ == '__main__':
    precision_x = 2
    precision_y_integer = 1
    precision_y_decimal = 4
    precision_consts = 3

    # fitting_coeffs = [0.03348032, 0.77149773]
    fitting_coeffs = [0.068893, 0.786448]
    # fitting_coeffs = [0.125, 0.5]

    # fitting_coeffs = [[0.1, 0.9360805], [0.17697803, 0.59464997]]  # 0.00256892
    # domain_data = Domain([Segment([get_float_info(num, precision_consts, precision_y_integer, precision_y_decimal) for num in segment]) for segment in fitting_coeffs])

    # fitting_coeffs = [0.1, 0.17697803, 0.9360805, 0.59464997]
    # fitting_coeffs = [0.125, 0.125, 0.125, 0.125]
    # int_repr_list = [float_info.int_repr for segment in domain_data.segments for float_info in segment.coeffs]
    # num_zeros_list = [float_info.num_zeros for segment in domain_data.segments for float_info in segment.coeffs]
    repr_list = [get_float_info(val, precision_consts, precision_y_integer, precision_y_decimal).repr for val in fitting_coeffs]
    int_repr_list = [get_float_info(val, precision_consts, precision_y_integer, precision_y_decimal).int_repr for val in fitting_coeffs]
    num_zeros_list = [get_float_info(val, precision_consts, precision_y_integer, precision_y_decimal).num_zeros for val in fitting_coeffs]
    domain_data = OldStruct(repr_list, int_repr_list, num_zeros_list)

    qmod = create_model(main)
    qmod = set_constraints(qmod, Constraints(optimization_parameter='width'))
    qprog = synthesize(qmod)

    # save_new_file(qmod, 'qmod2.qmod')
    # save_new_file(qprog, 'qprog2.qprog')

    # # show(qprog)
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
