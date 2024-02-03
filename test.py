from classiq import *


@QFunc
def prepare_minus_state(x: QBit):
    X(x)
    H(x)


@QFunc
def create_initial_state(reg: QArray[QBit]):
    apply_to_all(H, reg)


@QFunc
def linear_func(a: QParam[float], b: QParam[float], x: QNum, y: QNum):
    tmp = QNum('tmp')
    tmp |= a * x + b
    inplace_add(tmp, y)


@QFunc
def inplace_linear_attempt(a: QParam[int], b: QParam[int], x: QNum, y: QNum):
    tmp = QNum('tmp')
    linear_func(a, b, x, tmp)
    inplace_xor(tmp, y)


@QFunc
def inplace_linear_func(a: QParam[int], b: QParam[int], x: QNum, y: QNum):
    tmp = QNum('tmp')
    within_apply(compute=lambda: linear_func(a, b, x, tmp), action=lambda: inplace_xor(tmp, y))


@QFunc
def control_logic(a: QParam[list[int]], b: QParam[list[int]], controller: QNum, x: QNum, y: QNum):
    repeat(count=a.len(), iteration=lambda i: quantum_if(controller == i, lambda: inplace_linear_func(a[i], b[i], x, y)))


def print_parsed_counts(job):
    results = job.result()
    parsed_counts = results[0].value.parsed_counts
    for parsed_state in parsed_counts:
        print(parsed_state.state)


def print_depth_width(quantum_program):
    generated_circuit = GeneratedCircuit.parse_raw(quantum_program)
    print(f"Synthesized circuit width: {generated_circuit.data.width}, depth: {generated_circuit.transpiled_circuit.depth}")


@QFunc
def set_val(val: QParam[int], target: QNum):
    tmp = QNum('tmp')
    within_apply(lambda: prepare_int(val, tmp), lambda: inplace_add(tmp, target))
    # prepare_int(val, tmp)
    # inplace_add(tmp, target)


@QFunc
def main(x: Output[QNum], y: Output[QNum]):
    a = 2
    b = 1

    allocate_num(4, False, 0, x)
    hadamard_transform(x)
    # allocate_num(6, False, 0, y)
    # set_val(1, y)
    probs = [0] * (2 ** 6)
    probs[1] = 1
    prepare_state(probs, 0, y)

    linear_func(a, b, x, y)

    # allocate_num(1, False, 0, controller)
    # H(controller)

    # control_logic(a, b, controller, x, y)


qmod = create_model(main)
qmod = set_constraints(qmod, Constraints(optimization_parameter='width'))
qprog = synthesize(qmod)
print_depth_width(qprog)
# show(qprog)
job = execute(qprog)
print_parsed_counts(job)
