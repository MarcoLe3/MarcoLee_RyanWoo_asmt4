#------------------------------------
# Author: T. D. Devlin 
#-----------------------------------

import math
from math import sin, pi
from random import random


def f(x):
    return sin(pi * x)


def generate_training_examples(n=2):
    xs = [random() * 2 - 1 for _ in range(n)]
    return [(x, f(x)) for x in xs]


def fit_without_reg(examples):
    """Computes values of w0 and w1 that minimize the sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    """
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##
    x1, y1 = examples[0]
    x2, y2 = examples[1]
    diff_y = y2 - y1
    diff_x = x2 - x1
    w1 = diff_y / diff_x
    w0 = y1 - w1 * x1
    ## END YOUR CODE ##
    return w0, w1


def fit_with_reg(examples, lambda_hp):
    """Computes values of w0 and w1 that minimize the regularized sum-of-squared-errors cost function

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    - lambda_hp: a float representing the value of the lambda hyperparameter; a larger value means more regularization
    """
    w0 = 0
    w1 = 0
    ## BEGIN YOUR CODE ##
    n = 1000
    lr = 0.05
    for i in range(n):
        grad_w0 = 0
        grad_w1 = 0
        for data in examples:
            x, y = data
            err = y - (w0+w1*x)
            grad_w0 += -2 * err
            grad_w1 += -2 * x * err
        
        grad_w0 += 2 * lambda_hp * w0
        grad_w1 += 2 * lambda_hp * w1
        w0 = w0 - lr*grad_w0
        w1 = w1 - lr*grad_w1
    ## END YOUR CODE ##
    return (w0, w1)


def test_error(w0, w1):
    n = 100
    xs = [i/n for i in range(-n, n + 1)]
    return sum((w0 + w1 * x - f(x)) ** 2 for x in xs) / len(xs)


if __name__ == "__main__": 
    
    ## BEGIN YOUR SIMULATION CODE ##
    res_with = 0
    res_without = 0
    for i in range(1000):
        examples = generate_training_examples()
        without_reg_w0, without_reg_w1 = fit_without_reg(examples)
        with_reg_w0, with_reg_w1 = fit_with_reg(examples,1.0)
        res_without += test_error(without_reg_w0,without_reg_w1)
        res_with += test_error(with_reg_w0, with_reg_w1)

    res_with = res_with / 1000
    res_without = res_without / 1000
    print(f"res_with avg: {res_with}")
    print(f"res_without avg: {res_without}")
