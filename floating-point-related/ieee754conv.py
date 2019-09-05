"""Number to IEEE 754 Floating Point standard format and vice-versa."""
import typing as t
import sys

IEEE_754_BITS_SINGLE = {
    "sign": 1,
    "exponent": 8,
    "mantissa": 23,
    "significand": 23,
}

IEEE_754_BITS_DOUBLE = {
    "sign": 1,
    "exponent": 11,
    "mantissa": 52,
    "significand": 52,
}


def is_equal(a: t.Union[float, int],
             b: t.Union[float, int],
             epsilon: float = 1.0e-8) -> bool:
    """Check if two numbers are equal considering floating point errors.

    Font: https://floating-point-gui.de/errors/comparison/
    """
    abs_a = abs(a)
    abs_b = abs(b)
    abs_diff = abs(a - b)

    if a == b:
        return True

    if a == 0 or b == 0 or (abs_a + abs_b < sys.float_info.min):
        return abs_diff < epsilon * sys.float_info.min

    return abs_diff / min(abs_a + abs_b, sys.float_info.max) < epsilon


def dec_to_bin(x: t.Union[float, int], max_bits: int = 64) -> str:
    """Transform a number into its binary representation."""
    if x < 0:
        raise ValueError(
            "Negative values not allowed to binary transformation.")

    def int_to_bin(x: int) -> str:
        """Transform an integer x into its binary representation."""
        b = ""

        while x:
            b += str(x % 2)
            x //= 2

        return b[::-1]

    def frac_to_bin(x: float, max_bits: int) -> str:
        """Transform a x in [0, 1) into its binary representation."""
        b = ""

        while not is_equal(x, 0.0) and len(b) < max_bits:
            x *= 2.0
            whole = int(x)
            b += str(whole)
            x -= whole

        return b

    if is_equal(x, 0):
        return "0"

    if isinstance(x, int):
        return int_to_bin(x)

    whole = int(x)
    return int_to_bin(whole) + "." + frac_to_bin(x - whole, max_bits=max_bits)


def num_to_fp(x: float, precision: str = "single",
              verbose: bool = False) -> str:
    """Transform a floating into its IEEE 754 standard binary representation."""
    if precision not in ("single", "double"):
        raise ValueError("'precision' must be either 'single' or 'double'.")

    if not isinstance(x, float):
        x = float(x)

    if precision == "single":
        ieee_format = IEEE_754_BITS_SINGLE

    else:
        ieee_format = IEEE_754_BITS_DOUBLE

    def get_exp_val_and_mantissa(bin_x: str) -> t.Tuple[int, str]:
        """Get the exponent value and the mantissa bits.

        The mantissa bits and exponent value are extremely dependent.
        
        To get the corrent exponent value, there must be a single '1'
        bit left to the decimal point, and all other bits of the
        mantissa must be to the right of the decimal point.

        For instance, take the binary number '110.1010'. This number
        must be transformed into '1.101010' (there must be a single '1' bit
        before the decimal point), and the exponent value must be increased
        by 2.

        Another example, take the binary number '0.0011101'. This number
        must be transformed into '1.1101' (there must be a leading '1' bit
        before the decimal point) and the exponent value must be decreased
        by 3.

        The leading '1' bit is necessary simply to omit it in the mantissa
        representation, as it is guaranteed to be the only possibility.
        Therefore, a mantissa bit of information is preserved for more
        precision.
        """
        dec_point_ind = bin_x.find(".")

        exp = (dec_point_ind - 1)

        bin_mantissa = bin_x[:dec_point_ind] + bin_x[(dec_point_ind + 1):]

        first_one_ind = bin_mantissa.find("1")

        exp -= first_one_ind

        bin_mantissa = bin_mantissa[(first_one_ind + 1):]

        if len(bin_mantissa) > ieee_format["mantissa"]:
            bin_mantissa_formated = bin_mantissa[:ieee_format["mantissa"]]

        else:
            bin_mantissa_formated = bin_mantissa + "0" * (
                ieee_format["mantissa"] - len(bin_mantissa))

        return exp, bin_mantissa_formated

    def get_bin_exp(exp: int) -> str:
        """Get binary exponent added to its bias.

        The exponent bias B is calculated as:

        B = 2 ** (number_of_bits_for_exponent - 1) - 1

        Then, the binary value stored in the ``exponent`` is actually
        (exponent + B). Therefore:

        In single precision (32 bits), the exponent bias value is 127.
        In double precision (64 bits), the exponent bias value is 1023.

        This strategy is used to make the floating point manipulation
        easier, as it is not necessary a signal bit to represent both
        positive and negative values (actually, the fist bit is a kind
        of 'flipped' sign bit, as it assumes value 1 for positive exponents
        and 0 for non-positive exponent values.
        """
        # Calculate the exponent bias based on the current precision
        exponent_bias = 2**(ieee_format["exponent"] - 1) - 1

        # Transform the exponent value added to the bias to the binary representation
        bin_exp = dec_to_bin(exp + exponent_bias, max_bits=ieee_format["exponent"])

        # Fill the most significant bits with zeros, if needed
        bin_exp = "0" * (ieee_format["exponent"] - len(bin_exp)) + bin_exp
        return bin_exp

    # The bit sign is '1' for negative values, and '0' otherwise
    bin_sign = "1" if x < 0 else "0"

    # Transform the absolute value of current value to its binary form
    bin_x = dec_to_bin(abs(x), max_bits=ieee_format["mantissa"])

    # Get the exponent value, and also the mantissa bits
    exp, bin_mantissa = get_exp_val_and_mantissa(bin_x)

    # Get the exponent bits
    bin_exp = get_bin_exp(exp)

    if verbose:
        print("SIGN {}:".format(len(bin_sign)), bin_sign)
        print("EXPONENT {}:".format(len(bin_exp)), bin_exp)
        print("MANTISSA {}:".format(len(bin_mantissa)), bin_mantissa)

    # This format order (sign + exp + mantissa) makes FP manipulation easier,
    # for instance in floating point comparisons.
    return bin_sign + bin_exp + bin_mantissa


for i in [1.0, 7.0, 1.375, 13.75, 182.327, 11110.0, 11110.9182]:
    print(i, "-", num_to_fp(i, verbose=True), "\n")
