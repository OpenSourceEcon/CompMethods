# specs.py
"""Python Essentials: Unit Testing.
<Name>
<Class>
<Date>
"""


def add(a, b):
    """Add two numbers."""
    return a + b


def divide(a, b):
    """Divide two numbers, raising an error if the second number is zero."""
    if b == 0:
        raise ZeroDivisionError("second input cannot be zero")
    return a / b


# Problem 1
def smallest_factor(n):
    """Return the smallest prime factor of the positive integer n."""
    if n == 1:
        return 1
    for i in range(2, int(n**0.5)):
        if n % i == 0:
            return i
    return n


# Problem 2
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {
        "January",
        "March",
        "May",
        "July",
        "August",
        "October",
        "December",
    }:
        return 31
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None


# Problem 3
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == "+":
        return a + b
    elif oper == "-":
        return a - b
    elif oper == "*":
        return a * b
    elif oper == "/":
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    raise ValueError("oper must be one of '+', '/', '-', or '*'")


# Problem 4
class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""

    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        def gcd(a, b):
            while b != 0:
                a, b = b, a % b
            return a

        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        if self.denom != 1:
            return "{}/{}".format(self.numer, self.denom)
        else:
            return str(self.numer)

    def __float__(self):
        return self.numer / self.denom

    def __eq__(self, other):
        if type(other) is Fraction:
            return self.numer == other.numer and self.denom == other.denom
        else:
            return float(self) == other

    def __add__(self, other):
        return Fraction(
            self.numer * other.numer + self.denom * other.denom,
            self.denom * other.denom,
        )

    def __sub__(self, other):
        return Fraction(
            self.numer * other.numer - self.denom * other.denom,
            self.denom * other.denom,
        )

    def __mul__(self, other):
        return Fraction(self.numer * other.numer, self.denom * other.denom)

    def __truediv__(self, other):
        if self.denom * other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Fraction(self.numer * other.denom, self.denom * other.numer)


# Problem 6
def count_sets(cards):
    """Return the number of sets in the provided Set hand.

    Parameters:
        cards (list(str)) a list of twelve cards as 4-bit integers in
        base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
        (int) The number of sets in the hand.
    Raises:
        ValueError: if the list does not contain a valid Set hand, meaning
            - there are not exactly 12 cards,
            - the cards are not all unique,
            - one or more cards does not have exactly 4 digits, or
            - one or more cards has a character other than 0, 1, or 2.
    """
    raise NotImplementedError("Problem 6 Incomplete")


def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.

    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
            For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b,
            and c are either the same or all different for i=1,2,3,4.
        False if a, b, and c do not form a set.
    """
    raise NotImplementedError("Problem 6 Incomplete")
