# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input(
        "Enter a 3-digit number where the first and last "
        "digits differ by 2 or more: "
    )
    step_2 = input(
        "Enter the reverse of the first number, obtained "
        "by reading it backwards: "
    )
    step_3 = input("Enter the positive difference of these numbers: ")
    step_4 = input("Enter the reverse of the previous result: ")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    for i in range(int(max_iters)):
        walk += choice(directions)
    return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """

    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

    # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""

    def uniform(self, outfile, mode="w", case="upper"):
        """Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError."""

    def reverse(self, outfile, mode="w", unit="word"):
        """Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError."""

    def transpose(self, outfile, mode="w"):
        """Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file.
        """

    def __str__(self):
        """Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
