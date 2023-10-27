# box.py
"""Python Essentials: The Standard Library. Auxiliary file (do not modify)."""

from itertools import combinations


def isvalid(roll, remaining):
    """Check to see whether or not a roll is valid. That is, check if there
    exists a combination of the entries of 'remaining' that sum up to 'roll'.

    Parameters:
        roll (int): The value of a dice roll, between 1 and 12 (inclusive).
            This can either be a roll of one die or a roll of two dice
            depending on the remaining values.
        remaining (list): The list of the numbers that still need to be
            removed before the box can be shut.

    Returns:
        True if the roll is valid.
        False if the roll is invalid.
    """
    if roll not in range(1, 13):
        return False
    for i in range(1, len(remaining) + 1):
        if any([sum(combo) == roll for combo in combinations(remaining, i)]):
            return True
    return False


def parse_input(player_input, remaining):
    """Convert a string of numbers into a list of unique integers, if possible.
    Then check that each of those integers is an entry in the other list.

    Parameters:
        player_input (str): A string of integers, separated by spaces.
            The player's choices for which numbers to remove.
        remaining (list): The list of the numbers that still need to be
            removed before the box can be shut.

    Returns:
        A list of the integers if the input was valid.
        An empty list if the input was invalid.
    """
    try:
        choices = [int(i) for i in player_input.split()]
        if len(set(choices)) != len(choices):
            raise ValueError
        if any([number not in remaining for number in choices]):
            raise ValueError
        return choices
    except ValueError:
        return []
