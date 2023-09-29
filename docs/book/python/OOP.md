(Chap_OOP)=


# Object Oriented Programming

ACME materials link...


# Exercises

1. Define a class called `Specifications` with an attribute that is the rate of time preference, $\beta$.  Create instances of this class called `p` for $\beta=0.96$ and $\beta=0.99$.
2. Update the `Specifications` class so that allows one to specify the value of $\beta$ upon initialization of the class and checks that $\beta$ is between 0 and 1.
3. Modify the `Specifications` class so that it has a method that prints the value of $\beta$.
4. Change the input of $\beta$ to the class so that it is input at an annual rate.  Allow another attribute of the class called `S` that is the number of periods in an economic agent's life.  Include a method in the `Specifications` class that adjusts the value  the value of $\beta$ to represent the discount rate applied per model period, which will be equivalent to `S/80` years.
5. Add a method to the `Specifications` class that allows one to update the values of the class attributes `S` and `beta_annual` by providing a dictionary of the form `{"S": 40, "beta_annual": 0.8}`.  Ensure that when the instance is updated, the new `beta` attribute is consistent with the new `S` and `beta_annual`.