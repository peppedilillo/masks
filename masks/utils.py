def is_prime(n: int) -> bool:
    """
    A simple function to check if an integer is a prime.
    Not fit for large numbers, but will work well for checking masks dimensions.
    """
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
