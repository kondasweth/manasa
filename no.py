def print_sum_of_current_and_previous():
    previous_number = 0
    for i in range(1, 11):
        current_number = i
        sum_of_previous_and_current = previous_number + current_number
        print(f"Current number: {current_number}, Previous number: {previous_number}, Sum: {sum_of_previous_and_current}")
        previous_number = current_number

print_sum_of_current_and_previous()
