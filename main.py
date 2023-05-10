# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def evalRPN(tokens):
    """
    :type tokens: List[str]
    :rtype: int  aaa a a s yyy
    """
    stack_ = []
    for t in tokens:
        print(t)
        if t in ['+', '-', '*', '/']:
            print(eval('6/-132'))
            print(stack_)
            tmp_ans = int(eval(str(str(stack_[-2]) + t + str(stack_[-1]))))
            stack_ = stack_[:-2]
            stack_.append(tmp_ans)
        else:
            stack_.append(t)

    return stack_[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ans = evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])
    print(ans)
    print_hi('hhhhh')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
