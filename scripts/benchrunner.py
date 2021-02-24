import sys
from timeit import timeit

def output(m):
    print(m)
    sys.stdout.flush()

def input():
    return str(sys.stdin.readline()).strip()

def run_something():
    from torchbenchmark.models.alexnet import Model
    m = Model(cpu, jit=False)
    runtime = timeit(m.eval, number=1)
    return runtime

if __name__ == "__main__":
    try:
        output(f"benchrunner startup: python={sys.executable}")
        token = input()

        # if token.find("run") == 0:
        #     output(f"got a run token! running {token[4:]}")
        # else:
        #     output(f"got some input: {token}, str: {str(token)}")
        while len(token):
            # print(f"token: {token}")
            if token == "exit":
                output("clean exit")
                exit(0)
            elif token[:3] == "run":
                # runtime = run_something()
                import torch
                runtime = 0.1
                output(runtime)
            else:
                output(f"???? {token}")
            token = input()

    except Exception as e:
        output(e)