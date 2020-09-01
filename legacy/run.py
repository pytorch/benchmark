from subprocess import call
import os


def get_docker_run_cmd(image_name):
    return ["sudo", "docker", "run", "--rm", "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined", "-v", os.getcwd() + ":/mnt/localdrive",
            "--cpuset-cpus=0-3", "-t", "--user=jenkins", image_name]

if __name__ == "__main__":
    call(get_docker_run_cmd("tmp-utcnpjpdbsorhktnnnttixmkvlyxirwl") + ["/bin/bash", "/mnt/localdrive/python/run.sh"])
