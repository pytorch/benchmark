from distutils.sysconfig import get_python_lib

def helper():
    torchdynamo_backend_path = get_python_lib()+"/torchdynamo/optimizations/backends.py"
    file_data = ""
    with open(torchdynamo_backend_path, 'r', encoding="utf-8") as f:
        for line in f:
            if "dev," in line:
                line = line.replace("dev,", "dev.type,")
            file_data += line
    with open(torchdynamo_backend_path, 'w', encoding="utf-8") as f:
        f.write(file_data)

if __name__ == "__main__":
    found = False
    try:
        import torchdynamo
        found = True
    except ImportError:
        found = False
    if found==True:
       helper()
