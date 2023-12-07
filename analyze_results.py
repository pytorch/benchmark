

textfile_name = "log.log"
file1 = open(textfile_name, 'r')
lines = file1.readlines()
baseline_technique_number = 3

results = {}
models = []
technique="err"
cur_model_name = "err"
prev_line = "err"
for line in lines:
    if "start" in line and line[:6] == "start ":
        start = line.find("start ")
        technique = line[start+6:-1]
        results[technique]={}
        cur_model_name = "err"
    elif "cuda eval" in line:
        assert technique!="err"
        end = line[11:].find(" ")
        cur_model_name = line[11:11+end]
        if cur_model_name not in models:
            models.append(cur_model_name)
    elif "running benchmark: 100%" in prev_line:
        try:
            result = float(line[0:line.find("x")])
            assert cur_model_name!="err"
            results[technique][cur_model_name]=result
            cur_model_name = "err"
        except:
            if not "running benchmark" in line:
                print("err parsing line: \n", line)
    prev_line = line

techniques = [x for x in results.keys()]
baseline_technique = techniques[baseline_technique_number]

max_model_len = max([len(x) for x in models])+1
max_technique_len = max([len(x) for x in techniques])+1
num_techniques = len(techniques)
print(max_model_len, max_technique_len)

print("|"+"-"*max_model_len+"|"+("-"*max_model_len+"|")*num_techniques)
print(f"|{f'model'.ljust(max_model_len)}|{''.join([f'{x.ljust(max_model_len)}|' for x in techniques])}")
print("|"+"-"*max_model_len+"|"+("-"*max_model_len+"|")*num_techniques)
working = {}
faster = {}
for model in models:
    out = f"|{model.ljust(max_model_len)}|"
    for technique in techniques:
        try:
            result = f"{results[technique][model]/results[baseline_technique][model]:0.4}"
        except:
            result = "err"

        working[technique]=working.get(technique, 0)+(result!="err")
        faster[technique]=faster.get(technique, 0)+(result!="err" and float(result)>=1.0)
        out+=f"{result.ljust(max_model_len)}|"
    print(out)
print("|"+"-"*max_model_len+"|"+("-"*max_model_len+"|")*num_techniques)
print(f"|{f'TOTAL COVERAGE'.ljust(max_model_len)}|{''.join([f'{working[x]/working[baseline_technique]*100:0.5}%'.ljust(max_model_len)+'|' for x in techniques])}")
print(f"|{f'TOTAL FASTER'.ljust(max_model_len)}|{''.join([f'{faster[x]/faster[baseline_technique]*100:0.5}%'.ljust(max_model_len)+'|' for x in techniques])}")

print("|"+"-"*max_model_len+"|"+("-"*max_model_len+"|")*num_techniques)
