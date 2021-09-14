### parse results of genetic
with open('min5.txt', 'r') as f:
    lines = f.readlines()

    # for i in range(10):
    #     print(lines[-i])
    
    for idx, line in enumerate(lines):
        if 'Cost for this individual: (451724' in line:
            print(idx)
    
            # print(line)

            print(lines[idx-1])
