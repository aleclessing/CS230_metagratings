
array = []
with open('logs/txt_logs/1_log.txt') as f:
    lines = f.readlines()

for line in lines:
    line.replace('train ep ', '')
    print(line)

# print(lines)
