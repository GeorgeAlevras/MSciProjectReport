

for i in range(100):
    for j in range(0, i):
        print('j:', j)
        if j > 5:
            break
    print('i: ', i)