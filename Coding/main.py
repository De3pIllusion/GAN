while True:
    try:
        target = str(input())
        if(len(target)<=8):
            print("NG")
            continue
        flag = [0,0,0,0]
        TRUE = 0
        for i in target:
            if(i.isdigit()):flag[0]=1
            elif(i.islower()):flag[1]=1
            elif(i.isupper()):flag[2]=1
            else: flag[3]=1
            if(flag.count(1)>=3):
                TRUE = 1
                break
        if(TRUE==0):
            print("NG")
            continue
        temp = []
        TRUE = 1
        for i in range(len(target)-2):
            temp2 = target[0:i]+" "+target[i+3:]
            if(target[i]+target[i+1]+target[i+2] in temp2):
                TRUE = 0
                print("NG")
                break
        if(TRUE == 1 ):print("OK")
    except:
        break
