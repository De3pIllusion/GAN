
import copy

def dfs(data ,temp ,target):
    temp.append(data[0])
    i f(len(temp )> =2):
        a = copy.deepcopy(temp)
        target.append(a)
    i f(len(data ) >1):
        dfs(data[1:] ,temp ,target)


if __name__ == "__main__":
    a = [2 ,3 ,4 ,5 ,6 ,7 ,8]
    b = 3
    target = []
    for i in range(len(a)):
        temp = []
        dfs(a[i:] ,temp ,target)
    for i in target:
        for j in range(len(i)):
            i f( j= =len(i ) -1):
                print(str(i[j]))
            else:
                print(str(i[j] ) +"," ,end="")



