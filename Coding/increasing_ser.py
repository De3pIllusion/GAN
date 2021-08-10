
a = "123456sdfsafsdf"
for i in a:
    if(not i.isdigit()):
        a=a.replace(i,"")
max = 1
cur = 1

for i in range(len(a)-1):
    if(a[i]<=a[i+1]):
        cur+=1
    else:
        max = (lambda x,y: (x>=y)*x + (x<y)*y)(cur,max)
        cur = 1
    max = (lambda x, y: (x >= y) * x + (x < y) * y)(cur, max)
print(max)