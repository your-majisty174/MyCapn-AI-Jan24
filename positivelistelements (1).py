list1, list2= ([] for i in range(2))
n=int(input("Enter the number of elements in list :"))
#input the values in list from the user in list1
for i in range(0,n):
    element=int(input())
    list1.append(element) 

#appending the positive values to list2
for i in range(0,n):
    if(list1[i]>0):
        list2.append(list1[i])
print("Positive elements :",list2)  


  
