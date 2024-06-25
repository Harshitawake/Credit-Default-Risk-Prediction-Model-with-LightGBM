class MyRange:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self.current

    def __next__(self):
        if self.current < self.end:
            number = self.current
            number += 1
            return number
        else:
            raise StopIteration
        
x=MyRange(1,10)
# print(x.__iter__())
# print(x.__next__())
# mylist=[1,2,3,4,5]
# myiter=iter(mylist)
# print(next(myiter))
# print(next(myiter))
# print(myiter.__next__())
from copy import copy,deepcopy
a=[1,2,3,4,5]
b=a
print(id(a),id(b))
c=deepcopy(a)
print(id(a),id(c))
c.append(8)
print(a,b,c)
import copy

original_list = [[1, 2, 3], [4, 5, 6]]
shallow_copied_list = copy.copy(original_list)
shallow_copied_list.append(99)
print(original_list)  # Output: [[1, 2, 3, 99], [4, 5, 6]]
print(shallow_copied_list)