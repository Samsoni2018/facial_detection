import numpy as np

class Rectangle:
    def __init__(self, c_x, c_y, length_x, length_y):
        self.bottom_left_corner = [c_x, c_y]
        self.l_x = length_x
        self.l_y = length_y

    def get_top_right_corner(self):
        top_right = self.bottom_left_corner + [self.l_x, self.l_y]
        return  top_right

    def area(self):
        a = self.l_x*self.l_y
        return a


class Square(Rectangle):
    def __init__(self, c_x, c_y, length):
        super().__init__(c_x, c_y, length, length)


rect = Rectangle(0, 0, 2, 4)
print(rect.area())


square_1 = Square(0, 0, 2)

print(square_1.area())