from typing import List

class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class Link:
    def __init__(self, id, point1, point2):
        self.id = id
        self.point1 = point1
        self.point2 = point2

class Graph:
    def __init__(self, points: List[Point], links: List[Link]):
        self.points = points
        self.links = links
    
    def get_point_by_id(self, point_id):
        for point in self.points:
            if point.id == point_id:
                return point
        return None