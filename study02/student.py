class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def average(self):
        return sum(self.scores) / len(self.scores)

    def is_passed(self):
        if self.average() >= 60:
            return True
        return False