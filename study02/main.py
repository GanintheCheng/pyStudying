from student import Student

if __name__ == '__main__':
    students = [Student("张三", [80, 90, 70]), Student("李四", [95, 88, 92]), Student("王五", [60, 75, 68])]
    for student in students:
        if student.is_passed():
            print(f"{student.name}:{student.scores},通过")