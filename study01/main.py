def hi(students):
    max = -1
    name = ''
    for student in students:
        ava = sum(student['scores'])/len(student['scores'])
        print(f"{student['name']}的平均分是{ava:.1f}")
        if ava > max:
            max = ava
            name = student['name']
    print(f"平均分最高为{name},其平均分为{max:.1f}")

if __name__ == '__main__':
    students = [
        {"name": "张三", "scores": [80, 90, 70]},
        {"name": "李四", "scores": [95, 88, 92]},
        {"name": "王五", "scores": [60, 75, 68]},
    ]
    hi(students)

    # 要求：
    # 1. 定义一个函数，计算一个学生的平均分
    # 2. 遍历 students，输出格式例如：
    #    张三 的平均分是 80.0
    # 3. 找到平均分最高的学生，并输出其名字和平均分

