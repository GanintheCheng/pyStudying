import numpy as np

def getAva(student):
    return student.mean()

if __name__ == '__main__':
    # 二维数组：3 个学生，每人 3 门成绩
    score_matrix = np.array([
        [80, 90, 70],
        [95, 88, 92],
        [60, 75, 68],
    ])
    for index,score in enumerate(score_matrix):
        print(f"第{index+1}位同学的平均分是{getAva(score):.1f}")

    print(f"最高分学生下标为{score_matrix.mean(axis=1).argmax()}")
    print(np.minimum(score_matrix+5, 100))


    # 练习：使用上面的
    # score_matrix完成：
    # 1.输出每位学生的平均分。
    # 2.输出平均分最高的学生下标（预期为1）。
    # 3.给所有成绩加5分，但最高不得超过100。
    # 4.输出加分后的矩阵。