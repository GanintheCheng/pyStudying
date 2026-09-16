import json
import pandas as pd

with open("student.json", "r", encoding="utf-8") as file:
    data = json.load(file)

rows = []

for student in data["students"]:
    scores = [course["score"] for course in student["courses"]]

    row = {
        "age": student["profile"]["age"],
        "city": student["profile"]["city"],
        "course_count": len(scores),
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "passed": student["passed"],
    }

    rows.append(row)

df = pd.DataFrame(rows)

print(df)