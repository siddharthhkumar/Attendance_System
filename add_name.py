import pandas as pd

# Create the students file with your details
df = pd.DataFrame({
    "Id": [101],           # <--- Make sure this matches the ID you typed earlier
    "Name": ["Siddharth"], # <--- Your Name
    "Department": ["IT"]
})

df.to_csv("students.csv", index=False)
print("✅ Name Linked Successfully!")