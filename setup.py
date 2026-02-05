#!/usr/bin/env python3
import os
import sys

print("⚡ Setting up Recipe RAG System")

# Create directories
os.makedirs("recipe_uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Create sample recipe
sample = """Title: Zucchini Noodles
Time: 15 minutes
Calories: 220
Diet: Low-carb
Cuisine: Italian

Ingredients:
- zucchini
- garlic
- olive oil
- parmesan

Steps:
1. Spiralize zucchini.
2. Fry garlic in oil.
3. Add noodles and parmesan."""

with open("recipe_uploads/zucchini_noodles.md", "w") as f:
    f.write(sample)

print("✅ Created sample recipe: recipe_uploads/zucchini_noodles.md")
print("✅ Directories created")
print("\n📦 Install dependencies:")
print("   pip install -r requirements.txt")
print("\n🚀 Run the app:")
print("   streamlit run app.py")