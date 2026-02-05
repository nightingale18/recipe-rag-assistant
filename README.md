# 🍳 Recipe RAG Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Real-time recipe search and management with AI-powered semantic search
and answer validation**

------------------------------------------------------------------------

## ✨ Features

### 🔍 Smart Recipe Search

-   Semantic search using sentence transformers\
-   Filter by cuisine, diet, time, and calories\
-   Real-time results as you type\
-   Answer validation against source recipes

### ⚡ Real-time Document Handling

-   Auto-detects changes in recipe files\
-   Instant vector store updates\
-   Live editing with auto-save\
-   Version tracking and change history

### 🤖 AI-Powered Answers

-   LLM-generated answers based on retrieved recipes\
-   Answer validation against multiple sources\
-   Confidence scoring for responses\
-   Source citation and attribution

### 📊 Recipe Management

-   Upload markdown recipe files\
-   In-app editing with preview\
-   Batch operations\
-   Export recipes in multiple formats

------------------------------------------------------------------------

## 🚀 Quick Start

### Prerequisites

-   Python 3.8+\
-   pip package manager

------------------------------------------------------------------------

### Installation

``` bash
git clone https://github.com/yourusername/recipe-rag-assistant.git
cd recipe-rag-assistant
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the app:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📝 Recipe Format

Recipes should be written in Markdown:

``` markdown
Title: Zucchini Noodles
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
3. Add noodles and parmesan.
```

------------------------------------------------------------------------

## 🎯 Usage Examples

### 1️⃣ Upload Recipes

Place `.md` files in:

    recipe_uploads/

Or use the upload interface in the app.\
Files are indexed automatically in real time.

------------------------------------------------------------------------

### 2️⃣ Search Recipes

Example queries:

    Search: "quick low-carb Italian dinner"
    → Finds: Zucchini Noodles, Chicken Piccata, etc.

    Search: "recipes with basil and tomatoes"
    → Finds: Caprese Salad, Tomato Basil Pasta, etc.

------------------------------------------------------------------------

### 3️⃣ Edit Recipes

-   Click **Edit** on any recipe\
-   Modify content in the editor\
-   Changes are saved and re-indexed automatically\
-   View change history and rollback if needed

------------------------------------------------------------------------

### 4️⃣ Get AI Answers

Example:

    Q: How do I make a vegetarian pasta quickly?

    A: Based on the recipes, try Pasta Primavera (20 min)
    with zucchini, bell peppers, and olive oil...

------------------------------------------------------------------------

## 🛠️ Configuration

Edit `config.py`:

``` python
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RECIPE_DIR = "recipe_uploads"
    TOP_K_RESULTS = 10
    CHECK_INTERVAL = 2
```

------------------------------------------------------------------------

## 📊 Supported Cuisines

-   Italian\
-   European\
-   Fusion\
-   American\
-   Asian\
-   Mexican\
-   Mediterranean\
-   Middle Eastern\
-   Nordic\
-   Indian

------------------------------------------------------------------------

## 🔧 Advanced Features

### Real-time Monitoring

The system monitors your `recipe_uploads/` directory and updates the
search index automatically when files change.

### Answer Validation

Generated answers are validated against source recipes and include a
confidence score.

### Batch Operations

-   Bulk upload\
-   Batch edit\
-   Export as JSON or Markdown

------------------------------------------------------------------------

## 📈 Statistics & Analytics

-   Recipe count by cuisine\
-   Update frequency tracking\
-   Search performance metrics

------------------------------------------------------------------------

## 🧪 Testing

Add a test recipe:

``` bash
echo "Title: Test Recipe
Time: 10 minutes
Cuisine: Italian

Ingredients:
- ingredient1
- ingredient2

Steps:
1. Step one.
2. Step two." > recipe_uploads/test.md
```

Edit the file and verify real-time updates in the app.

Try searches like:

    Italian recipes
    quick dinner

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License --- see the `LICENSE`
file for details.

------------------------------------------------------------------------

## 🙏 Acknowledgments

-   Sentence Transformers for embeddings\
-   FAISS for vector search\
-   Streamlit for UI\
-   Hugging Face for transformer models
