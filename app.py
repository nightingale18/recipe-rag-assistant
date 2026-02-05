import streamlit as st
import os
import json
from datetime import datetime

from config import Config
from recipe_manager import SimpleRecipeManager
from recipe_parser import RecipeParser

# Page config
st.set_page_config(page_title="Recipe RAG", page_icon="👨‍🍳", layout="wide")


# Initialize
@st.cache_resource
def get_manager():
    return SimpleRecipeManager(Config())


manager = get_manager()
parser = RecipeParser()

# Sidebar
with st.sidebar:
    st.title("⚡ Recipe RAG")

    if st.button("🔄 Load Recipes", use_container_width=True):
        result = manager.load_recipes()
        st.success(f"Loaded {result['loaded']} recipes")

    # Upload
    st.subheader("📤 Upload")
    uploaded = st.file_uploader(
        "Recipe files", type=["md", "txt"], accept_multiple_files=True
    )

    if uploaded and st.button("Add Recipes", use_container_width=True):
        added = 0
        for file in uploaded:
            content = file.getvalue().decode("utf-8")
            recipe = parser.parse(content, file.name)
            result = manager.store.add_or_update(recipe, "upload")
            if result["success"]:
                added += 1
        st.success(f"Added {added} recipes")

    # Filters
    st.subheader("🔍 Filters")
    cuisine = st.selectbox("Cuisine", ["All"] + manager.config.DEFAULT_CUISINES)
    diet = st.selectbox(
        "Diet", ["All", "Low-carb", "Vegetarian", "Vegan", "Gluten-free"]
    )

    filters = {}
    if cuisine != "All":
        filters["cuisine"] = cuisine
    if diet != "All":
        filters["diet"] = diet

    # Stats
    st.subheader("📊 Stats")
    stats = manager.get_stats()
    st.metric("Total Recipes", stats["total_recipes"])
    st.metric("Recent Changes", stats["recent_changes"])

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📝 Edit", "📈 Analytics"])

# Tab 1: Search
with tab1:
    st.header("Search Recipes")

    query = st.text_input(
        "What would you like to cook?", placeholder="e.g., quick Italian pasta"
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            result = manager.search(
                query, generate_answer=True, filters=filters if filters else None
            )

            if result.get("answer"):
                st.markdown("### 🤖 Answer")
                st.write(result["answer"])

                # Validation badge
                validation = result.get("answer_validation", {})
                conf = validation.get("confidence", "medium")
                color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                st.caption(f"{color} Confidence: {conf}")

            # Results
            if result["results"]:
                st.subheader(f"Found {result['count']} recipes")

                for i, recipe in enumerate(result["results"]):
                    with st.expander(
                        f"{i+1}. {recipe['title']} ({recipe.get('confidence', 'medium')})",
                        expanded=i == 0,
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Cuisine**: {recipe.get('cuisine', '—')}")
                            st.write(f"**Diet**: {recipe.get('diet', '—')}")
                        with col2:
                            st.write(f"**Time**: {recipe.get('time', '—')}")
                            if recipe.get("calories"):
                                st.write(f"**Calories**: {recipe['calories']}")

                        st.write("**Ingredients**:")
                        for ing in recipe.get("ingredients", [])[:5]:
                            st.write(f"- {ing}")

                        if st.button("Edit", key=f"edit_{recipe['title']}"):
                            st.session_state.edit_title = recipe["title"]
                            st.session_state.edit_content = recipe.get("content", "")
                            st.switch_page("📝 Edit")

# Tab 2: Edit
with tab2:
    st.header("Edit Recipe")

    # Get all recipes for selector
    all_recipes = manager.get_all()
    recipe_titles = [r["title"] for r in all_recipes if not r.get("deleted")]

    # Select recipe
    selected = st.selectbox(
        "Select recipe to edit",
        recipe_titles,
        index=(
            recipe_titles.index(st.session_state.get("edit_title", recipe_titles[0]))
            if st.session_state.get("edit_title") in recipe_titles
            else 0
        ),
    )

    if selected:
        # Get current recipe
        recipe = manager.get_recipe(selected)

        if recipe and not recipe.get("deleted"):
            current_content = recipe.get("content", "")

            # Editor
            new_content = st.text_area(
                "Edit recipe (auto-saves when you save)",
                value=current_content,
                height=400,
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", type="primary", use_container_width=True):
                    if new_content != current_content:
                        result = manager.edit_recipe(selected, new_content)
                        if result["success"]:
                            st.success(f"✅ {result['action'].title()}d: {selected}")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('errors', 'Unknown')}")
                    else:
                        st.info("No changes made")

            with col2:
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    if st.checkbox("Confirm deletion"):
                        success = manager.store.delete_recipe(selected)
                        if success:
                            st.success(f"Deleted: {selected}")
                            st.rerun()

            # Preview
            with st.expander("Preview"):
                st.markdown(new_content)

# Tab 3: Analytics
with tab3:
    st.header("Analytics")

    stats = manager.get_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recipes", stats["total_recipes"])
    with col2:
        st.metric("Cuisine Types", len(stats["cuisine_distribution"]))
    with col3:
        st.metric("Real-time", "Active" if stats["monitoring_active"] else "Inactive")

    # Cuisine distribution
    st.subheader("Cuisine Distribution")
    if stats["cuisine_distribution"]:
        import pandas as pd

        df = pd.DataFrame(
            list(stats["cuisine_distribution"].items()), columns=["Cuisine", "Count"]
        )
        st.bar_chart(df.set_index("Cuisine"))

    # Recent changes
    changes = manager.store.get_changes(10)
    if changes:
        st.subheader("Recent Changes")
        for change in reversed(changes):
            st.write(
                f"**{change['title']}** - {change['action']} at {change['time'][11:16]}"
            )

    # Validate test
    st.subheader("Answer Validation Test")

    test_query = st.text_input("Test query", "How to make zucchini noodles?")
    test_answer = st.text_area(
        "Test answer", "Spiralize zucchini and cook with garlic and oil."
    )

    if st.button("Validate Answer"):
        # Get relevant recipes
        search_result = manager.search(test_query, generate_answer=False)

        if search_result["results"]:
            validation = manager.store.validate_answer(
                test_answer, search_result["results"]
            )

            st.write(f"**Validation**: {validation['confidence'].upper()}")
            st.write(f"**Score**: {validation['score']:.1%}")
            st.write(f"**Sources Checked**: {validation['sources_checked']}")

            if validation["valid"]:
                st.success("✅ Answer is well-supported")
            else:
                st.warning("⚠️ Answer may not be fully supported")

# Run
if __name__ == "__main__":
    st.info(
        "Real-time monitoring is active. Edit files in 'recipe_uploads/' to see instant updates!"
    )
