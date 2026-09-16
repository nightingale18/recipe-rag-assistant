import streamlit as st
import os
import json
import pickle
import numpy as np
import faiss
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import uuid


# ========== CONFIG ==========
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    RECIPE_DIR = "recipe_uploads"
    DATA_DIR = "data"
    TOP_K = 5
    SIM_THRESHOLD = 0.4  # Lower threshold for better matching

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.RECIPE_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)


# ========== RECIPE PARSER ==========
class RecipeParser:
    @staticmethod
    def parse(content: str, filename: str = "") -> Dict:
        """Parse recipe from markdown format"""
        try:
            lines = [line.rstrip() for line in content.strip().split("\n")]

            data = {
                "id": str(uuid.uuid4())[:8],  # Unique ID for each recipe
                "title": "Untitled Recipe",
                "time": "",
                "calories": None,
                "diet": "",
                "cuisine": "",
                "ingredients": [],
                "steps": [],
                "source": filename,
                "content": content,
                "search_text": "",  # Will be filled later
                "created": datetime.now().isoformat(),
            }

            section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check section headers
                if line.lower() == "ingredients:":
                    section = "ingredients"
                    continue
                elif line.lower() == "steps:":
                    section = "steps"
                    continue

                # Parse key-value pairs (before sections)
                if section is None and ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "title":
                        data["title"] = value
                    elif key == "time":
                        data["time"] = value
                    elif key == "calories":
                        try:
                            data["calories"] = int(re.search(r"\d+", value).group())
                        except:
                            data["calories"] = None
                    elif key == "diet":
                        data["diet"] = value
                    elif key == "cuisine":
                        data["cuisine"] = value

                # Parse ingredients
                elif section == "ingredients" and line.startswith("-"):
                    ingredient = line[1:].strip()
                    if ingredient:
                        data["ingredients"].append(ingredient)

                # Parse steps
                elif section == "steps":
                    # Match numbered steps: "1. ..."
                    step_match = re.match(r"^(\d+)\.\s+(.+)$", line)
                    if step_match:
                        data["steps"].append(step_match.group(2))
                    elif line and not line.startswith("-") and ":" not in line:
                        # Also accept non-numbered lines in steps section
                        data["steps"].append(line)

            # Create search text for embeddings
            search_parts = [
                data["title"],
                data.get("cuisine", ""),
                data.get("diet", ""),
                " ".join(data.get("ingredients", [])),
                " ".join(data.get("steps", [])),
            ]
            data["search_text"] = " ".join(filter(None, search_parts))

            return data

        except Exception as e:
            print(f"Error parsing recipe: {e}")
            return {
                "id": str(uuid.uuid4())[:8],
                "title": f"Error Recipe - {filename}",
                "search_text": content[:200],
            }


# ========== VECTOR STORE ==========
class RecipeVectorStore:
    def __init__(self):
        Config.ensure_dirs()
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index_path = os.path.join(Config.DATA_DIR, "recipes_faiss.index")
        self.data_path = os.path.join(Config.DATA_DIR, "recipes_data.pkl")

        # Initialize or load data
        self.recipes = []  # List of recipe dicts
        self.id_to_index = {}  # recipe id -> index in recipes list
        self.index = None

        self._load_or_create()

    def _load_or_create(self):
        """Load existing data or create new index"""
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)

                # Load recipe data
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)
                    self.recipes = data.get("recipes", [])
                    self.id_to_index = data.get("id_to_index", {})

                print(f"✓ Loaded {len(self.recipes)} recipes from storage")

            except Exception as e:
                print(f"Error loading existing data: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        print("Creating new FAISS index...")
        self.index = faiss.IndexFlatL2(Config.EMBEDDING_DIM)
        self.recipes = []
        self.id_to_index = {}

    def _save(self):
        """Save index and data to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save recipe data
            with open(self.data_path, "wb") as f:
                pickle.dump(
                    {"recipes": self.recipes, "id_to_index": self.id_to_index}, f
                )

            print(f"✓ Saved {len(self.recipes)} recipes")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return self.embedder.encode(text).astype("float32")

    def add_recipe(self, recipe: Dict) -> Dict:
        """Add or update a recipe"""
        recipe_id = recipe["id"]

        # Check if recipe already exists
        if recipe_id in self.id_to_index:
            # Update existing recipe
            idx = self.id_to_index[recipe_id]
            self.recipes[idx] = recipe
            print(f"Updated recipe: {recipe['title']}")
        else:
            # Add new recipe
            idx = len(self.recipes)
            self.recipes.append(recipe)
            self.id_to_index[recipe_id] = idx
            print(f"Added new recipe: {recipe['title']}")

        # Create or update embedding
        embedding = self._get_embedding(recipe["search_text"]).reshape(1, -1)

        if idx < self.index.ntotal:
            # Update existing vector (FAISS doesn't support direct update, so we remove and add)
            # For simplicity, we'll just add new - FAISS will have duplicates but that's OK for small scale
            self.index.add(embedding)
        else:
            # Add new vector
            self.index.add(embedding)

        # Save changes
        self._save()

        return {
            "success": True,
            "id": recipe_id,
            "action": "updated" if recipe_id in self.id_to_index else "added",
        }

    def search(self, query: str, filters: Dict = None, k: int = None) -> List[Dict]:
        """Search recipes"""
        if k is None:
            k = Config.TOP_K

        if not self.recipes:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query).reshape(1, -1)

        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding, min(k * 2, len(self.recipes))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.recipes):
                continue

            recipe = self.recipes[idx].copy()

            # Skip if marked as deleted
            if recipe.get("deleted", False):
                continue

            # Apply filters
            if filters:
                if (
                    filters.get("cuisine")
                    and recipe.get("cuisine") != filters["cuisine"]
                ):
                    continue
                if filters.get("diet") and recipe.get("diet") != filters["diet"]:
                    continue
                if filters.get("max_time"):
                    # Simple time filter (extract minutes)
                    try:
                        time_str = recipe.get("time", "")
                        mins = int(re.search(r"\d+", time_str).group())
                        max_mins = int(re.search(r"\d+", filters["max_time"]).group())
                        if mins > max_mins:
                            continue
                    except:
                        pass

            # Calculate similarity score
            similarity = 1.0 / (1.0 + dist)

            if similarity >= Config.SIM_THRESHOLD:
                recipe["similarity"] = similarity
                recipe["confidence"] = (
                    "high"
                    if similarity > 0.7
                    else "medium" if similarity > 0.5 else "low"
                )
                results.append(recipe)

            if len(results) >= k:
                break

        return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)

    def get_all_recipes(self) -> List[Dict]:
        """Get all non-deleted recipes"""
        return [r for r in self.recipes if not r.get("deleted", False)]

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict]:
        """Get recipe by ID"""
        idx = self.id_to_index.get(recipe_id)
        if idx is not None and not self.recipes[idx].get("deleted", False):
            return self.recipes[idx]
        return None

    def get_recipe_by_title(self, title: str) -> Optional[Dict]:
        """Get recipe by title"""
        for recipe in self.recipes:
            if not recipe.get("deleted", False) and recipe.get("title") == title:
                return recipe
        return None

    def delete_recipe(self, recipe_id: str) -> bool:
        """Soft delete a recipe"""
        idx = self.id_to_index.get(recipe_id)
        if idx is not None:
            self.recipes[idx]["deleted"] = True
            self._save()
            return True
        return False


# ========== RECIPE MANAGER ==========
class RecipeManager:
    def __init__(self):
        self.parser = RecipeParser()
        self.store = RecipeVectorStore()
        self._load_existing_recipes()

    def _load_existing_recipes(self):
        """Load recipes from the upload directory"""
        if not os.path.exists(Config.RECIPE_DIR):
            return

        loaded = 0
        for filename in os.listdir(Config.RECIPE_DIR):
            if filename.endswith((".md", ".txt")):
                try:
                    filepath = os.path.join(Config.RECIPE_DIR, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    recipe = self.parser.parse(content, filename)
                    self.store.add_recipe(recipe)
                    loaded += 1

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        if loaded > 0:
            st.success(f"Loaded {loaded} recipes from {Config.RECIPE_DIR}")

    def search_recipes(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search recipes with given query and filters"""
        return self.store.search(query, filters)

    def add_recipe_from_content(self, content: str, filename: str = "") -> Dict:
        """Add a recipe from text content"""
        recipe = self.parser.parse(content, filename)
        return self.store.add_recipe(recipe)

    def update_recipe(self, recipe_id: str, new_content: str) -> Dict:
        """Update an existing recipe"""
        recipe = self.parser.parse(new_content)
        recipe["id"] = recipe_id  # Keep the same ID
        return self.store.add_recipe(recipe)

    def get_all_recipes(self) -> List[Dict]:
        """Get all recipes"""
        return self.store.get_all_recipes()

    def get_recipe_by_title(self, title: str) -> Optional[Dict]:
        """Get recipe by title"""
        return self.store.get_recipe_by_title(title)

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict]:
        """Get recipe by ID"""
        return self.store.get_recipe_by_id(recipe_id)

    def delete_recipe(self, title: str) -> bool:
        """Delete recipe by title"""
        recipe = self.store.get_recipe_by_title(title)
        if recipe:
            return self.store.delete_recipe(recipe["id"])
        return False


# ========== STREAMLIT APP ==========
def main():
    st.set_page_config(
        page_title="Recipe RAG Assistant", page_icon="👨‍🍳", layout="wide"
    )

    st.title("👨‍🍳 Recipe RAG Assistant")
    st.markdown("Search, edit, and manage your recipes with AI-powered search")

    # Initialize session state
    if "recipe_manager" not in st.session_state:
        st.session_state.recipe_manager = RecipeManager()

    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    manager = st.session_state.recipe_manager

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Recipes")

        uploaded_files = st.file_uploader(
            "Choose recipe files",
            type=["md", "txt"],
            accept_multiple_files=True,
            help="Upload recipes in the correct format",
        )

        if uploaded_files:
            for file in uploaded_files:
                content = file.getvalue().decode("utf-8")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        f"Add {file.name}", key=f"add_{file.name}_{uuid.uuid4()}"
                    ):
                        result = manager.add_recipe_from_content(content, file.name)
                        if result["success"]:
                            st.success(f"✓ Added: {file.name}")
                            st.rerun()

                with col2:
                    if st.button(
                        f"Preview {file.name}",
                        key=f"preview_{file.name}_{uuid.uuid4()}",
                    ):
                        with st.expander(f"Preview {file.name}"):
                            st.text(content[:500])

        st.header("🔍 Search Filters")

        # Get unique cuisines from recipes
        all_recipes = manager.get_all_recipes()
        cuisines = sorted(
            set(r.get("cuisine", "") for r in all_recipes if r.get("cuisine"))
        )
        diets = sorted(set(r.get("diet", "") for r in all_recipes if r.get("diet")))

        selected_cuisine = st.selectbox("Cuisine", ["All"] + cuisines)
        selected_diet = st.selectbox("Diet", ["All"] + diets)

        filters = {}
        if selected_cuisine != "All":
            filters["cuisine"] = selected_cuisine
        if selected_diet != "All":
            filters["diet"] = selected_diet

        st.header("📊 Stats")
        st.metric("Total Recipes", len(all_recipes))

        if all_recipes:
            st.caption(f"Unique cuisines: {len(cuisines)}")
            st.caption(f"Unique diets: {len(diets)}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📝 Edit", "📚 Browse All"])

    # Tab 1: Search
    with tab1:
        st.header("Search Recipes")

        # Search input
        search_query = st.text_input(
            "What would you like to cook?",
            value=st.session_state.search_query,
            placeholder="Try: 'Italian', 'low-carb', 'zucchini', 'quick dinner'",
            key="search_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            search_button = st.button(
                "🔍 Search", type="primary", use_container_width=True
            )
        with col2:
            if st.button("Clear Results", use_container_width=True):
                st.session_state.search_query = ""
                st.session_state.selected_recipe = None
                st.rerun()

        if search_button and search_query:
            st.session_state.search_query = search_query

            with st.spinner("Searching recipes..."):
                results = manager.search_recipes(
                    search_query, filters if filters else None
                )

                if results:
                    st.success(f"Found {len(results)} recipes for '{search_query}'")

                    # Display results
                    for i, recipe in enumerate(results):
                        # Create a unique key for each button
                        unique_key = f"{recipe['id']}_{i}_{uuid.uuid4()}"

                        with st.expander(
                            f"{i+1}. {recipe['title']} ({recipe.get('confidence', 'medium')} match)",
                            expanded=i == 0,
                        ):
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                if recipe.get("cuisine"):
                                    st.write(f"**Cuisine**: {recipe['cuisine']}")
                                if recipe.get("diet"):
                                    st.write(f"**Diet**: {recipe['diet']}")
                            with col_info2:
                                if recipe.get("time"):
                                    st.write(f"**Time**: {recipe['time']}")
                                if recipe.get("calories"):
                                    st.write(f"**Calories**: {recipe['calories']}")

                            st.write("**Ingredients**:")
                            for ingredient in recipe.get("ingredients", [])[:5]:
                                st.write(f"- {ingredient}")

                            # Action buttons with unique keys
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button(f"✏️ Edit", key=f"edit_btn_{unique_key}"):
                                    st.session_state.selected_recipe = recipe["title"]
                                    st.switch_page("📝 Edit")
                            with col_btn2:
                                if st.button(
                                    f"📋 View Full", key=f"view_btn_{unique_key}"
                                ):
                                    st.subheader(f"Full Recipe: {recipe['title']}")
                                    st.text(recipe.get("content", ""))
                else:
                    st.warning(f"No recipes found for '{search_query}'")

                    # Show suggestions
                    st.info("Try searching for:")
                    st.write("- Cuisine type (Italian, Mexican, etc.)")
                    st.write("- Diet type (low-carb, vegetarian, etc.)")
                    st.write("- Ingredients (zucchini, pasta, chicken, etc.)")
                    st.write("- Time (quick, 30 minutes, etc.)")

        # Show selected recipe details if any
        if st.session_state.selected_recipe:
            recipe = manager.get_recipe_by_title(st.session_state.selected_recipe)
            if recipe:
                st.divider()
                st.subheader(f"📄 {recipe['title']}")

                col_view1, col_view2 = st.columns(2)
                with col_view1:
                    st.write(f"**Source**: {recipe.get('source', 'Unknown')}")
                    st.write(
                        f"**Last Updated**: {recipe.get('created', 'Unknown')[:10]}"
                    )
                with col_view2:
                    if st.button(
                        "Edit This Recipe", key=f"edit_selected_{uuid.uuid4()}"
                    ):
                        st.switch_page("📝 Edit")

    # Tab 2: Edit
    with tab2:
        st.header("Edit Recipe")

        # Get all recipes for dropdown
        all_recipes = manager.get_all_recipes()

        if not all_recipes:
            st.info("No recipes found. Upload some recipes first!")
            st.markdown(
                """
            **Expected format:**
            ```
            Title: Recipe Name
            Time: 30 minutes
            Calories: 350
            Diet: Vegetarian
            Cuisine: Italian
            
            Ingredients:
            - ingredient 1
            - ingredient 2
            
            Steps:
            1. Step one
            2. Step two
            ```
            """
            )
        else:
            # Recipe selector
            recipe_titles = [r["title"] for r in all_recipes]

            # Determine which recipe to select
            default_index = 0
            if (
                st.session_state.selected_recipe
                and st.session_state.selected_recipe in recipe_titles
            ):
                default_index = recipe_titles.index(st.session_state.selected_recipe)

            selected_title = st.selectbox(
                "Select recipe to edit:", recipe_titles, index=default_index
            )

            if selected_title:
                recipe = manager.get_recipe_by_title(selected_title)

                if recipe:
                    # Editor
                    current_content = recipe.get("content", "")

                    edited_content = st.text_area(
                        "Edit recipe content:",
                        value=current_content,
                        height=400,
                        key=f"editor_{recipe['id']}",
                    )

                    col_save, col_delete, col_preview = st.columns(3)

                    with col_save:
                        if st.button(
                            "💾 Save Changes", type="primary", use_container_width=True
                        ):
                            if edited_content != current_content:
                                result = manager.update_recipe(
                                    recipe["id"], edited_content
                                )
                                if result["success"]:
                                    st.success(f"✓ Recipe updated!")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("Failed to save changes")
                            else:
                                st.info("No changes detected")

                    with col_delete:
                        if st.button(
                            "🗑️ Delete Recipe",
                            type="secondary",
                            use_container_width=True,
                        ):
                            if st.checkbox(
                                "Confirm deletion", key=f"confirm_del_{recipe['id']}"
                            ):
                                success = manager.delete_recipe(selected_title)
                                if success:
                                    st.success(f"Deleted: {selected_title}")
                                    st.session_state.selected_recipe = None
                                    st.rerun()

                    with col_preview:
                        if st.button("👁️ Preview", use_container_width=True):
                            with st.expander("Preview", expanded=True):
                                st.text(edited_content)

                    # Show current metadata
                    with st.expander("Current Recipe Info", expanded=False):
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.write(f"**ID**: {recipe.get('id', 'N/A')}")
                            st.write(f"**Source File**: {recipe.get('source', 'N/A')}")
                        with col_meta2:
                            st.write(
                                f"**Created**: {recipe.get('created', 'N/A')[:19]}"
                            )
                            st.write(
                                f"**Ingredients Count**: {len(recipe.get('ingredients', []))}"
                            )

    # Tab 3: Browse All
    with tab3:
        st.header("Browse All Recipes")

        all_recipes = manager.get_all_recipes()

        if not all_recipes:
            st.info("No recipes available. Upload some recipes first!")
        else:
            st.subheader(f"All Recipes ({len(all_recipes)})")

            # Search within browse
            browse_filter = st.text_input(
                "Filter recipes by name:", placeholder="Type to filter..."
            )

            # Display recipes
            for i, recipe in enumerate(all_recipes):
                # Apply filter
                if (
                    browse_filter
                    and browse_filter.lower() not in recipe["title"].lower()
                ):
                    continue

                # Use a unique key for each row
                row_key = f"row_{recipe['id']}_{i}"

                with st.container():
                    col_title, col_actions = st.columns([3, 1])

                    with col_title:
                        st.markdown(f"**{recipe['title']}**")
                        if recipe.get("cuisine"):
                            st.caption(f"🍝 {recipe['cuisine']}")
                        if recipe.get("diet"):
                            st.caption(f"🥗 {recipe['diet']}")

                    with col_actions:
                        # Use unique keys for buttons
                        edit_key = f"browse_edit_{recipe['id']}_{i}"
                        view_key = f"browse_view_{recipe['id']}_{i}"

                        if st.button("✏️ Edit", key=edit_key):
                            st.session_state.selected_recipe = recipe["title"]
                            st.switch_page("📝 Edit")

                    # Show recipe preview on click
                    if st.button(f"📋 View {recipe['title']}", key=view_key):
                        st.markdown(f"**Full Recipe: {recipe['title']}**")
                        st.text(recipe.get("content", "No content"))

                    st.divider()


if __name__ == "__main__":
    main()
