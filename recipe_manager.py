import os
import glob
import threading
import time
from typing import List, Dict, Optional
from datetime import datetime

from config import Config
from recipe_parser import RecipeParser
from vector_store import RealTimeFAISSStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class SimpleRecipeManager:
    """Simple manager with real-time capabilities"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = pipeline("text-generation", model=self.config.EMBEDDING_MODEL)
        self.config.ensure_dirs()

        self.parser = RecipeParser()
        self.store = RealTimeFAISSStore(self.config)

        # Real-time monitoring
        self._monitor_active = False
        self._monitor_thread = None

        # Start monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start real-time file monitoring"""
        if self._monitor_active:
            return

        self._monitor_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self._monitor_thread.start()
        print("Started real-time monitoring")

    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitor_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    def _monitor_files(self):
        """Monitor recipe directory for changes"""
        last_check = {}

        while self._monitor_active:
            try:
                current_files = {}

                # Check all recipe files
                for ext in ["*.md", "*.txt"]:
                    for filepath in glob.glob(
                        os.path.join(self.config.RECIPE_DIR, ext)
                    ):
                        try:
                            mtime = os.path.getmtime(filepath)
                            current_files[filepath] = mtime

                            # Check if file changed
                            if (
                                filepath not in last_check
                                or last_check[filepath] != mtime
                            ):
                                self._process_file(filepath)
                                last_check[filepath] = mtime

                        except Exception as e:
                            print(f"Error checking {filepath}: {e}")

                # Check for deleted files
                for filepath in list(last_check.keys()):
                    if filepath not in current_files:
                        # File deleted - we could handle this
                        del last_check[filepath]

                time.sleep(self.config.CHECK_INTERVAL)

            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)

    def _process_file(self, filepath: str):
        """Process a recipe file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse
            recipe = self.parser.parse(content, os.path.basename(filepath))

            # Add to store
            result = self.store.add_or_update(recipe, "auto_update")

            if result["success"]:
                print(f"Auto-{result['action']}: {recipe['title']}")
            else:
                print(f"Failed to auto-update: {recipe['title']}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    def load_recipes(self) -> Dict:
        """Load all recipes from directory"""
        recipes = []

        for ext in ["*.md", "*.txt"]:
            for filepath in glob.glob(os.path.join(self.config.RECIPE_DIR, ext)):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    recipe = self.parser.parse(content, os.path.basename(filepath))
                    recipes.append(recipe)

                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

        # Add all to store
        added = 0
        for recipe in recipes:
            result = self.store.add_or_update(recipe, "initial_load")
            if result["success"]:
                added += 1

        return {"success": True, "loaded": len(recipes), "added": added}

    def search(
        self, query: str, generate_answer: bool = True, filters: Dict = None
    ) -> Dict:
        """Search recipes with optional answer generation"""
        search_result = self.store.search(query, filters)

        if generate_answer and search_result["results"]:
            answer = self._generate_answer(query, search_result["results"])
            validation = self.store.validate_answer(answer, search_result["results"])

            return {
                **search_result,
                "answer": answer,
                "answer_validation": validation,
                "timestamp": datetime.now().isoformat(),
            }

        return search_result

    def _generate_answer(self, query: str, results: List[Dict]) -> str:
        """Generate simple answer"""
        if not results:
            return "No recipes found."

        top = results[0]

        parts = [
            f"I found {len(results)} recipes for '{query}'.",
            f"Top match: **{top['title']}**",
        ]

        if top.get("cuisine"):
            parts.append(f"Cuisine: {top['cuisine']}")
        if top.get("diet"):
            parts.append(f"Diet: {top['diet']}")
        if top.get("time"):
            parts.append(f"Time: {top['time']}")

        if top.get("ingredients"):
            parts.append(f"Ingredients include: {', '.join(top['ingredients'][:3])}")

        parts.append(f"Confidence: {top.get('confidence', 'medium')}")

        return " ".join(parts)

    def edit_recipe(self, old_title: str, new_content: str) -> Dict:
        """Edit a recipe"""
        # Validate
        valid, errors = self.parser.validate(new_content)
        if not valid:
            return {"success": False, "errors": errors}

        # Parse
        new_recipe = self.parser.parse(new_content)

        # Update in store
        result = self.store.add_or_update(new_recipe, "user_edit")

        return result

    def _generate_answer(self, query, recipes):
        context = "\n".join([r["title"] for r in recipes[:3]])
        prompt = f"Recipes: {context}\nQuestion: {query}\nAnswer based on recipes:"
        result = self.llm(prompt, max_length=200)
        return result[0]["generated_text"]

    def get_recipe(self, title: str) -> Optional[Dict]:
        """Get recipe by title"""
        return self.store.get_recipe(title)

    def get_all(self) -> List[Dict]:
        """Get all recipes"""
        return self.store.get_all_recipes()

    def get_stats(self) -> Dict:
        """Get statistics"""
        recipes = self.store.get_all_recipes()
        changes = self.store.get_changes(10)

        cuisines = {}
        for recipe in recipes:
            cuisine = recipe.get("cuisine", "Unknown")
            cuisines[cuisine] = cuisines.get(cuisine, 0) + 1

        return {
            "total_recipes": len(recipes),
            "cuisine_distribution": cuisines,
            "recent_changes": len(changes),
            "monitoring_active": self._monitor_active,
        }


class LLMAnswerGenerator:
    def __init__(self):
        # Use a free small LLM
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate_answer(self, query: str, recipes: List[Dict]) -> str:
        context = "\n".join(
            [
                f"Recipe: {r['title']}\nIngredients: {r['ingredients']}"
                for r in recipes[:3]
            ]
        )

        prompt = f"""Use these recipes to answer: {query}
        
        {context}
        
        Answer:"""

        result = self.pipe(prompt, max_new_tokens=200, temperature=0.7)
        return result[0]["generated_text"]
