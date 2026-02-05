import faiss
import numpy as np
import pickle
import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class RealTimeFAISSStore:
    """FAISS vector store with real-time updates"""

    def __init__(self, config):
        self.config = config
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Load or create FAISS index
        self.index_path = config.VECTOR_STORE_PATH
        self.data_path = config.RECIPES_DATA_PATH

        # Recipe storage: list of dicts
        self.recipes = []
        # Mapping: title -> index in recipes list
        self.title_to_idx = {}
        # Track changes
        self.changes = []

        self._load()

    def _load(self):
        """Load existing data"""
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.data_path, "rb") as f:
                    saved = pickle.load(f)
                    self.recipes = saved.get("recipes", [])
                    self.title_to_idx = saved.get("title_to_idx", {})
                    self.changes = saved.get("changes", [])
                print(f"Loaded {len(self.recipes)} recipes")
            except:
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatL2(self.config.EMBEDDING_DIM)
        self.recipes = []
        self.title_to_idx = {}
        self.changes = []
        print("Created new index")

    def _save(self):
        """Save index and data"""
        if not self.config.AUTO_SAVE:
            return

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save recipe data
        with open(self.data_path, "wb") as f:
            pickle.dump(
                {
                    "recipes": self.recipes,
                    "title_to_idx": self.title_to_idx,
                    "changes": self.changes[-100:],  # Keep last 100 changes
                },
                f,
            )

    def _embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text"""
        return self.embedder.encode(text).astype("float32")

    def _create_search_text(self, recipe: Dict) -> str:
        """Create text for embedding/search"""
        return f"""
        {recipe.get('title', '')}
        {recipe.get('cuisine', '')}
        {recipe.get('diet', '')}
        {' '.join(recipe.get('ingredients', []))}
        {' '.join(recipe.get('steps', []))}
        """

    def _hash_recipe(self, recipe: Dict) -> str:
        """Create hash for change detection"""
        content = json.dumps(recipe, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def add_or_update(self, recipe: Dict, user: str = "user") -> Dict:
        """Add or update recipe in real-time"""
        title = recipe.get("title", "Unknown")
        content_hash = self._hash_recipe(recipe)

        # Check if recipe exists
        if title in self.title_to_idx:
            idx = self.title_to_idx[title]
            old_recipe = self.recipes[idx]
            old_hash = self._hash_recipe(old_recipe)

            if old_hash == content_hash:
                return {"success": True, "action": "no_change", "title": title}

            # Update existing recipe
            self.recipes[idx] = recipe

            # Update embedding
            search_text = self._create_search_text(recipe)
            new_embedding = self._embed_text(search_text).reshape(1, -1)

            # FAISS doesn't support updates, so we need to rebuild or mark as stale
            # For simplicity, we'll remove old and add new
            self.index.remove_ids(np.array([idx], dtype=np.int64))
            self.index.add(new_embedding)

            # Log change
            self.changes.append(
                {
                    "title": title,
                    "action": "update",
                    "old_hash": old_hash[:8],
                    "new_hash": content_hash[:8],
                    "user": user,
                    "time": datetime.now().isoformat(),
                }
            )

            self._save()
            return {"success": True, "action": "updated", "title": title}

        else:
            # Add new recipe
            idx = len(self.recipes)
            self.recipes.append(recipe)
            self.title_to_idx[title] = idx

            # Add embedding
            search_text = self._create_search_text(recipe)
            embedding = self._embed_text(search_text).reshape(1, -1)
            self.index.add(embedding)

            # Log change
            self.changes.append(
                {
                    "title": title,
                    "action": "add",
                    "user": user,
                    "time": datetime.now().isoformat(),
                }
            )

            self._save()
            return {"success": True, "action": "added", "title": title}

    def search(self, query: str, filters: Dict = None, k: int = None) -> Dict:
        """Search recipes with validation"""
        if k is None:
            k = self.config.TOP_K

        # Generate query embedding
        query_embedding = self._embed_text(query).reshape(1, -1)

        # Search
        distances, indices = self.index.search(
            query_embedding, min(k * 2, len(self.recipes))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.recipes):
                continue

            recipe = self.recipes[idx]
            similarity = 1.0 / (1.0 + dist)

            # Apply filters
            if filters:
                if (
                    filters.get("cuisine")
                    and recipe.get("cuisine") != filters["cuisine"]
                ):
                    continue
                if filters.get("diet") and recipe.get("diet") != filters["diet"]:
                    continue

            if similarity >= self.config.SIM_THRESHOLD:
                # Calculate validation score
                validation_score = self._validate_match(query, recipe)

                results.append(
                    {
                        **recipe,
                        "similarity": similarity,
                        "validation_score": validation_score,
                        "confidence": self._get_confidence_level(
                            similarity, validation_score
                        ),
                    }
                )

                if len(results) >= k:
                    break

        # Generate validation report
        validation = self._generate_validation_report(query, results)

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "validation": validation,
        }

    def _validate_match(self, query: str, recipe: Dict) -> float:
        """Validate how well recipe matches query"""
        score = 0.0
        query_lower = query.lower()

        # Check title
        if recipe.get("title", "").lower() in query_lower:
            score += 0.3

        # Check ingredients
        ingredients_text = " ".join(recipe.get("ingredients", [])).lower()
        for word in query_lower.split():
            if len(word) > 3 and word in ingredients_text:
                score += 0.1

        # Check cuisine/diet
        if recipe.get("cuisine", "").lower() in query_lower:
            score += 0.2
        if recipe.get("diet", "").lower() in query_lower:
            score += 0.2

        return min(1.0, score)

    def _get_confidence_level(self, similarity: float, validation: float) -> str:
        """Get confidence level"""
        combined = (similarity + validation) / 2

        if combined > 0.7:
            return "high"
        elif combined > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_validation_report(self, query: str, results: List[Dict]) -> Dict:
        """Generate validation report"""
        if not results:
            return {"confidence": "low", "message": "No results found"}

        avg_similarity = sum(r["similarity"] for r in results) / len(results)
        avg_validation = sum(r["validation_score"] for r in results) / len(results)

        confidence = self._get_confidence_level(avg_similarity, avg_validation)

        # Check for issues
        issues = []
        if avg_similarity < 0.3:
            issues.append("Low semantic similarity")
        if avg_validation < 0.2:
            issues.append("Limited keyword matching")

        return {
            "confidence": confidence,
            "avg_similarity": avg_similarity,
            "avg_validation": avg_validation,
            "issues": issues,
        }

    def validate_answer(self, answer: str, sources: List[Dict]) -> Dict:
        """Validate answer against sources"""
        if not sources:
            return {"valid": False, "score": 0, "confidence": "low"}

        # Extract facts from answer
        facts = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

        validation_scores = []
        for source in sources:
            source_text = json.dumps(source, ensure_ascii=False).lower()
            matches = 0

            for fact in facts:
                fact_lower = fact.lower()
                # Simple keyword matching
                fact_words = set(w for w in fact_lower.split() if len(w) > 3)
                source_words = set(source_text.split())

                if fact_words:
                    match_ratio = len(fact_words & source_words) / len(fact_words)
                    if match_ratio > 0.3:
                        matches += 1

            score = matches / len(facts) if facts else 0
            validation_scores.append(score)

        avg_score = sum(validation_scores) / len(validation_scores)

        if avg_score > 0.6:
            confidence = "high"
        elif avg_score > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "valid": avg_score > 0.3,
            "score": avg_score,
            "confidence": confidence,
            "sources_checked": len(sources),
        }

    def get_all_recipes(self) -> List[Dict]:
        """Get all recipes"""
        return self.recipes

    def get_recipe(self, title: str) -> Optional[Dict]:
        """Get recipe by title"""
        idx = self.title_to_idx.get(title)
        if idx is not None:
            return self.recipes[idx]
        return None

    def delete_recipe(self, title: str) -> bool:
        """Delete recipe"""
        if title in self.title_to_idx:
            idx = self.title_to_idx[title]

            # Remove from index (FAISS limitation: we'd need to rebuild)
            # For simplicity, mark as deleted in our list
            del self.title_to_idx[title]
            self.recipes[idx] = {"title": title, "deleted": True}

            self.changes.append(
                {"title": title, "action": "delete", "time": datetime.now().isoformat()}
            )

            self._save()
            return True

        return False

    def get_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent changes"""
        return self.changes[-limit:]
