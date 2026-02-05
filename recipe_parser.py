import re
from typing import Dict, List, Optional, Tuple


class RecipeParser:
    """Simple parser for your exact recipe format"""

    @staticmethod
    def parse(content: str, filename: str = "") -> Dict:
        """Parse recipe text"""
        lines = content.strip().split("\n")

        data = {
            "title": "Untitled",
            "time": "",
            "calories": None,
            "diet": "",
            "cuisine": "",
            "ingredients": [],
            "steps": [],
            "source": filename,
            "content": content,
        }

        section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Section headers
            if line.lower() == "ingredients:":
                section = "ingredients"
                continue
            elif line.lower() == "steps:":
                section = "steps"
                continue

            # Key-value pairs
            if ":" in line and section is None:
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

            # Ingredients
            elif section == "ingredients" and line.startswith("-"):
                data["ingredients"].append(line[1:].strip())

            # Steps
            elif section == "steps":
                step_match = re.match(r"^(\d+)\.\s+(.+)$", line)
                if step_match:
                    data["steps"].append(step_match.group(2))
                elif line and not line.startswith("-"):
                    data["steps"].append(line)

        return data

    @staticmethod
    def validate(content: str) -> Tuple[bool, List[str]]:
        """Validate recipe format"""
        errors = []

        content_lower = content.lower()
        # Check required sections
        if "title:" not in content_lower:
            errors.append("Missing 'Title:'")
        if "time:" not in content_lower:
            errors.append("Missing 'Time:'")
        if "ingredients" not in content_lower:  # Just check for the word
            errors.append("Missing ingredients section")
        if "steps" not in content_lower:  # Just check for the word
            errors.append("Missing steps section")
        return len(errors) == 0, errors

    @staticmethod
    def to_markdown(data: Dict) -> str:
        """Convert back to markdown"""
        lines = [
            f"Title: {data['title']}",
            f"Time: {data['time']}",
        ]

        if data.get("calories"):
            lines.append(f"Calories: {data['calories']}")
        if data.get("diet"):
            lines.append(f"Diet: {data['diet']}")
        if data.get("cuisine"):
            lines.append(f"Cuisine: {data['cuisine']}")

        lines.append("")
        lines.append("Ingredients:")
        for ing in data.get("ingredients", []):
            lines.append(f"- {ing}")

        lines.append("")
        lines.append("Steps:")
        for i, step in enumerate(data.get("steps", []), 1):
            lines.append(f"{i}. {step}")

        return "\n".join(lines)
