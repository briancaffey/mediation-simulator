# generates case_list_data.json

import os
import yaml
from pathlib import Path
from datetime import datetime
import random


def get_random_cover_image():
    """Return a random cover image from the local flux/data directory."""
    images = [
        "flux/data/1747700517310.png",
        "flux/data/1747960956975.png",
        "flux/data/1747960939727.png",
        "flux/data/1747960922365.png",
        "flux/data/1747960904999.png",
        "flux/data/1747960905066.png",
        "flux/data/1747960887710.png",
        "flux/data/1747960870512.png",
        "flux/data/1747960853301.png",
    ]
    return random.choice(images)


def generate_case_list():
    """Generate the case list YAML file from mediation state data."""
    data_dir = Path("aiq/data")
    cases = []

    # Walk through the data directory
    for case_dir in data_dir.iterdir():
        if not case_dir.is_dir() or case_dir.name == "all_cases.yml":
            continue

        # Check if case_generation_state.yaml exists
        state_file = case_dir / "case_generation_state.yaml"
        if not state_file.exists():
            continue

        # Read the case generation state
        with open(state_file, "r", encoding="utf-8") as f:
            try:
                state_data = yaml.safe_load(f)
            except yaml.YAMLError:
                print(f"Error reading {state_file}")
                continue

        # Extract case information
        case_title = state_data.get("case_title", f"Case {case_dir.name}")
        case_summary = state_data.get("case_summary", "")

        # Get the date from the file's creation time
        date = datetime.fromtimestamp(state_file.stat().st_ctime).strftime("%Y-%m-%d")

        # Create case entry
        case = {
            "id": case_dir.name,
            "title": case_title,
            "cover_image_url": get_random_cover_image(),
            "description": (
                case_summary[:200] + "..." if len(case_summary) > 200 else case_summary
            ),
            "date": date,
            "status": "In Progress",  # Default status since we don't have mediation state
        }

        cases.append(case)

    # Sort cases by date (newest first)
    cases.sort(key=lambda x: x["date"], reverse=True)

    # Create the YAML file
    output_data = {"cases": cases}
    output_file = data_dir / "all_cases.yml"

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {output_file} with {len(cases)} cases")


if __name__ == "__main__":
    generate_case_list()
