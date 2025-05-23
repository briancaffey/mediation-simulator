"""
Generate images with flux
"""

import os
import base64
import time
import random
import requests
from pathlib import Path
from typing import List

def ensure_data_dir():
    """Ensure the data directory exists"""
    data_dir = Path("flux/data")
    data_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Using data directory: {data_dir}")
    return data_dir

def generate_images(prompts: List[str]) -> List[str]:
    """
    Generate images using the Flux API for each prompt in the list.

    Args:
        prompts: List of prompt strings to generate images for

    Returns:
        List of paths to the generated images
    """
    data_dir = ensure_data_dir()
    generated_files = []

    url = "http://192.168.5.173:8000/v1/infer"
    print(f"🎨 Starting image generation for {len(prompts)} prompts")

    for i, prompt in enumerate(prompts, 1):
        print(f"🔄 Processing prompt {i}/{len(prompts)}")
        print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        payload = {
            "prompt": prompt,
            "height": 768, # 1024,
            "width": 1344, # 1024,
            "cfg_scale": 5,
            "mode": "base",
            "samples": 1,
            "seed": random.randint(0, 2**32 - 1),  # Random 32-bit integer
            "steps": 50
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }

        try:
            print("📡 Sending request to API...")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            if "artifacts" in result:
                for artifact in result["artifacts"]:
                    if artifact["finishReason"] == "SUCCESS":
                        # Generate filename using Unix epoch timestamp
                        timestamp = int(time.time() * 1000)  # milliseconds for more precise ordering
                        filename = f"{timestamp}.png"
                        filepath = data_dir / filename

                        # Convert base64 to image and save
                        print("💾 Converting and saving image...")
                        image_data = base64.b64decode(artifact["base64"])
                        with open(filepath, "wb") as f:
                            f.write(image_data)

                        generated_files.append(str(filepath))
                        print(f"✅ Image saved successfully: {filepath}")
                    else:
                        print(f"⚠️ Image generation finished with status: {artifact['finishReason']}")
            else:
                print(f"❌ Error generating image for prompt: {prompt}")
                print(f"🔍 API response: {result}")

        except requests.exceptions.RequestException as e:
            print(f"🚫 Network error for prompt '{prompt}': {str(e)}")
        except Exception as e:
            print(f"💥 Unexpected error for prompt '{prompt}': {str(e)}")

    print(f"✨ Generation complete! Generated {len(generated_files)} images successfully")
    return generated_files

if __name__ == "__main__":
    # Example usage
    test_prompts = [
        "Corporate mediation session in a modern office with diverse professionals in business attire, serious expressions, discussing over a glass conference table",
        "Business mediation between two startup founders with a mediator, whiteboard in the background, casual but professional setting",
        "Formal mediation meeting with a male mediator guiding two diverse female executives in a high-rise boardroom",
        "Team conflict resolution session with four coworkers, mixed ethnicities, around a round table in a bright coworking space",
        "Legal mediation between a tech company and a client, with laptops and printed contracts on the table, professional attire",
        "Dispute resolution meeting in a corporate law firm, people in suits, neutral body language, one person taking notes",
        "Business negotiation with a mediator present, two parties shaking hands, diverse group, downtown office skyline visible through windows",
        "Mediation training workshop, participants listening to instructor, flip chart with diagram, semi-formal setting",
        "Cross-cultural mediation session with international businesspeople, translator present, all in professional dress",
        "HR-led mediation session addressing workplace conflict, empathetic expressions, office plants and natural lighting",
        "Startup co-founders with their mentor acting as mediator, tech-themed office, diverse ethnic backgrounds",
        "High-stakes contract mediation between executives, tense atmosphere, legal documents spread out, neutral third party moderating",
        "Remote mediation session shown in hybrid format, some people on screen, others around the table, corporate casual clothing",
        "Government official mediating between two private sector representatives, flags and formal décor in background",
        "Facilitated negotiation between two businesswomen, mediator using hand gestures, soft lighting and business charts behind them",
        "Panel-style mediation session with observers, people in business suits, taking notes and evaluating the interaction",
        "Young professionals engaged in peer mediation, casual tech office with glass walls and open layout",
        "Multi-party mediation in a boardroom, one person speaking while others listen, corporate diversity visible",
        "Conflict resolution simulation during a corporate training seminar, diverse participants roleplaying in teams",
        "Formal arbitration session with legal counsel present, mediator presiding, courtroom-style business setting"
    ]

    test_prompts = [
        "A tense boardroom meeting with executives from NovaTech Solutions (Lumina Republic) and Aether Industries (Zephyra Emirates), seated at a long table reviewing a disputed supply contract, with digital screens displaying delivery timelines and financial terms in LMD/ZED currency",
        "A high-tech semiconductor manufacturing facility in Lumina Republic, workers in cleanroom suits inspecting advanced components on automated assembly lines, with signage indicating quality control checkpoints for the disputed shipments",
        "A logistics hub with shipping containers labeled 'Zephyra Emirates' and 'Lumina Republic', customs officials reviewing documentation, while a team of auditors examines a third-party report highlighting component defects under bright overhead lighting",
        "A corporate courtroom in Lumina Republic featuring a judge presiding over contractual disputes, lawyers from both firms presenting evidence on large screens showing payment records and delivery delay timelines, with a jury panel observing the proceedings",
        "A quality assurance lab in Zephyra Emirates where engineers in white coats conduct tests on semiconductor components under microscopes, digital readouts displaying defect rates disputed by both parties, with a stack of audit reports nearby",
        "A financial review meeting at NovaTech's headquarters featuring accountants analyzing payment discrepancies, spreadsheets showing withheld funds for delayed shipments, and a map highlighting supply chain routes between Lumina Republic and Zephyra Emirates",
        "A mediation session in a neutral conference room with representatives from both companies seated around a circular table, legal advisors reviewing contract clauses related to force majeure exceptions, while a mediator facilitates discussion under soft lighting",
        "A secure data center in Zephyra Emirates housing servers storing supply chain records, IT professionals accessing digital logs of shipment tracking and quality inspections, with encrypted files labeled 'contractual compliance audit' visible on monitors",
        "A warehouse in Lumina Republic with forklifts loading damaged semiconductor shipments, workers documenting defects on inspection forms, and a sign indicating 'delayed deliveries - dispute pending', surrounded by stacks of contractual documentation",
        "An international arbitration panel featuring legal experts from both jurisdictions reviewing a third-party audit report, with holographic displays showing component defect analysis and payment reconciliation charts in multiple currencies"
    ]

    test_prompts = [
        "A tense boardroom meeting with executives from NovaTech Solutions (Lumina Republic) and Aether Industries (Zephyra Emirates), one of them is whispering in the ear of the other and that person has a surprised look on their face",
        "An envelope labeled 'Confidential' sits on a desk with a man in a suit looking at it",
    ]

    test_prompts = [
        "a law school mediation competitions with two groups of students on each side and a mediator in the middle engaging in dialog",
        "two business women shaking hands after resolving a big dispute",
    ]

    generated_files = generate_images(test_prompts)

