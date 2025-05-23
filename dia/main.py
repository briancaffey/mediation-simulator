import requests
import json
import logging
import os
import time
import soundfile as sf
import numpy as np
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_audio_from_prompt(audio_prompt_path, voice_sample_text, prompt, output_dir="output"):
    """
    Generate audio from a prompt using a voice sample.

    Args:
        audio_prompt_path (str): Path to the audio prompt file
        voice_sample_text (str): Text describing the voice sample
        prompt (str): The text prompt to generate audio for
        output_dir (str): Directory to save the output audio file

    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logging.info("🎯 Starting audio generation process...")
    logging.info(f"📂 Loading audio file from: {audio_prompt_path}")

    try:
        # First, upload the audio file
        with open(audio_prompt_path, "rb") as audio_file:
            files = {
                'files': ('Alice_.wav', audio_file, 'audio/wav')
            }
            upload_response = requests.post(
                "http://192.168.5.173:7860/gradio_api/upload",
                files=files
            )

            if upload_response.status_code != 200:
                logging.error(f"❌ Failed to upload audio file. Status code: {upload_response.status_code}")
                raise Exception("Failed to upload audio file")

            upload_data = upload_response.json()
            if not isinstance(upload_data, list) or len(upload_data) == 0:
                logging.error("❌ Invalid response from upload endpoint")
                raise Exception("Invalid upload response")

            uploaded_file_path = upload_data[0]
            logging.info(f"✅ Successfully uploaded audio file: {uploaded_file_path}")

    except Exception as e:
        logging.error(f"❌ Failed to process audio file: {str(e)}")
        raise

    # First request to initiate the generation
    logging.info("📤 Sending initial request to generate audio...")
    try:
        response = requests.post(
            "http://192.168.5.173:7860/gradio_api/call/generate_audio",
            headers={"Content-Type": "application/json"},
            json={
                "data": [
                    voice_sample_text + prompt,  # text_input
                    {
                        "path": uploaded_file_path,
                        "meta": {"_type": "gradio.FileData"}
                    },  # audio_prompt_input
                    3072,  # max_new_tokens
                    3.0,  # cfg_scale
                    1.3,  # temperature
                    0.95,  # top_p
                    30,   # cfg_filter_top_k
                    0.94   # speed_factor_slider
                ]
            }
        )

        # Log the response status and content for debugging
        logging.info(f"📥 Received response with status code: {response.status_code}")
        response_data = response.json()
        logging.info(f"📦 Response data: {json.dumps(response_data, indent=2)}")

        if "event_id" not in response_data:
            logging.error("❌ Response does not contain 'event_id' key. Available keys: " +
                         ", ".join(response_data.keys()))
            raise KeyError("Expected 'event_id' key in response")

        # Extract the event ID from the response
        event_id = response_data["event_id"]
        logging.info(f"✅ Successfully received event ID: {event_id}")

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Network error occurred: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"❌ Failed to parse JSON response: {str(e)}")
        logging.error(f"Raw response content: {response.text}")
        raise
    except KeyError as e:
        logging.error(f"❌ Missing expected key in response: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        raise

    # Second request to get the actual audio data
    logging.info("🎵 Requesting audio data...")
    try:
        audio_response = requests.get(
            f"http://192.168.5.173:7860/gradio_api/call/generate_audio/{event_id}",
            stream=True
        )

        if audio_response.status_code == 200:
            # Process the SSE response
            audio_url = None
            for line in audio_response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    logging.info(f"📝 Received event: {line}")

                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            if isinstance(data, list) and len(data) > 0:
                                audio_data = data[0]
                                if isinstance(audio_data, dict) and 'url' in audio_data:
                                    audio_url = audio_data['url']
                                    logging.info(f"🎵 Found audio URL: {audio_url}")
                                    break
                        except json.JSONDecodeError:
                            continue

            if audio_url:
                # Now fetch the actual audio file using the provided URL
                logging.info(f"📥 Downloading audio from: {audio_url}")
                audio_file_response = requests.get(audio_url)

                if audio_file_response.status_code == 200:
                    # Generate timestamp-based filename
                    timestamp = int(time.time())
                    output_path = os.path.join(output_dir, f"{timestamp}.wav")

                    logging.info(f"💾 Saving audio data to file: {output_path}")
                    with open(output_path, "wb") as f:
                        f.write(audio_file_response.content)

                    # Verify the file was written correctly
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        logging.info(f"✨ Successfully saved audio to {output_path} (size: {file_size} bytes)")
                        return output_path
                    else:
                        logging.error("❌ Failed to create output file")
                else:
                    logging.error(f"❌ Failed to download audio file. Status code: {audio_file_response.status_code}")
                    logging.error(f"Response content: {audio_file_response.text}")
            else:
                logging.error("❌ Could not find audio URL in the response")
        else:
            logging.error(f"❌ Failed to get audio data. Status code: {audio_response.status_code}")
            logging.error(f"Response content: {audio_response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Network error occurred while fetching audio: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"❌ Error processing audio data: {str(e)}")
        raise

if __name__ == "__main__":
    # TODO: remove hard-coded prompts
    voice_sample_text = """[S1] We believe that exploring new ideas and sharing knowledge helps make the world a brighter place for everyone. Continuous learning and open communication are essential for progress and mutual understanding.
    """
    # sample prompt from mediation
    # prompt = """[S1] Welcome, both parties, to this mediation session. I am your neutral facilitator here today, tasked with guiding this discussion toward a constructive resolution. The dispute between Nordland SolarSync and Solevia GreenPulse Distributors centers on a multi-year supply agreement entered in 2023, aimed at distributing high-efficiency solar panels across the Solara Belt region."""
    prompt = """[S1] Good morning, Jamie Zhang, Taylor Kim, representatives of NovaSpire Inc. and HelixCorp Ltd., and members of their respective organizations. Thank you for joining this mediation session. My role is to facilitate a structured, neutral conversation aimed at resolving the disputes arising from your partnership on the SmartWeave project."""
    # Path to the audio prompt file
    audio_prompt_path = os.path.join("voices", "Alice.wav")

    voice_prompts = [
        "Good morning, Aria Venn and Kael Thorne, and thank you for being here today. My role as the mediator is to facilitate a structured and respectful dialogue between NovaTech Solutions and ZenCorp Industries to address the unresolved issues stemming from your supply contract dispute. I am committed to maintaining neutrality throughout this process, ensuring that both parties have equal opportunity to present their perspectives without interruption or bias.",
        "Good morning, Mediator, Kael Thorne, and thank you for being here today. I am Aria Venn, Head of Supply Chain at NovaTech Solutions, and I appreciate the opportunity to engage in this mediation process with ZenCorp Industries. My goal is to work collaboratively with both parties to resolve the outstanding issues stemming from our supply contract and find a resolution that respects the terms we agreed upon in 2021.",
        "Kael Thorne, Director of Procurement at ZenCorp Industries, acknowledges the importance of resolving this matter collaboratively and emphasizes that ZenCorp’s actions were rooted in ensuring compliance with contractual obligations to protect both parties’ interests. While we understand NovaTech’s concerns regarding delayed shipments, our decision to withhold payment for the disputed batch was based on documented quality defects that directly impacted our production processes and incurred significant financial losses.",
        "Thank you for your clarification, Kael. NovaTech fully understands ZenCorp’s concerns regarding the impact of quality defects on production schedules and financial planning. We recognize that our obligations under the contract include ensuring components meet agreed-upon specifications, and we are committed to addressing any gaps in compliance. However, we also need to ensure that our right to timely payment is not unduly compromised by disputes over deliverables that were already accounted for in the contractual terms.",
        "Regarding invoice discrepancies, we will submit a comprehensive reconciliation of all outstanding invoices, including dates, quantities, and supporting delivery receipts. Any delays in processing were due to internal administrative challenges, which we are now addressing to avoid future disruptions. We also propose a temporary payment arrangement for the disputed batch: once the third-party review confirms compliance with quality standards, NovaTech is willing to accept a phased payment plan to align with ZenCorp’s operational timelines."
    ]

    voice_prompts = [
        "ZenCorp Industries appreciates the mediator’s structured approach to resolving this matter and recognizes the value of collaborative problem-solving. We agree that a thorough examination of the quality defects is essential to determine whether our withholding of payment was justified under the contractual terms. To this end, we are prepared to cooperate fully by providing the requested documentation, including detailed records of defect assessments and their operational impact."
    ]
    for voice_prompt in voice_prompts:
        output_file = generate_audio_from_prompt(audio_prompt_path, voice_sample_text, f"[S1] {voice_prompt}")
        if output_file:
            logging.info(f"🎉 Audio generation completed successfully. Output saved to: {output_file}")

    # # Generate the audio
    # output_file = generate_audio_from_prompt(audio_prompt_path, voice_sample_text, prompt)
    # if output_file:
    #     logging.info(f"🎉 Audio generation completed successfully. Output saved to: {output_file}")
