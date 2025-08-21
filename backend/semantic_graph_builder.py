import re
from typing import List, Dict, Any
from concept_mapper import map_to_concept


def split_speakers(transcript: str) -> List[Dict[str, str]]:
    # Add Speaker N: to the pattern; make it case-insensitive
    pattern = r"(Interviewer|Interviewee(?:\s*-\s*[A-Za-z ]+)?|Speaker\s*\d+):"
    matches = list(re.finditer(pattern, transcript, flags=re.IGNORECASE))
    
    segments = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(transcript)
        speaker = matches[i].group(1).strip()
        text = transcript[start:end].strip()

        # Default any non-"Interviewer" to interviewee
        role = "interviewer" if "interviewer" in speaker.lower() else "interviewee"

        # Name extraction:
        # - "Interviewee - Nick Ruscher" -> "Nick Ruscher"
        # - "Speaker 0" / "Speaker1" -> keep that label as the name
        if "-" in speaker:
            name = speaker.split("-", 1)[-1].strip()
        elif re.match(r"(?i)^speaker\s*\d+$", speaker):
            name = speaker  # e.g., "Speaker 0"
        else:
            name = None

        segments.append({
            "speaker_id": role.lower(),
            "speaker_name": name if name else role.title(),
            "text": text
        })

    return segments
def build_reasoning_trace(transcript: str) -> Dict[str, Any]:
    segments = split_speakers(transcript)
    reasoning_steps = []

    # Step 1: Concept Mapping
    all_mappings = {}
    for seg in segments:
        sentences = [s.strip() for s in seg["text"].split(".") if len(s.strip()) > 5]
        for sent in sentences:
            concept, score = map_to_concept(sent)
            if concept not in all_mappings:
                all_mappings[concept] = []
            all_mappings[concept].append({
                "speaker": seg["speaker_id"],
                "speaker_name": seg["speaker_name"],
                "phrase": sent,
                "score": round(score, 2)
            })

    detailed_mapping = {}
    for concept, items in all_mappings.items():
        detailed_mapping[concept] = [item["phrase"] for item in items]

    reasoning_steps.append({
        "step_id": "concept_mapping",
        "title": "Mapped interview sentences to key concepts",
        "details": {
            "detailed_mapping": detailed_mapping
        },
        "content": f"Mapped {sum(len(v) for v in detailed_mapping.values())} sentences to concepts.",
    })

    # Step 2: Speaker Concept Analysis
    speaker_concept_count = {}
    for concept, items in all_mappings.items():
        for item in items:
            speaker = item["speaker"]
            speaker_concept_count[speaker] = speaker_concept_count.get(speaker, 0) + 1

    non_interviewer_speakers = {
        speaker: count for speaker, count in speaker_concept_count.items() 
        if 'interviewer' not in speaker.lower()
    }

    if non_interviewer_speakers:
        primary_speaker = max(non_interviewer_speakers.items(), key=lambda x: x[1])[0]
    else:
        primary_speaker = max(speaker_concept_count.items(), key=lambda x: x[1])[0]

    print(f"🔍 All speakers: {speaker_concept_count}")
    print(f"🔍 Non-interviewer speakers: {non_interviewer_speakers}")
    print(f"🔍 Selected primary: {primary_speaker}")
    reasoning_steps.append({
        "step_id": "speaker_analysis",
        "title": "Analyzed speaker contribution to concepts",
        "details": {
            "speaker_entity_counts": speaker_concept_count,
            "primary_speaker": primary_speaker
        },
        "content": "Identified primary speaker and their conceptual engagement."
    })

    return {
        "steps": reasoning_steps
    }
