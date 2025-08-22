import fitz  # PyMuPDF
import sys
from backend.semantic_graph_builder import build_reasoning_trace

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def pretty_print_reasoning(reasoning_trace):
    for step in reasoning_trace["steps"]:
        print(f"\n=== Step: {step['step_id']} ===")
        print(f"Title: {step['title']}")
        print(f"Content: {step['content']}")
        if "details" in step:
            for k, v in step["details"].items():
                print(f" - {k}: {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python graph_test_runner.py path/to/transcript.pdf")
        sys.exit(1)

    transcript_path = sys.argv[1]
    print(f"Loading transcript from: {transcript_path}")

    text = extract_text_from_pdf(transcript_path)
    reasoning_trace = build_reasoning_trace(text)
    pretty_print_reasoning(reasoning_trace)
