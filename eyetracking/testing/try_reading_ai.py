"""
Try the reading AI in the terminal.

    python -m eyetracking.testing.try_reading_ai                      (built-in sample)
    python -m eyetracking.testing.try_reading_ai full.txt read.txt    (your own text)

Options:
    --mode      summary-unread | summary-outline | questions-unread | questions-whole
    --provider  claude | gemini | openai

read.txt holds the parts you "read", copied from full.txt (any order).
In the question modes, type an answer or press Enter to skip.
"""
import argparse
from pathlib import Path

from eyetracking.core.reading import mark_unread, unread_word_count
from eyetracking.services.reading_ai import MODELS, grade_answer, make_questions, summarize

SAMPLE_FULL = """\
When you read, your eyes do not glide smoothly along a line. They move in quick jumps called saccades, each lasting only a few hundredths of a second. During a saccade you take in almost no new information; the brain hides the blur so you do not notice it.

Between saccades the eyes stop on a word for about a quarter of a second. These stops are called fixations, and they are when the actual reading happens. Short, common words like "the" are often skipped entirely, while long or unfamiliar words get longer fixations, or even two.

About one saccade in ten goes backwards. These regressions usually happen when a reader misunderstands a sentence or skipped a word too quickly. Skilled readers make fewer regressions, but no reader avoids them completely.

At the end of each line the eyes make a long jump back to the start of the next line, called a return sweep. Because an eye tracker can see all of these movements, it can estimate which parts of a text a person actually read and which parts they skipped."""

# the sample reader read paragraphs 1 and 4 and skipped 2 and 3
SAMPLE_READ = "\n\n".join(SAMPLE_FULL.split("\n\n")[i] for i in (0, 3))

MODES = ["summary-unread", "summary-outline", "questions-unread", "questions-whole"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Try the reading AI in the terminal.")
    parser.add_argument("full", nargs="?", type=Path, help="text file with the whole text")
    parser.add_argument("read", nargs="?", type=Path, help="text file with the parts you read")
    parser.add_argument("--mode", choices=MODES, default="summary-unread")
    parser.add_argument("--provider", choices=list(MODELS), default="claude")
    args = parser.parse_args()

    if args.full and not args.read:
        parser.error("give both files: full.txt read.txt")
    if args.full:
        full_text = args.full.read_text(encoding="utf-8")
        read_text = args.read.read_text(encoding="utf-8")
    else:
        full_text, read_text = SAMPLE_FULL, SAMPLE_READ

    unread = unread_word_count(mark_unread(full_text, read_text))
    print(f"{unread} of {len(full_text.split())} words unread")
    print(f"Asking {args.provider} ({MODELS[args.provider]})...")

    kind, scope = args.mode.split("-")
    if kind == "summary":
        print("\n" + summarize(full_text, read_text, scope, args.provider))
        return

    questions = make_questions(full_text, read_text, scope, args.provider)
    if not questions:
        print("You read the whole text: nothing unread to ask about.")
        return
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {q['question']}")
        answer = input("Your answer (Enter to skip): ").strip()
        if answer:
            print("Checking...")
            result = grade_answer(full_text, q, answer, args.provider)
            print(f"Score: {result['score']}/10. {result['explanation']}")
        print(f"Model answer: {q['model_answer']}")


if __name__ == "__main__":
    main()