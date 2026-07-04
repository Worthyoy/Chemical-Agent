from collections import defaultdict
from typing import List

from langgraph_workflow.benchmark_utils import (
    SUPPORTED_SOURCE_MODALITIES,
    normalize_source_modality,
)


def build_paper_question_option_summary(
    reactions: List[dict],
    question_option_counts: List[dict],
) -> dict:
    paper_modalities = defaultdict(set)
    for reaction in reactions:
        paper = str(reaction.get("source_paper") or "").strip()
        if not paper:
            continue
        modality = normalize_source_modality(reaction.get("source_modality"))
        if modality not in SUPPORTED_SOURCE_MODALITIES:
            raise ValueError(
                f"Unsupported or missing source_modality for paper {paper}: {modality}"
            )
        paper_modalities[paper].add(modality)

    question_ids = set()
    questions_by_paper = defaultdict(list)
    for question in question_option_counts:
        question_id = str(question.get("question_id") or "").strip()
        paper = str(question.get("source_paper") or "").strip()
        modality = normalize_source_modality(question.get("source_modality"))
        if not question_id or question_id in question_ids:
            raise ValueError(f"Duplicate or missing question_id: {question_id}")
        if not paper or paper not in paper_modalities:
            raise ValueError(
                f"Question {question_id} references an unknown source_paper: {paper}"
            )
        if modality not in paper_modalities[paper]:
            raise ValueError(
                f"Question {question_id} has modality {modality}, which is absent for {paper}"
            )
        question_ids.add(question_id)
        questions_by_paper[paper].append(
            {
                "question_id": question_id,
                "task": str(question.get("task") or "").upper(),
                "source_modality": modality,
                "reaction_type": question.get("reaction_type"),
                "option_count": int(question.get("option_count", 0)),
            }
        )

    papers = []
    for paper in sorted(paper_modalities, key=lambda value: value.casefold()):
        questions = sorted(
            questions_by_paper.get(paper, []),
            key=lambda question: question["question_id"],
        )
        by_modality = {}
        for modality in SUPPORTED_SOURCE_MODALITIES:
            modality_questions = [
                question
                for question in questions
                if question["source_modality"] == modality
            ]
            option_counts = sorted(
                {question["option_count"] for question in modality_questions}
            )
            by_modality[modality] = {
                "q1_questions": sum(
                    question["task"] == "Q1" for question in modality_questions
                ),
                "q2_questions": sum(
                    question["task"] == "Q2" for question in modality_questions
                ),
                "total_questions": len(modality_questions),
                "option_count_distribution": {
                    str(count): sum(
                        question["option_count"] == count
                        for question in modality_questions
                    )
                    for count in option_counts
                },
            }
        papers.append(
            {
                "source_paper": paper,
                "source_modalities": sorted(paper_modalities[paper]),
                "total_questions": len(questions),
                "q1_questions": sum(question["task"] == "Q1" for question in questions),
                "q2_questions": sum(question["task"] == "Q2" for question in questions),
                "by_modality": by_modality,
                "questions": questions,
            }
        )

    total_questions = sum(paper["total_questions"] for paper in papers)
    if total_questions != len(question_option_counts):
        raise ValueError("Paper question totals do not match question_option_counts")

    return {
        "total_papers": len(papers),
        "papers_with_text": sum(
            "text" in modalities for modalities in paper_modalities.values()
        ),
        "papers_with_image": sum(
            "image" in modalities for modalities in paper_modalities.values()
        ),
        "papers_with_both_modalities": sum(
            {"text", "image"} <= modalities
            for modalities in paper_modalities.values()
        ),
        "total_questions": total_questions,
        "questions_by_modality": {
            modality: sum(
                paper["by_modality"][modality]["total_questions"] for paper in papers
            )
            for modality in SUPPORTED_SOURCE_MODALITIES
        },
        "papers": papers,
    }
