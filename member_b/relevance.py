
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class RelevanceChecker:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.30
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def check(self, claim, evidence):
        claim_embedding = self.model.encode(
            claim,
            convert_to_tensor=True
        )

        evidence_embedding = self.model.encode(
            evidence,
            convert_to_tensor=True
        )

        similarity = cos_sim(
            claim_embedding,
            evidence_embedding
        ).item()

        return {
            "similarity": float(similarity),
            "is_relevant": similarity >= self.threshold
        }

    def check_multiple(self, claim, evidence_passages):
        results = [
            self.check(claim, passage)
            for passage in evidence_passages
        ]

        max_similarity = max(
            (r["similarity"] for r in results),
            default=0.0
        )

        return {
            "similarity": max_similarity,
            "is_relevant": max_similarity >= self.threshold,
            "passage_results": results
        }
