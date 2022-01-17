from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import wordnet

class WuPalmerScoreCalculator:
    def __init__(self, answer_space: List[str]):
        self.answer_space = answer_space

    def wup_measure(self, a: str, b: str, similarity_threshold: float = 0.925):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        def get_semantic_field(a):
            weight = 1.0
            semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
            return (semantic_field,weight)


        def get_stem_word(a):
            """
            Sometimes answer has form word\d+:wordid.
            If so we return word and downweight
            """
            weight = 1.0
            return (a,weight)


        global_weight=1.0

        (a,global_weight_a)=get_stem_word(a)
        (b,global_weight_b)=get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0


        interp_a,weight_a = get_semantic_field(a) 
        interp_b,weight_b = get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score


    def batch_wup_measure(self, labels: np.ndarray, preds: np.ndarray) -> float:
        wup_scores = [self.wup_measure(self.answer_space[label], self.answer_space[pred]) for label, pred in zip(labels, preds)]
        return np.mean(wup_scores)


    def compute_metrics(self, eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_tuple
        preds = logits.argmax(axis=-1)
        return {
            "wups": self.batch_wup_measure(labels, preds),
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='macro')
        }