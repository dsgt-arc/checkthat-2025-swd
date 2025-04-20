from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

class Rerank:
    def __init__(self, device=None, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.cross_encoder = CrossEncoder(self.model_name, device=device)

    def rerank_with_crossencoder(self, row, k=10):
        tweet = row['tweet_text']
        title_abstracts = row['title_abstract']

        results = self.cross_encoder.rank(query=tweet, 
                                    documents=[f"{paper_data['title']} {paper_data['abstract']}"
                                                    for paper_id, paper_data in title_abstracts.items()],
                                                    top_k=k, 
                                                    show_progress_bar = True
                                                    )

        ranked_document_indices = [result['corpus_id'] for result in results]
        ranked_paper_ids = [list(title_abstracts.keys())[index] for 
                            index in ranked_document_indices]

        return ranked_paper_ids