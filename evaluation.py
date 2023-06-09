from qrels_wiki import dic
from Cosine_Similarity import wiki_query_documents
# Compute the average precision (AP) for a single query
def compute_ap(query_results, query_qrels):
    # Compute the precision and recall at each rank
    precision = []
    recall = []
    relevant = 0
    for i, doc in enumerate(query_results):
        if doc in query_qrels:
            relevant += query_qrels[doc]
            precision.append(relevant / (i+1))
            recall.append(relevant / len(query_qrels))
    # Compute the average precision
    ap = sum(precision[i] * (recall[i] - recall[i-1]) for i in range(1, len(precision)) if recall[i] > recall[i-1])
    return ap

# Compute the mean average precision (MAP) for all queries
map_score = 0
for query in wiki_query_documents:
    ap = compute_ap(wiki_query_documents[query], dic[query])
    map_score += ap
map_score /= len(wiki_query_documents)
print("MAP score:", map_score)