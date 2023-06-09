
dic = {}
#  if qry_id==doc_id :continue
# Open and read the qrels file
with open('Wikir1k/qrels.REL') as f:
    for line in f.readlines():
        # Split the line into its components
        query_id, _, doc_id, relevance = line.strip().split()
        if query_id==doc_id :continue
        # If the query_id is not already in the result_dict, add it
        if query_id not in dic:
            dic[query_id] = {}

        # Add the doc_id and relevance to the query_id dictionary
        dic[query_id][doc_id] = int(relevance)

# Print the final result_dict
print(dic)

# Note : the output of this function will be  dictionary which has every single query with its  files
# relevance  , So we can use it with the output of cosine_Similarity to calculate the evaluation

