def construct_command(model_name, embedder_name, task_name, top_k, confidence_threshold,priming_method):
    res = "python run_experiment.py -n"
    if model_name != 'albert-xlarge-v2':
        res += ' -m ' + model_name
    if embedder_name != 'paraphrase-MiniLM-L6-v2':
        res += ' -e ' + embedder_name
    res += ' -t ' + task_name
    res += ' -k ' + str(top_k)

    if top_k!= 0:
        res += ' -c ' + str(confidence_threshold)
        res += ' -p ' + priming_method
    res += '\n'
    return res


if __name__ == '__main__':

    with open("experiments.txt",'w') as f:
        f.write("# main experiments\n")
        model_name = 'albert-xlarge-v2'
        embedder_name = 'paraphrase-MiniLM-L6-v2'
        for task_name in ["agnews", "imdb", "sst2", "yahoo", "yelp", "rte", "boolq"]:
            f.write(construct_command(model_name, embedder_name, task_name, 0, 0, 0)) # unlabeled priming
            for k in [3, 10, 50]:
                for c in [0, 0.8]:
                    for priming_method in ['sim','uniform']:
                        f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))
            for c in [0, 0.8]:
                f.write(construct_command(model_name, embedder_name, task_name, 3, c, 'concat'))

        f.write("\n# experiments about priming method\n")
        c = 0
        task_name = 'agnews'
        model_name = 'albert-xlarge-v2'
        embedder_name = 'paraphrase-MiniLM-L6-v2'
        for k in [3, 10, 50]:
            for priming_method in ['s+c', 'sc']:
                f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))

        f.write("\n# experiments about confidence threshold\n")
        model_name = 'albert-xlarge-v2'
        embedder_name = 'paraphrase-MiniLM-L6-v2'
        task_name = 'agnews'
        priming_method = 'sim'
        for k in [3, 10]:
            for c in [0.75, 0.85]:
                f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))

        f.write("\n# experiments about underlying LM and Sentence Transformer\n")

        f.write("# priming case: in setting k=3, c=0, similarity weighted\n")
        k = 3
        c = 0
        priming_method = 'sim'
        for task_name in ['agnews' , 'yahoo', 'rte', 'imdb']:
            embedder_name = 'paraphrase-MiniLM-L6-v2'
            for model_name in ['albert-xxlarge-v2', 'roberta-large', 'bert-large-uncased']:
                f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))

            embedder_name = 'multi-qa-mpnet-base-dot-v1'
            model_name = 'albert-xlarge-v2'
            f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))

            model_name = 'albert-xxlarge-v2'
            for embedder_name in ['multi-qa-mpnet-base-dot-v1', 'multi-qa-MiniLM-L6-cos-v1', 'all-mpnet-base-v2',
                                  'msmarco-bert-base-dot-v5', 'princeton-nlp/unsup-simcse-roberta-large']:
                f.write(construct_command(model_name, embedder_name, task_name, k, c, priming_method))

        f.write("# without priming case (independent on the sentence transformer)\n")
        embedder_name = 'paraphrase-MiniLM-L6-v2'
        for task_name in ['agnews', 'yahoo', 'rte', 'imdb']:
            for model_name in ['albert-xxlarge-v2', 'roberta-large', 'bert-large-uncased']:
                f.write(construct_command(model_name, embedder_name, task_name, 0, 0, 0))



