import datetime

import metapy


def main():
    idx = metapy.index.make_inverted_index('config.toml')
    print(idx.num_docs())
    ranker = metapy.index.OkapiBM25()
    query = metapy.index.Document()
    query.content('great')
    top_docs = ranker.score(idx, query, num_results=5)
    print(top_docs)
    for num, (d_id, _) in enumerate(top_docs):
        print(d_id)
        print(idx.metadata(d_id).get('content'))


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    main()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))

# https://github.com/meta-toolkit/meta-toolkit.org/blob/master/overview-tutorial.md
