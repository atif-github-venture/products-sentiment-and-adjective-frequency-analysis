import datetime
import metapy

import pandas

from resources.constants import Constants


def main():
    idx = metapy.index.make_inverted_index('config.toml')
    print(idx.num_docs())
    ranker = metapy.index.OkapiBM25()
    query = metapy.index.Document()
    column = ['adjective', 'freq', 'sentiment']
    df = pandas.read_csv(Constants.OUTPUT_PATH + '/' + Constants.TOP_POS_NEG_ADJ, names=column)

    for ind, row in df.iterrows():
        q = row['adjective']
        print('Query -> '+ q)
        query.content(q)
        top_docs = ranker.score(idx, query, num_results=5)
        print(top_docs)
        for num, (d_id, _) in enumerate(top_docs):
            print(idx.metadata(d_id).get('content'))
        print('*****************')


if __name__ == '__main__':
    st_time = datetime.datetime.now()
    main()
    en_time = datetime.datetime.now()
    print('Total execution time (milliseconds): ' + str((en_time - st_time).total_seconds() * 1000))

# https://github.com/meta-toolkit/meta-toolkit.org/blob/master/overview-tutorial.md
