import matplotlib.pyplot as plt
import metapy


def save_dataset_to_csv(dataset, path, filename):
    dataset.to_csv(path_or_buf=path + '/' + filename, index=False)


def show_plots(dataset, color, group_by):
    dataset.groupby(group_by).size().plot(kind='bar', colormap=color).set_ylabel('Count')
    plt.show()


def build_collection(doc, path, filename):
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.ListFilter(tok, path + '/' + filename,
                                      metapy.analyzers.ListFilter.Type.Reject)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok.set_content(doc.content())
    return tok


def write_to_file(path, filename, content):
    with open(path + '/' + filename, 'w') as fp:
        fp.write(content)
    fp.close()
