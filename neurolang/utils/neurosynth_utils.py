import logging
import os

try:
    import neurosynth as ns
except ModuleNotFoundError:
    raise ImportError("Neurosynth not installed in the system")


def fetch_neurosynth_data(terms, frequency_threshold=0.05, q=0.01, prior=0.5, image_type='pAgF_z_FDR_0.01'):

    file_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(file_dir, './data')
    file = os.path.join(path, 'dataset.pkl')

    if not os.path.isfile(file):
        logging.info(
            f'Downloading neurosynth database and features in path: {path}'
        )
        dataset = download_ns_dataset(path)
    else:
        dataset = ns.Dataset.load(file)

    studies_ids = dataset.get_studies(
        features=terms, frequency_threshold=frequency_threshold
    )
    ma = ns.meta.MetaAnalysis(dataset, studies_ids, q=q, prior=prior)
    data = ma.images[image_type]
    masked_data = dataset.masker.unmask(data)
    affine = dataset.masker.get_header().get_sform()
    dim = dataset.masker.dims
    return masked_data, affine, dim


def download_ns_dataset(path):
    if not os.path.exists(path):
        os.makedirs(path)
    ns.dataset.download(path=path, unpack=True)
    dataset = ns.Dataset(os.path.join(path, 'database.txt'))
    dataset.add_features(os.path.join(path, 'features.txt'))
    dataset.save(os.path.join(path, 'dataset.pkl'))
    return dataset