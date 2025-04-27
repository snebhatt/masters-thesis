import numpy as np


def dirichlet_distribution(dataset, n_clients, alpha=0.5):
    labels = dataset.get_ytrain().numpy()
    n_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    client_data_indices = [[] for _ in range(n_clients)]

    for label in range(n_classes):
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        proportions = label_distribution[label]
        split_indices = (proportions * len(label_indices)).astype(int)

        start = 0
        for client_id, count in enumerate(split_indices):
            client_data_indices[client_id].extend(label_indices[start:start + count])
            start += count

    client_datasets = []
    for indices in client_data_indices:
        client_X = dataset.get_Xtrain()[indices]
        client_y = dataset.get_ytrain()[indices]
        client_datasets.append((client_X, client_y))

    return client_datasets