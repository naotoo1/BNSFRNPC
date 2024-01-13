"""
Module implementation of beyond neural scaling laws beating power scaling laws through data pruning
"""

from enum import Enum
from operator import add
from functools import reduce
from sklearn.cluster import KMeans
import prosemble as ps
import pandas as pd
import numpy as np


class PruneMode(str, Enum):
    EASY = "easy"
    HARD = "hard"
    BOTH = "both"


class PruneType(str, Enum):
    IN = "in"
    OUT = "out"
    BOTH = "both"


class SSLType(str, Enum):
    HCM = "hcm"
    FCM = "fcm"
    PCM = "pcm"
    PFCM = "pfcm"
    KFPCM = "kfpcm"
    FPCM = "fpcm"
    IPCM = "ipcm"
    IPCM2 = "ipcm_2"
    KMEANS = "kmeans"
    BGPC = "bgpc"
    KFCM = "kfcm"


def get_extracted_file_features(dataset):
    """

    :param dataset
    :return: array-like: extracted features and corresponding file names
    """
    extracted_features, file_name = [], []
    for data_index, data_point in enumerate(dataset):
        extracted_features.append([data_index])
        file_name.append(data_point)
    return extracted_features, file_name


def get_cluster_distance_space(distance_space, file_names):
    """


    :param distance_space: array-like: 2D list containing the distance space
        wrt to the centroids
    :param file_names: list: files names
    :return: clustered distance space for a given cluster label.
    """

    subspace_information_list, subspace_list, counter = [], [], []
    label_basis, sorted_subspace = [], []
    nearest_sort, cluster_distance_space = [], []

    distance_space_information = [
        [
            file_names[sub_space_index],
            np.argmin(distance_subspace),
            np.min(distance_subspace),
        ]
        for sub_space_index, distance_subspace in enumerate(distance_space)
    ]

    number_cluster = len(np.unique([i[1] for i in distance_space_information]))

    for label in range(number_cluster):
        count = 0
        for index, distance_subspace_information in enumerate(
            distance_space_information
        ):
            if distance_subspace_information[1] == label:
                count += 1
                subspace_list.append(distance_subspace_information)
        counter.append(count)

    # get data subspace information on label basis
    get_sorted_space(counter, subspace_information_list, subspace_list)

    for subspace_information in subspace_information_list:
        for index, subspace in enumerate(subspace_information):
            label_basis.append(subspace[2])

    #  get the distance subspace information based on labels
    get_sorted_space(counter, sorted_subspace, label_basis)

    # get sorted subspace index based on closest prototype
    for i in sorted_subspace:
        nearest_sort.append(np.argsort(i))

    # get the sort files base on subspace information
    for i in range(len(nearest_sort)):
        cluster_distance_space.append(
            [subspace_information_list[i][v][0] for j, v in enumerate(nearest_sort[i])]
        )

    return cluster_distance_space


def get_sorted_space(x, y, z):
    init_count = 0
    for count in x:
        count = count + init_count
        y.append(z[init_count:count])
        init_count = count


def get_pruned_easy_hard_examples(sorted_distance_space, prune_fraction):
    """

    :param sorted_distance_space: list: list with indexes
        of sorted distanced space
    :param prune_fraction: float: prune percentage or fraction
    :return: array-like: pruned list with indices of sorted distance space
    """

    maximum_prune = int(
        len(sorted_distance_space)
        - np.ceil((1 - prune_fraction) * len(sorted_distance_space))
    )
    return sorted_distance_space[:maximum_prune], sorted_distance_space[maximum_prune:]


def get_prune_set(pruned_indexes, prune_mode):
    """

    :param pruned_indexes:list: list with pruned indexes
    :param prune_mode:str: easy , hard and both. If none default is both
    :return: array-like: list with file names of pruned data set.
    """
    match prune_mode:
        case PruneMode.EASY:
            return pruned_indexes[0]
        case PruneMode.HARD:
            return pruned_indexes[1]
        case PruneMode.BOTH:
            return pruned_indexes
        case _:
            raise RuntimeError("get_prune_set: none of the case match")


def get_prune_set_folder(
    distance_space,
    file_names,
    prune_fraction,
):
    """
    :param prune_fraction:float: fraction of the prune
    :param distance_space:computed distances from the nearest prototype
    :param file_names:
    :return:
    """
    clustered_distance_space = get_cluster_distance_space(
        distance_space,
        file_names,
    )

    prune_set = [
        get_prune_set(
            get_pruned_easy_hard_examples(
                clustered_distance_space[i],
                prune_fraction,
            ),
            PruneMode.BOTH,
        )
        for i in range(len(clustered_distance_space))
    ]

    return prune_set


class SSL:
    def __init__(
        self,
        number_cluster,
        random_state,
        dataset,
    ):
        self.number_cluster = number_cluster
        self.random_state = random_state
        self.dataset = dataset
        self.number_topology = []

    def get_embedded_space(self):
        file_features, file_names = get_extracted_file_features(
            dataset=self.dataset,
        )
        return file_features, file_names

    def ssl_kmeans(self, init="k-means++", n_init="auto"):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)  #
        self.number_topology.append(number_of_clusters)  #
        self_supervised_learning_model = KMeans(
            n_clusters=number_of_clusters,
            random_state=self.random_state,
            init=init,
            n_init=n_init,
        )
        self_supervised_learning_model.fit(embedded_space)

        prototype_responsibilities = self_supervised_learning_model.fit_transform(
            embedded_space
        )

        cluster_labels = self_supervised_learning_model.labels_
        cluster_centers = self_supervised_learning_model.cluster_centers_

        return (
            prototype_responsibilities,
            cluster_labels,
            file_names,
            cluster_centers,
            embedded_space,
        )

    def ssl_fcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.FCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            ord="fro",
            set_U_matrix=None,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_pcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.pcm.PCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            k=1,
            ord="fro",
            set_U_matrix=SSLType.FCM,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_fpcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.fpcm.FPCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            eta=2,
            ord=None,
            set_U_matrix=SSLType.FCM,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_pfcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.pfcm.PFCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            k=1,
            eta=2,
            a=2,
            b=2,
            ord=None,
            set_U_matrix=SSLType.FCM,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_ipcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.ipcm.IPCM1(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=None,
            m_f=2,
            m_p=2,
            k=2,
            ord=None,
            set_centroids=None,
            set_U_matrix=SSLType.FCM,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_ipcm_2(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.ipcm_2.IPCM2(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m_f=2,
            m_p=2,
            ord=None,
            set_U_matrix=SSLType.FCM,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_bgpc(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.bgpc.BGPC(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=100,
            a_f=0.8,
            b_f=0.004,
            ord=None,
            set_centroids=None,
            set_U_matrix=None,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_hcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.hcm.Kmeans(
            data=np.array(embedded_space),
            c=number_of_clusters,
            epsilon=0.01,
            num_inter=100,
            ord="fro",
            set_prototypes=None,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def ssl_kfpcm(self):
        embedded_space, image_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.KFPCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=100,
            m=2,
            sigma=1,
            eta=2,
            ord="fro",
            set_centroids=None,
            set_U_matrix=SSLType.KFCM.value,
            plot_steps=False,
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, image_names

    def get_number_clusters(self, x):
        """

        :param x: embeded space
        :return: number of clusters.
        """
        if isinstance(self.number_cluster, int):
            return self.number_cluster
        if self.number_cluster == "default":
            number_cluster = 2
            return number_cluster
        return None


class Prune(SSL):
    """
    Prune
    params:

    random_state: int:
        Random seed
    prune_fraction: float:
        fraction or percentage of dataset to prune
    prune_mode: str:
        easy , hard and both
    prune_type: bool:
        True or False .Indicating the ceiling or floor of the prune results.


    """

    def __init__(
        self,
        prune_type,
        ssl_type,
        number_cluster,
        random_state,
        dataset,
    ):
        super().__init__(
            number_cluster,
            random_state,
            dataset,
        )
        self.prune_mode = [PruneMode.EASY.value, PruneMode.HARD.value]
        self.prune_type = prune_type
        self.ssl_type = ssl_type
        # self.date_frame_clustered = data_frame_clustered

        match self.ssl_type:
            case SSLType.KMEANS:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                    self.cluster_centers,
                    self.embedded_space,
                ) = self.ssl_kmeans()

            case SSLType.FCM.value:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_fcm()

            case SSLType.KFPCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_kfpcm()

            case SSLType.PCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_pcm()

            case SSLType.FPCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_fpcm()

            case SSLType.PFCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_pfcm()

            case SSLType.IPCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_ipcm()

            case SSLType.IPCM2:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_ipcm_2()

            case SSLType.BGPC:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_bgpc()

            case SSLType.HCM:
                (
                    self.distance_space,
                    self.clustered_labels,
                    self.clustered_file_names,
                ) = self.ssl_hcm()

    def get_number_topologies(self):
        """

        :return: The learned topologies.
        """
        return self.number_topology

    def get_cluster_results(self):
        """

        :return: panda dataframe and populate folders wit the clustering results.
        """
        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]

        file_cluster_df = pd.DataFrame(
            self.clustered_file_names, columns=["file_names"]
        )

        file_cluster_df["cluster_label"] = self.clustered_labels

        return file_cluster_df

    def prune(self, prune_fraction):
        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]

        prune_features = get_prune_set_folder(
            distance_space=self.distance_space,
            prune_fraction=prune_fraction,
            file_names=self.clustered_file_names,
        )
        return prune_features

    def get_prune_features(self, prune_fraction, prune_mode):
        """

        :param prune_fraction:float: fraction of prune
        :param prune_mode: str: 'easy' ,'hard' and 'both'
        :return: prune_set on feature basis
        """
        prune_list, prune_easy, prune_hard = [], [], []
        prune_features = self.prune(prune_fraction=prune_fraction)
        for label, prune_features_cluster in enumerate(prune_features):
            prune_easy.append(
                [
                    list(prune_label)
                    for index, prune_label in enumerate(prune_features_cluster[0])
                ]
            )
            prune_hard.append(
                [
                    list(prune_label)
                    for index, prune_label in enumerate(prune_features_cluster[1])
                ]
            )
        match prune_mode:
            case PruneMode.EASY.value:
                return prune_easy
            case PruneMode.HARD:
                return prune_hard
            case PruneMode.BOTH:
                return prune_easy, prune_hard
            case _:
                raise RuntimeError("get_prune_features: none of the case matches")

    def get_prune_indexes(self, prune_fraction, prune_mode):
        """

        :param prune_fraction:
        :param prune_mode:
        :return:
        """

        prune_index = []
        pruned_features = self.get_prune_features(
            prune_fraction=prune_fraction, prune_mode=prune_mode
        )
        for label in pruned_features:
            prune_index.append(
                [
                    index
                    for prune_specified_feature in label
                    for index, features in enumerate(self.dataset)
                    if np.allclose(features, prune_specified_feature)
                ]
            )
        all_prune_index_in = reduce(add, prune_index)

        all_prune_index_out = [
            index
            for index in list(range(len(self.dataset)))
            if index not in all_prune_index_in
        ]
        return all_prune_index_in, all_prune_index_out

    def get_prune(self, prune_fraction, pruned_mode, prune_type):
        """

        :param prune_fraction:float: fraction of pruning.
            easy is proportional and reverse for hard.
        :param pruned_mode:str: easy or hard
        :param prune_type: str: 'in' ,'out', 'both'.
            'in' for keeps, 'out' for exclusion and 'both' for in and out
        :return: pruned index of dataset
        """
        prune = self.get_prune_indexes(
            prune_fraction=prune_fraction,
            prune_mode=pruned_mode,
        )
        match prune_type:
            case PruneType.IN:
                return prune[0]
            case PruneType.OUT:
                return prune[1]
            case PruneType.BOTH:
                return prune
            case _:
                raise RuntimeError("get_prune: none of the case matches")


if __name__ == "__main__":
    print("import module to use")
