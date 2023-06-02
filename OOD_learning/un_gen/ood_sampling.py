"""BROOD sampling method for OOD sampling"""

# Author: Lennert Van der Schraelen <lennert.vanderschraelen@vlerick.com>

import numpy as np
import scipy
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist


class BaseBROOD:
    """Base class for static methods"""

    @staticmethod
    def cosine_matrix(mat_vec_1, mat_vec_2):

        """
        Determines the angle between two matrices of vectors.

        Parameters
        -------

        mat_vec_1: numpy array
            The first matrix of vectors.

        mat_vec_2: : numpy array
            The second matrix of vectors.

        Returns
        -------

        theta: float
            A matrix of angles between the vectors in mat_vec_1 and mat_vec_2.

        """

        D = np.einsum('ij,nj->in', mat_vec_1, mat_vec_2)
        norm_vec = np.linalg.norm(mat_vec_1.T, axis=0)
        norm_vec_2 = np.linalg.norm(mat_vec_2.T, axis=0)

        a = np.dot(norm_vec[:, np.newaxis], norm_vec_2[np.newaxis, :])
        val = D / a
        np.putmask(val, val > 1, 1)
        np.putmask(val, val < -1, -1)

        theta = np.arccos(val)
        return theta

    @staticmethod
    def direction_sampling(n_dir_samp, number_of_dir_m, d):

        """
        Samples random data points on the surface of a hypersphere (by sampling directions).
        We ensure that the data points are approximately evenly spaced

        Parameters
        -------

        n_dir_samp: int
            Number of directions too choose from.

        number_of_dir_m: int
            Number of final directions.

        d: int
            The dimension of the hypersphere.

        Returns
        -------

        directions: numpy array
            Matrix that consist of directions.

        min_angle: min_angle
            The minimal angle between the directions in the matrix of directions.


        References
        -------
        Simon, C. (2015). Generating uniformly distributed numbers on a sphere.
        http://corysimon.github.io/articles/uniformdistn-on-sphere/

        """

        directions = np.random.randn(d, 1).T

        for i in range(number_of_dir_m - 1):
            # A high n_dir_samp will enhance the creation of a mesh grid.
            # To use Mitchell's best candidate algorithm to enhance 'blue noise sampling' use the commented line
            
            directions_prop = np.random.randn(d, n_dir_samp).T
            # directions_prop = np.random.randn(d, n_dir_samp*(i+1)).T    

            angles = BaseBROOD.cosine_matrix(directions, directions_prop)
            index = np.argmax(np.min(angles, axis=0))
            directions = np.append(directions, np.expand_dims(directions_prop[index, :], axis=0), axis=0)

        angles = BaseBROOD.cosine_matrix(directions, directions)
        delete_diag = angles[~np.eye(angles.shape[0], dtype=bool)].reshape(
            angles.shape[0], -1)
        min_angle = np.min(delete_diag)

        return directions, min_angle

    @staticmethod
    def mahalob_dist(mat_vec_1, scaler, mat_vec_2=0):

        """
        Calculates the (vector-wise) Mahalanobis distance between two matrices of vectors.

        Parameters
        -------

        mat_vec_1: numpy array
            The first matrix of vectors.

        scaler: numpy array
            The covariance matrix of the Mahalanobis distance.

        mat_vec_2: : numpy array, default = 0
            The second matrix of vectors.

        Returns
        -------

        D_mahal: numpy array
            Vector that consist of the Mahalanobis distances between the vectors in mat_vec_1 and mat_vec_2.

        """

        D_mahal = np.sqrt(np.einsum('nj,jk,nk->n', mat_vec_1 - mat_vec_2, scaler, mat_vec_1 - mat_vec_2))

        return D_mahal

    @staticmethod
    def direction_scaling(directions, scaler):

        """
        Scales the directions. (For instance to represent data points on an ellipse.)

        Parameters
        -------

        directions: numpy array
            Matrix of initial directions.

        scaler: numpy array
            The trace of the covariance matrix of the Mahalanobis distance.

        Returns
        -------

        mat_vec: numpy array
            Matrix of directions.

        """

        mat_vec = np.expand_dims(np.sqrt(1 / scaler), axis=-1) * directions.T
        D = BaseBROOD.mahalob_dist(mat_vec.T, np.diag(scaler))
        mat_vec /= D
        mat_vec = mat_vec.T
        return mat_vec

    @staticmethod
    def mahalob_dist_matrix(mat_vec_1, mat_vec_2, array_scalers, y_1=None):

        """
        Calculates the Mahalanobis distance between two matrices of vectors, dependent on the instance

        Parameters
        -------

        mat_vec_1: numpy array
            The first matrix of vectors.

        mat_vec_2: : numpy array
            The second matrix of vectors.

        array_scalers: numpy array
            Array that contains the diagonal of the covariance matrices to calculate the Mahalanobis distance

        y_1: numpy array
            labels of the vectors in mat_vec_1

        Returns
        -------

        D_mahal: numpy array
            Matrix that consist of the (cross) Mahalanobis distances between the vectors in mat_vec_1 and mat_vec_2.

        """

        a = mat_vec_1[:, np.newaxis, :]

        diff = a - mat_vec_2

        if y_1 is None:

            b = np.vstack([array_scalers.min(axis=0)] * np.shape(diff)[0])
            A = np.einsum('il,ijl->ijl', b, diff)

        else:

            a = array_scalers[y_1]
            A = np.einsum('il,ijl->ijl', a, diff)

        B = (diff).T

        D_mahal = np.sqrt(np.einsum('ijk,kji->ij', A, B))

        return D_mahal


class BROOD:
    """Boundary Regularising Out Of Distribution sampling technique

    Parameters
    -------

    number_of_dir_m: int, default 30
        Maximum number of directions from each queried data point.

    dist_id_ood: float, default 4
        The distance between synthetic samples and the data point it is queried from.

    query_strategy: 2-dimensional list or numpy array, default None
        The first element of the list or array captures the method to determine the data points we will query.
        If the first element of the list or array equals 'outlying', we make use of isolation forests to
         determine the outlying score of the data points. In this case, the second element represents
         the fraction of data points of the total data-set we will sample from. Data points with the highest outlying
         score are sampled first.
        If the first element of the list or array equals 'label', the second element represents
         the label of the data points we will sample from.

    max_ood = int, default None
        If integer, imposes a maximum number of OOD samples.

    simple = bool, default True
        Captures the label strategy of the OOD sampler.
        If True, the label of the OOD sample is equal to the label from the data point it is sampled from
        If False, the label of the OOD sample depends on the neighbourhood of the data point it is sampled from

    h_strategy = int default 0
        Captures whether we make a distinction between instances, labels
        If 0, we use one scaler for all data: Rule of thumb
        If 1, we use scalers depending on label: label dependent Rule of thumb
        If 2, we use scalers depending on instance: KNN bandwith selection

    n_dir_samp: int, default 90
        Needs to be larger than or equal to 1. Used to sample more uniformly on the surface of a hypersphere.
        In every iteration we keep the direction furthest away from the existing directions.


    Returns
    -------

    X_ood: numpy array
        The sampled boundary OOD data.

    y_ood: numpy array
        The labels of the sampled boundary OOD data. The labels equal the labels of the
        data points they are sampled from.


    """

    def __init__(self, number_of_dir_m=30, dist_id_ood=4, query_strategy=None, max_ood=None,
                 simple=True, h_strategy=0, n_dir_sample=90, equal=False):
        if query_strategy is None:
            query_strategy = ['outlying', 1]
        self.number_of_dir_m = number_of_dir_m
        self.dist_id_ood = dist_id_ood
        self.max_ood = max_ood
        self.query_strategy = query_strategy
        self.simple = simple
        self.h_strategy = h_strategy
        self.n_dir_samp = n_dir_sample
        self.equal = equal

    def fit_resample(self, X, y, seed=None):

        print('start ood sampling')

        # Step 1: Initialisation

        if ((seed != None) and (isinstance(seed, int))):
            np.random.seed(seed)

        eps = 0.00001
        jitter = 0.00001

        try:
            if self.query_strategy[0] == 'outlying':

                clf = IsolationForest(random_state=seed)
                clf.fit(X)
                scores = -clf.score_samples(X)  # sklearn calculates the opposite of the outlying score
                number = int(
                    self.query_strategy[1] * X.shape[0])  # we sample from all data if self.query_strategy[1] = 1
                arr1inds = scores.argsort()[::-1]  # We start to sample from the most outlying data, only relevant if
                # max_ood is reached and/or self.query_strategy[1] != 1

            elif self.query_strategy[0] == 'label':

                number = (y == self.query_strategy[1]).sum() # we sample from all data with label self.query_strategy[1]
                lab = np.where(y == self.query_strategy[1],1,0)
                arr1inds = lab.argsort()[::-1]

            else:
                raise

            sorted_X = X[arr1inds]
            sorted_y = y[arr1inds]
            X_query = sorted_X[:number]
            y_query = sorted_y[:number]

        except Exception:
            raise SyntaxError("Please provide a suitable syntax for the query strategy parameter")

        if self.h_strategy == 2:
            self.knn = round(2 * X.shape[0] ** (4 / (4 + X.shape[1])))

            dist = cdist(sorted_X, sorted_X)
            a = np.partition(dist, self.knn, axis=0)[self.knn]
            dist_inf = dist.copy()
            np.fill_diagonal(dist_inf, np.inf)
            result = [np.where(row <= a[i]) for i, row in enumerate(dist_inf)]
            width = np.array([np.mean(a[result[l]]) for l in range(X.shape[0])])
            width = width ** 2

            width[width == 0] = 0 + jitter
            array_scalers = 1 / width

            array_scalers = np.vstack([array_scalers] * np.shape(X)[1]).T

        if self.h_strategy == 1:

            array_scalers = np.zeros(np.shape(X))
            for y_label in np.unique(y):
                X_y = X[y == y_label]
                sigma_y_1 = np.std(X_y, axis=0)
                width_y = ((4 / (2 + X_y.shape[1])) ** (1 / (4 + X_y.shape[1]))) * (X_y.shape[0] ** (
                        -1 / (4 + X_y.shape[1]))) * sigma_y_1
                width_y = width_y ** 2

                width_y[width_y == 0] = 0 + jitter

                array_scalers[sorted_y == y_label, :] = 1 / width_y

        if self.h_strategy == 0:
            sigma = np.std(X, axis=0)
            width = ((4 / (2 + X.shape[1])) ** (1 / (4 + X.shape[1]))) * (X.shape[0] ** (
                    -1 / (4 + X.shape[1]))) * sigma
            width = width ** 2

            width[width == 0] = 0 + jitter

            array_scalers = np.full(np.shape(X), 1 / width)

        # Step 2: Directions

        directions, theta_min = BaseBROOD.direction_sampling(n_dir_samp=self.n_dir_samp,
                                                             number_of_dir_m=self.number_of_dir_m, d=X.shape[1])

        # Step 3: dist_ood & Neighbourhood construction

        # dist_ood

        # We take dist_ood equal to one tenth of the minimum distance between
        # artificial OOD samples generated from the same ID data point.

        # Define $\tilde{\fat{v}} = \text{diag}(\hat{h}_{1}, \ldots, \hat{h}_{d)} \cdot \fat{v}$
        # One can deduce that  $||\fat{v}||_{euclid} = ||\tilde{\fat{v}}||_{(\text{mahal},\hat{H})}$}
        # (hence independent of scaler; we just take the unity scaler)

        scaler_dist = np.ones(np.shape(directions)[1])
        directions_sc = BaseBROOD.direction_scaling(directions, scaler=scaler_dist) #normalizes
        distances_sc = BaseBROOD.mahalob_dist_matrix(directions_sc, directions_sc, np.ones(np.shape(directions)),
                                                     np.arange(1))             #distances
        distances_sc *= self.dist_id_ood
        distances_diag_cleaned = distances_sc[~np.eye(distances_sc.shape[0], dtype=bool)].reshape(
            distances_sc.shape[0], -1)

        dist_ood = np.min(distances_diag_cleaned) / 10

        # Neighbourhood construction

        # Calculate the maximal Euclidean norm to determine the maximum surface of the hypersphere
        # only used if equal is True
        if self.equal == True:
            directions_max = BaseBROOD.direction_scaling(directions, scaler=array_scalers[:number].min(axis=0))
            directions_max *= self.dist_id_ood
            max_norm = np.mean(np.linalg.norm(directions_max, axis=1))

            sphere = ((2 * np.pi ** (np.shape(directions)[1] / 2)) / (scipy.special.gamma(np.shape(directions)[1] / 2)))
            sphere_sur_max = sphere * (max_norm ** (np.shape(directions)[1] - 1))


        # distance of id data to the queried points, equals 1 if point in ngbh of queried point, 0 otherwise

        id_distances = BaseBROOD.mahalob_dist_matrix(sorted_X, X_query, array_scalers, None).T
        id_ngbh = np.where(id_distances <= self.dist_id_ood + self.dist_id_ood + eps, 1, 0)

        # Only if simple is False
        # Determine a ngbh to assign a suitable label to the artificial samples

        if self.simple == False:
            n_y = float(y.size)
            prior_dict = {}
            for y_label in np.unique(y):
                pr_vec = np.where(y == y_label, 1, 0)
                prior_dict[y_label] = np.count_nonzero(pr_vec) / n_y
            id_ng = np.where(id_distances <= 1 + eps, 1, 0)

        # The two step ngbh is only calculated for the instances we will query in the future
        # Only applies for the instances we query of the same class

        id_ngbh_2_step = np.where(id_distances[:, :number] <= self.dist_id_ood + self.dist_id_ood
                                  + dist_ood + eps, 1, 0)
        lower_index = np.tril_indices(number)
        id_ngbh_2_step[lower_index] = 0

        # In the simple case, the label of artificial samples equals the label of the queried data point
        # Hence in this case, for the two step ngbh, only the data points of the same label are important

        if self.simple == True:
            b = (y_query[:, np.newaxis] == y_query).astype(int)
            id_ngbh_2_step = np.multiply(id_ngbh_2_step, b)

        X_data = sorted_X.copy()
        y_data = sorted_y.copy()

        # Delete all data not in the ngbh of the queried points, of course not relevant if one queries all points

        X_data = X_data[~(id_ngbh == 0).all(axis=0), :]
        y_data = y_data[~(id_ngbh == 0).all(axis=0)]
        id_ood_ngbh = id_ngbh.copy()
        id_ood_ngbh = id_ood_ngbh[:, ~(id_ngbh == 0).all(axis=0)]

        if self.simple == False:
            id_ng = id_ng[:, ~(id_ngbh == 0).all(axis=0)]

        n_orig_data = y_data.shape[0]
        number_of_ood = 0
        point_sample_hist = np.array([], dtype='int32')

        # Step 4: Query instances and get its neighbourhoods

        for i, x_id in enumerate(X_query):

            queried_y = y_query[i]

            # Get the ngbh of all the current data

            x_id_ngbh = X_data[:n_orig_data][id_ood_ngbh[i, :][:n_orig_data] == 1]
            x_ood_ngbh = X_data[n_orig_data:][id_ood_ngbh[i, :][n_orig_data:] == 1]

            # Get the two step nghb of the queried instance of the id data that would be queried in the future

            x_id_ngbh_2_step = X_query[id_ngbh_2_step[i, :] == 1]
            y_id_ngbh_2_step = y_query[id_ngbh_2_step[i, :] == 1]

            y_id_ngbh_ind = np.where(id_ood_ngbh[i, :][:n_orig_data] == 1)[0]
            y_ood_ngbh_ind = point_sample_hist[id_ood_ngbh[i, :][n_orig_data:] == 1]
            y_id_ngbh_2_step_ind = np.where(id_ngbh_2_step[i, :] == 1)[0]

            if self.simple == False:
                y_id_ng = y_data[:n_orig_data][id_ng[i, :] == 1]

            # This array represents the two step ngbh of the queried instance

            id_ngbh_2_step_query = np.squeeze(np.array(np.nonzero(id_ngbh_2_step[i, :] == 1)), axis=0)

            # Step 5: Investigate possible directions, keep allowed samples

            # Get the directions, check if distance to current data is large enough
            # Put them in the database

            directions_nd = BaseBROOD.direction_scaling(directions, scaler=array_scalers[i, :])
            directions_nd *= self.dist_id_ood

            # If equal is True. If the Euclidean norms of the directions differs
            # for different instances, ensure an equal distribution of OOD samples, incorporate the
            # surface of a sphere
            if self.equal == True:
                sphere_sur = sphere * (
                            np.mean(np.linalg.norm(directions_nd, axis=1)) ** (np.shape(directions_nd)[1] - 1))
                b = np.int((sphere_sur / sphere_sur_max) * np.shape(directions_nd)[0])
                directions_nd = directions_nd[:b, :]

            x_samp = x_id + directions_nd

            # Check if the the new points are not too close to the other points in the ngbh

            dist = BaseBROOD.mahalob_dist_matrix(x_id_ngbh, x_samp, array_scalers, y_id_ngbh_ind).T

            keep_ngbh = dist.min(axis=1)
            x_samp_keep = x_samp[keep_ngbh >= self.dist_id_ood - eps]

            if (x_ood_ngbh.shape[0] != 0) & (x_samp_keep.shape[0] != 0):

                bidirectional_array_scalers = np.minimum(array_scalers,array_scalers[np.newaxis,i, :])
                dist_2 = BaseBROOD.mahalob_dist_matrix(x_ood_ngbh, x_samp_keep, bidirectional_array_scalers,
                                                       y_ood_ngbh_ind).T

                # dist_2 = BaseBROOD.mahalob_dist_matrix(x_ood_ngbh, x_samp_keep, array_scalers,
                #                                        y_ood_ngbh_ind).T

                if dist_2.size != 0:
                    keep_ngbh_2 = dist_2.min(axis=1)
                    x_samp_keep = x_samp_keep[(keep_ngbh_2 >= dist_ood - eps)]

            number_of_samples = np.shape(x_samp_keep)[0]

            if self.max_ood is not None:
                number_of_ood += number_of_samples
                if number_of_ood >= self.max_ood:
                    break

            X_data = np.vstack((X_data, x_samp_keep))

            # If simple is True, use the label from the queried point
            # If simple is False, exploit the labels of close data points

            if self.simple == True:
                y_data = np.concatenate((y_data, np.full((number_of_samples,), queried_y)))
            if self.simple == False:
                prior_dict_id = {}
                max_sc = {}
                for y_label in np.unique(y):
                    n_y_id = int(float(y_id_ng.size))
                    pr_vec_id = np.where(y_id_ng == y_label, 1, 0)
                    prior_dict_id[y_label] = np.count_nonzero(pr_vec_id) / n_y_id
                    max_sc[y_label] = prior_dict_id[y_label] / prior_dict[y_label]
                max_label = max(max_sc, key=max_sc.get)
                y_data = np.concatenate((y_data, np.full((number_of_samples,), max_label)))

            # Step 6: Check if there exists data points in the 2-step ngbh of the same class and ensure
            # sufficient distance in the future

            # We create an array that captures the new ngbh points for the queried data

            array_append = np.zeros((X_query.shape[0], x_samp_keep.shape[0]))

            # Check distance of new artificial points to the 2 step ngbh
            # Find the ngbh of the new points and put indices in id_ood_ngbh

            if (x_samp_keep.shape[0] != 0) & (np.shape(id_ngbh_2_step_query)[0] != 0):

                # Calculate the Mahalanobis distance from the 2 step ngbh to the new points

                dist = BaseBROOD.mahalob_dist_matrix(x_id_ngbh_2_step, x_samp_keep,
                                                     array_scalers, y_id_ngbh_2_step_ind).T

                dist_keep = np.where(dist <= self.dist_id_ood + dist_ood + eps, 1, 0).astype(int)

                # In the simple case, we already deleted the points in the two step ngbh with a different label
                # Now, as we also determined the labels of the new artificial points in the other case,
                # we can nullify the points with another label

                if self.simple == False:
                    dist_keep = np.where(y_id_ngbh_2_step == queried_y, dist_keep, 0).astype(int)

                for j, column in enumerate(array_append.T):
                    # Extracts the indices of the new artificial data that fall in the ngbh of queried points X_query
                    # on these indices, put 1 in the column of array_append
                    ind = id_ngbh_2_step_query[dist_keep[j, :] == 1]
                    np.add.at(column, ind, 1)
                    array_append[:, j] = column

            # Stack this array with the ngbh array
            id_ood_ngbh = np.hstack((id_ood_ngbh, array_append))

            point_sample_hist = np.append(point_sample_hist, np.full(x_samp_keep.shape[0], i))

        X_ood = X_data[n_orig_data:]
        # X_id = X_data[-n_orig_data:]
        y_ood = y_data[n_orig_data:]

        print('done with ood sampling')
        print('the number of ood samples equals ' + str(len(y_ood)))
        print('the number of id samples equals ' + str(len(y)))

        return X_ood, y_ood
