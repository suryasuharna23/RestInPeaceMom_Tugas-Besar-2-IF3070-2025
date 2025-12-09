import numpy as np

class Node:
    # COnstructor node
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None): 
        # Atribut Decision Node
        self.feature_index = feature_index # Index fitur yang dipakai split
        self.threshold = threshold # Threshold split fitur
        self.left = left # Subnode kiri
        self.right = right # Subnode kanan
        self.info_gain = info_gain # Information gain dari split ini
        
        # Atribut Leaf Node
        self.value = value # Hasi prediksi

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        #Set Hyperparameter
        self.min_samples_split = min_samples_split # Jumlah data minimal untuk split node
        self.max_depth = max_depth # Kedalaman maks pohon
        self.root = None

    def fit(self, X, y):
        X = np.array(X) #Fitur (matriks)
        y = np.array(y) # Label (vektor)
        self.root = self._build_tree(np.array(X), np.array(y))

    def _build_tree(self, X, y, curr_depth=0):
        # Build pohon
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Cek stopping criteria
        # Jika depth > max, atau data < min_split, atau node sudah pure)
        if (curr_depth >= self.max_depth or 
            num_labels == 1 or 
            num_samples < self.min_samples_split):
            
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value) # Kembalikan leaf node

        # Find best split
        best_split = self._get_best_split(X, y, num_features)

        # Jika best splitnya menguntungkan (Info Gain > 0), split
        if best_split["info_gain"] > 0:
            # Rekursi untuk subnode kiri
            left_subtree = self._build_tree(best_split["dataset_left"]["X"], 
                                            best_split["dataset_left"]["y"], 
                                            curr_depth + 1)
            # Rekursi untuk subnode kanan
            right_subtree = self._build_tree(best_split["dataset_right"]["X"], 
                                             best_split["dataset_right"]["y"], 
                                             curr_depth + 1)
            
            return Node(feature_index=best_split["feature_index"], 
                        threshold=best_split["threshold"], 
                        left=left_subtree, 
                        right=right_subtree, 
                        info_gain=best_split["info_gain"])

        # Jika tidak bisa split lagi, return Leaf Node
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    # --- Helper Functions ---
    
    def _get_best_split(self, X, y, num_features):
        # Mencari fitur dan threshold terbaik untuk split
        best_split = {}
        max_info_gain = -float("inf")

        # Loop semua fitur
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            
            # Ambil nilai unik sebagai kandidat threshold
            unique_values = np.unique(feature_values)

            # Jika variasi nilainya terlalu banyak, pakai persentil saja
            if len(unique_values) > 100:
                percentiles = np.linspace(0, 100, 12)[1:-1] # hindari 0% dan 100%
                possible_thresholds = np.percentile(feature_values, percentiles)
            else:
                possible_thresholds = unique_values

            # Loop semua kemungkinan threshold di fitur tersebut
            for threshold in possible_thresholds:
                
                # Coba lakukan split
                dataset_left_idxs, dataset_right_idxs = self._split_data(feature_values, threshold)
                
                # Jika split tidak valid (salah satu sisi kosong), skip
                if len(dataset_left_idxs) == 0 or len(dataset_right_idxs) == 0:
                    continue
                
                # Ambil target variable (y) hasil pecahan
                y_left = y[dataset_left_idxs]
                y_right = y[dataset_right_idxs]
                
                # Hitung Information gain
                gain = self._information_gain(y, y_left, y_right)

                # Jika gain > max gain, simpan
                if gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = {"X": X[dataset_left_idxs], "y": y_left}
                    best_split["dataset_right"] = {"X": X[dataset_right_idxs], "y": y_right}
                    best_split["info_gain"] = gain
                    max_info_gain = gain
        
        # Penanganan jika tidak ada split yang menghasilkan gain positif
        if "info_gain" not in best_split:
            best_split["info_gain"] = 0
            
        return best_split

    def _split_data(self, X_column, split_thresh):
        # Membagi data jadi kiri dan kanan berdasarkan threshold
        # np.argwhere mengembalikan nomor baris yang memenuhi syarat
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_leaf_value(self, y):
        # Menentukan nilai label
        Y = list(y)
        return max(Y, key=Y.count)
    
    def _gini(self, y):
        # Menghitung gini impurity dari suatu node(0 = Murni, 0.5 = Kotor)
        m = len(y)
        
        # Hitung probabilitas tiap kelas
        counts = np.bincount(y)
        probs = counts / m
        
        # Rumus: 1 - sum(p^2)
        return 1 - np.sum(probs ** 2)
    
    def _information_gain(self, y, y_left, y_right):
        # Menghitung info gain, semakin besar semakin bagus
        leftweight = len(y_left) / len(y) # Weight kiri
        rightweight = len(y_right) / len(y) # Weight kanan
        
        # Info gain = Gini awal - (Bobot kiri* Gini subnode kiri + Bobot kanan*Gini subnode kanan)
        return self._gini(y) - (leftweight * self._gini(y_left) + rightweight * self._gini(y_right))
    
    def predict(self, X):
        # Fungsi utama untuk memprediksi label, X dataframe
        # Konversi ke numpy array
        X = np.array(X)
        
        # Loop setiap baris data (x) dan cari prediksinya di pohon
        predictions = [self._make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def _make_prediction(self, x, tree_node):
        # Fungsi rekursif untuk menelusuri pohon
        # Jika node saat ini adalah leaf, return nilainya (label hasil prediksi)
        if tree_node.value is not None:
            return tree_node.value
        
        # Ambil nilai fitur dari data
        feature_val = x[tree_node.feature_index]
        
        # Bandingkan dengan threshold 
        if feature_val <= tree_node.threshold:
            # Belok kiri
            return self._make_prediction(x, tree_node.left)
        else:
            # Belok kanan
            return self._make_prediction(x, tree_node.right)
    
    def print_tree(self, tree=None, indent=" "):
        # Visualisasi sederhana pohon
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)