import tensorflow as tf
from tensorflow.keras import layers
from MolGAN.gumbel import GumbelSoftmax

class GNNGenerator(tf.keras.Model):
    def __init__(self, num_atoms, atom_types, bond_types, hidden_dim, latent_dim):
        super(GNNGenerator, self).__init__()
        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.latent_dim = latent_dim

        # Latent vector -> node feature matrix
        self.fc_latent = layers.Dense(units=num_atoms * atom_types, activation='relu')

        # Node feature processing (instead of GCN)
        self.node_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
        ])

        # Output layers for adjacency and node types
        self.dense_adj = layers.Dense(bond_types)  # Per-node edge class prediction
        self.dense_node = layers.Dense(atom_types)  # Per-node atom class prediction

        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)

    def call(self, node_inputs, adj_inputs, training=False):
        batch_size = tf.shape(node_inputs)[0]

        # Create z ~ N(0, 1)
        z = tf.random.normal((batch_size, self.latent_dim))
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)

        # Decode to node feature representation
        x = self.fc_latent(z)
        x = tf.reshape(x, (batch_size, self.num_atoms, self.atom_types))  # [B, N, F]

        h = self.node_mlp(x)  # [B, N, H]

        # Predict edge and node logits
        edge_logits = self.dense_adj(h)   # [B, N, bond_types]
        node_logits = self.dense_node(h)  # [B, N, atom_types]

        return edge_logits, node_logits, mu, logvar
