import os
import glob

import numpy as np

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.AlphaFoldDB1K")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class AlphaFoldDB1K(data.ProteinDataset):
    """
    3D protein structures predicted by AlphaFold.
    This dataset covers proteomes of 48 organisms, as well as the majority of Swiss-Prot.

    Statistics:
        See https://alphafold.ebi.ac.uk/download

    Parameters:
        path (str): path to store the dataset
        species_id (int, optional): the id of species to be loaded. The species are numbered
            by the order appeared on https://alphafold.ebi.ac.uk/download (0-20 for model
            organism proteomes, 21 for Swiss-Prot)
        split_id (int, optional): the id of split to be loaded. To avoid large memory consumption
            for one dataset, we have cut each species into several splits, each of which contains
            at most 22000 proteins.
        verbose (int, optional): output verbose level
        **kwargs
    """

    urls = ["https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000000805_243232_METJA_v2_mini1000.tar",]
    md5s = ["da938dfae4fabf6e144f4b5ede5885ec"]
    species_nsplit = [1]
    split_length = 22000

    def __init__(self, path, species_id=0, split_id=0, verbose=1, **kwargs):
        print(f"Loading alphafold dataset species id {species_id}, split {split_id}")
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        species_name = os.path.basename(self.urls[species_id])[:-4]
        if split_id >= self.species_nsplit[species_id]:
            raise ValueError("Split id %d should be less than %d in species %s" %
                            (split_id, self.species_nsplit[species_id], species_name))
        self.processed_file = "%s_%d.pkl.gz" % (species_name, split_id)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            # NOTE: Loading pickle
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            tar_file = utils.download(self.urls[species_id], path, md5=self.md5s[species_id])
            pdb_path = utils.extract(tar_file)
            gz_files = sorted(glob.glob(os.path.join(pdb_path, "*.pdb.gz")))
            pdb_files = []
            index = slice(split_id * self.split_length, (split_id + 1) * self.split_length)
            for gz_file in gz_files[index]:
                pdb_files.append(utils.extract(gz_file))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # NOTE: protein b_factor added
        if hasattr(protein, "b_factor"):
            with protein.residue():
                unique_values, counts = np.unique(protein.atom2residue, return_counts=True)
                cumulative_counts = np.concatenate(([0], np.cumsum(counts)))[:-1]
                protein.residue_b_factor = protein.b_factor[cumulative_counts]
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
